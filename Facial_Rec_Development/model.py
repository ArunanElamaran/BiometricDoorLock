"""
Lightweight Face Recognition Model
Target: < 1GB RAM, 5 identities
Architecture: Fixed-size face image + small CNN

Requirements:
    pip install torch torchvision numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from typing import Optional

# Optional: use PIL + torchvision for image load/transform (no opencv required for base path)
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class ImagePreprocessor:
    """
    Converts raw face images to fixed-size tensors.
    Expects images that are already face crops, or full frames (will resize).
    """

    def __init__(
        self,
        image_size: int = 112,  # H and W (square)
        normalize: bool = True  # ImageNet-style or [0,1]
    ):
        self.image_size = image_size
        self.normalize = normalize
        self._transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        # ImageNet mean/std if normalize
        self._mean = (0.485, 0.456, 0.406)
        self._std = (0.229, 0.224, 0.225)

    def load_image(self, path: str) -> torch.Tensor:
        """Load image file (supports jpg, png, etc.) as RGB tensor [1, 3, H, W]."""
        img = Image.open(path).convert("RGB")
        return self._transform(img).unsqueeze(0)  # [1, 3, H, W]

    def __call__(self, image_path: str) -> torch.Tensor:
        """Full preprocessing pipeline: load, resize, to tensor, optional normalize."""
        x = self.load_image(image_path)
        if self.normalize:
            x = TF.normalize(x, self._mean, self._std)
        return x  # Shape: [1, 3, image_size, image_size]


class LightweightFaceNet(nn.Module):
    """
    Small CNN for face / identity classification.

    Architecture designed for:
    - Small model size, low inference memory
    - 5 (or N) identity classification

    Input: [batch, 3, image_size, image_size]
    """

    def __init__(
        self,
        image_size: int = 112,
        num_persons: int = 5,
        embedding_dim: int = 64
    ):
        super().__init__()
        self.num_persons = num_persons
        self.embedding_dim = embedding_dim

        # Convolutional feature extractor
        # Input: [batch, 3, 112, 112]
        self.conv_layers = nn.Sequential(
            # Block 1: [3, 112, 112] -> [16, 56, 56]
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),

            # Block 2: [16, 56, 56] -> [32, 28, 28]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),

            # Block 3: [32, 28, 28] -> [64, 14, 14]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),

            # Block 4: [64, 14, 14] -> [64, 7, 7]
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(embedding_dim, num_persons)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Args:
            x: Face image [batch, 3, image_size, image_size]
            return_embedding: If True, also return the embedding vector

        Returns:
            logits: [batch, num_persons]
            embedding (optional): [batch, embedding_dim]
        """
        features = self.conv_layers(x)
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        emb = self.embedding(pooled)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embedding (useful for verification)"""
        _, emb = self.forward(x, return_embedding=True)
        return emb


class FaceRecognitionSystem:
    """
    Complete face recognition system.
    Handles training, inference, and model persistence.
    """

    def __init__(
        self,
        num_persons: int = 5,
        image_size: int = 112,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = ImagePreprocessor(image_size=image_size)
        self.model = LightweightFaceNet(
            image_size=image_size,
            num_persons=num_persons
        ).to(self.device)
        self.person_names: list[str] = []

    def prepare_dataset(self, data_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load dataset from directory structure:

        data_dir/
            person_name/
                img1.jpg
                img2.jpg
                ...
            another_person/
                ...

        Returns:
            features: [N, 3, image_size, image_size]
            labels: [N]
        """
        data_path = Path(data_dir)
        features = []
        labels = []

        person_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        self.person_names = [d.name for d in person_dirs]

        print(f"Found {len(person_dirs)} persons: {self.person_names}")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for person_idx, person_dir in enumerate(person_dirs):
            image_files = [
                f for f in person_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            image_files = sorted(image_files)
            print(f"  {person_dir.name}: {len(image_files)} samples")

            for image_file in image_files:
                try:
                    img_tensor = self.preprocessor(str(image_file))
                    features.append(img_tensor)
                    labels.append(person_idx)
                except Exception as e:
                    print(f"    Warning: Failed to load {image_file.name}: {e}")

        features = torch.cat(features, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        return features, labels

    def train(
        self,
        data_dir: str,
        epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 0.001,
        validation_split: float = 0.2
    ):
        """Train the model on the dataset"""

        print("Loading dataset...")
        features, labels = self.prepare_dataset(data_dir)

        n_samples = len(features)
        indices = torch.randperm(n_samples)
        n_val = int(n_samples * validation_split)

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_features = features[train_indices].to(self.device)
        train_labels = labels[train_indices].to(self.device)
        val_features = features[val_indices].to(self.device)
        val_labels = labels[val_indices].to(self.device)

        print(f"Training samples: {len(train_features)}, Validation: {len(val_features)}")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.CrossEntropyLoss()

        # Data augmentation (random horizontal flip + small brightness/contrast)
        def augment(x: torch.Tensor) -> torch.Tensor:
            if not self.model.training:
                return x
            x = x.clone()
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=(-1,))
            if torch.rand(1).item() > 0.5:
                brightness = 0.8 + 0.4 * torch.rand(1).item()
                x = x * brightness
            return x

        best_val_acc = 0
        print("\nStarting training...")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0

            perm = torch.randperm(len(train_features))
            for i in range(0, len(train_features), batch_size):
                batch_idx = perm[i : i + batch_size]
                batch_x = train_features[batch_idx]
                batch_y = train_labels[batch_idx]
                batch_x = augment(batch_x)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_x)
                train_correct += (logits.argmax(1) == batch_y).sum().item()

            scheduler.step()
            train_loss /= len(train_features)
            train_acc = train_correct / len(train_features)

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_features)
                val_loss = criterion(val_logits, val_labels).item()
                val_acc = (val_logits.argmax(1) == val_labels).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save("best_model.pt")

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:3d}: "
                    f"Train Loss={train_loss:.4f}, Acc={train_acc:.2%} | "
                    f"Val Loss={val_loss:.4f}, Acc={val_acc:.2%}"
                )

        print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2%}")
        self.load("best_model.pt")

    def predict(self, image_path: str) -> tuple[str, float, dict]:
        """
        Predict identity from image file.

        Returns:
            person_name: Predicted person
            confidence: Prediction confidence (0-1)
            all_probs: Dict of person_name -> probability
        """
        self.model.eval()

        with torch.no_grad():
            img = self.preprocessor(image_path)
            img = img.to(self.device)
            logits = self.model(img)
            probs = F.softmax(logits, dim=1).squeeze()

            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

            all_probs = {
                name: probs[i].item()
                for i, name in enumerate(self.person_names)
            }

        return self.person_names[pred_idx], confidence, all_probs

    def get_face_embedding(self, image_path: str) -> np.ndarray:
        """Extract face embedding for verification tasks"""
        self.model.eval()

        with torch.no_grad():
            img = self.preprocessor(image_path).to(self.device)
            embedding = self.model.get_embedding(img)

        return embedding.cpu().numpy().squeeze()

    def verify_face(
        self,
        image_path: str,
        reference_embedding: np.ndarray,
        threshold: float = 0.7
    ) -> tuple[bool, float]:
        """
        Verify if image matches a reference face embedding.

        Returns:
            is_match: True if similarity > threshold
            similarity: Cosine similarity score
        """
        test_embedding = self.get_face_embedding(image_path)
        similarity = np.dot(test_embedding, reference_embedding) / (
            np.linalg.norm(test_embedding) * np.linalg.norm(reference_embedding)
        )
        return similarity > threshold, float(similarity)

    def save(self, path: str):
        """Save model and metadata"""
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "person_names": self.person_names,
                "num_persons": self.model.num_persons,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model and metadata"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.person_names = checkpoint["person_names"]
        print(f"Model loaded from {path}")

    def memory_usage(self) -> dict:
        """Estimate memory usage"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())

        return {
            "model_params": f"{param_size / 1024:.1f} KB",
            "model_buffers": f"{buffer_size / 1024:.1f} KB",
            "total_model": f"{(param_size + buffer_size) / 1024:.1f} KB",
            "estimated_inference_ram": "~50-100 MB",
        }


def print_model_info(model: LightweightFaceNet):
    """Print model architecture summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 50)
    print("Model Summary")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024:.1f} KB (float32)")
    print("=" * 50)


if __name__ == "__main__":
    system = FaceRecognitionSystem(num_persons=5)
    print_model_info(system.model)
    print("\nMemory usage:")
    for k, v in system.memory_usage().items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 50)
    print("Usage Instructions")
    print("=" * 50)
    print("""
1. Organize your data:
   data/
       person_1/
           img1.jpg
           img2.jpg
           ...
       person_2/
           ...

2. Train:
   system = FaceRecognitionSystem(num_persons=5)
   system.train('data/', epochs=50)
   system.save('face_model.pt')

3. Predict:
   system.load('face_model.pt')
   person, confidence, probs = system.predict('test_face.jpg')
   print(f"Person: {person} ({confidence:.1%} confidence)")
""")
