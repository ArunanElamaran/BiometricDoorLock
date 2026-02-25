"""
Face Recognition Model
VGG-style CNN for face / identity classification and embedding.
Expects input size 224x224; use ImagePreprocessor with image_size=224.

Requirements:
    pip install torch torchvision numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from typing import Optional

import torchvision.transforms as T
import torchvision.transforms.functional as TF

from ImageProcessor import ImagePreprocessor


def _make_conv_block(in_c: int, out_c: int, num_convs: int) -> nn.Sequential:
    layers = []
    for i in range(num_convs):
        layers += [
            nn.ZeroPad2d(1),
            nn.Conv2d(in_c if i == 0 else out_c, out_c, kernel_size=3),
            nn.ReLU(inplace=True),
        ]
    layers.append(nn.MaxPool2d(2, stride=2))
    return nn.Sequential(*layers)


class LightweightFaceNet(nn.Module):
    """
    VGG-style face network (input 224x224).
    Same interface as before: num_persons, embedding_dim, forward(), get_embedding().
    """

    def __init__(
        self,
        image_size: int = 224,
        num_persons: int = 5,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.num_persons = num_persons
        self.embedding_dim = embedding_dim

        # VGG-style backbone: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.conv_blocks = nn.Sequential(
            _make_conv_block(3, 64, 2),    # 224 -> 112
            _make_conv_block(64, 128, 2),  # 112 -> 56
            _make_conv_block(128, 256, 3), # 56 -> 28
            _make_conv_block(256, 512, 3), # 28 -> 14
            _make_conv_block(512, 512, 3), # 14 -> 7
        )

        # Head: 7x7x512 -> 4096 -> 4096 -> embed -> num_persons
        self.head = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.embedding_proj = nn.Sequential(
            nn.Linear(4096, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(embedding_dim, num_persons)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Args:
            x: Face image [batch, 3, 224, 224]
            return_embedding: If True, also return the embedding vector

        Returns:
            logits: [batch, num_persons]
            embedding (optional): [batch, embedding_dim]
        """
        features = self.conv_blocks(x)       # [B, 512, 7, 7]
        head_out = self.head(features)      # [B, 4096, 1, 1]
        head_flat = head_out.view(head_out.size(0), -1)
        emb = self.embedding_proj(head_flat)
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
        image_size: int = 224,
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
        """Train the model on the dataset using triplet loss on embeddings."""

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

        triplet_criterion = nn.TripletMarginLoss(margin=0.5, p=2)

        def generate_triplets(
            embeddings: torch.Tensor,
            labels: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, None]:
            """
            Simple online triplet mining within a batch.
            For each anchor, pick one random positive (same label, different index)
            and one random negative (different label).
            """
            anchors: list[torch.Tensor] = []
            positives: list[torch.Tensor] = []
            negatives: list[torch.Tensor] = []

            num_samples = embeddings.size(0)
            for i in range(num_samples):
                anchor_label = labels[i]
                pos_mask = labels == anchor_label
                neg_mask = labels != anchor_label

                pos_indices = torch.where(pos_mask)[0]
                neg_indices = torch.where(neg_mask)[0]

                # Need at least one other positive and one negative
                if pos_indices.numel() <= 1 or neg_indices.numel() == 0:
                    continue

                # Exclude self from positives
                pos_indices = pos_indices[pos_indices != i]
                if pos_indices.numel() == 0:
                    continue

                pos_idx = pos_indices[torch.randint(pos_indices.numel(), (1,)).item()]
                neg_idx = neg_indices[torch.randint(neg_indices.numel(), (1,)).item()]

                anchors.append(embeddings[i])
                positives.append(embeddings[pos_idx])
                negatives.append(embeddings[neg_idx])

            if not anchors:
                return None, None, None

            return (
                torch.stack(anchors, dim=0),
                torch.stack(positives, dim=0),
                torch.stack(negatives, dim=0),
            )

        def compute_class_means(
            feats: torch.Tensor,
            lbls: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute mean embedding per class from given features and labels.
            Used for validation-time nearest-centroid accuracy.
            """
            self.model.eval()
            with torch.no_grad():
                emb = self.model.get_embedding(feats)

            num_classes = self.model.num_persons
            emb_dim = emb.size(1)
            means = torch.zeros(num_classes, emb_dim, device=self.device)

            for c in range(num_classes):
                mask = lbls == c
                if mask.any():
                    means[c] = emb[mask].mean(dim=0)

            return means

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

        best_val_acc = 0.0
        print("\nStarting training...")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            total_triplets = 0

            perm = torch.randperm(len(train_features))
            for i in range(0, len(train_features), batch_size):
                batch_idx = perm[i : i + batch_size]
                batch_x = train_features[batch_idx]
                batch_y = train_labels[batch_idx]
                batch_x = augment(batch_x)

                optimizer.zero_grad()
                # Compute embeddings and construct triplets within the batch
                embeddings = self.model.get_embedding(batch_x)
                anchor, positive, negative = generate_triplets(embeddings, batch_y)

                # If we cannot form any triplets from this batch, skip the step
                if anchor is None:
                    continue

                loss = triplet_criterion(anchor, positive, negative)
                loss.backward()
                optimizer.step()

                num_triplets = anchor.size(0)
                train_loss += loss.item() * num_triplets
                total_triplets += num_triplets

            scheduler.step()
            if total_triplets > 0:
                train_loss /= total_triplets
            else:
                train_loss = float("nan")

            # Validation: nearest-centroid classification in embedding space
            self.model.eval()
            with torch.no_grad():
                class_means = compute_class_means(train_features, train_labels)
                val_emb = self.model.get_embedding(val_features)

                # Cosine similarity between validation embeddings and class means
                class_means_norm = F.normalize(class_means, dim=1)
                val_emb_norm = F.normalize(val_emb, dim=1)
                sims = val_emb_norm @ class_means_norm.t()  # [N_val, num_persons]
                val_pred = sims.argmax(dim=1)
                val_acc = (val_pred == val_labels).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save("best_model.pt")

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:3d}: "
                    f"Triplet Loss={train_loss:.4f} | "
                    f"Val Acc (nearest-centroid)={val_acc:.2%}"
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
                "image_size": self.preprocessor.image_size,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model and metadata"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.person_names = checkpoint["person_names"]
        if "image_size" in checkpoint:
            self.preprocessor.image_size = checkpoint["image_size"]
            self.preprocessor._transform = T.Compose([
                T.Resize((self.preprocessor.image_size, self.preprocessor.image_size)),
                T.ToTensor(),
            ])
        print(f"Model loaded from {path}")

    def memory_usage(self) -> dict:
        """Estimate memory usage"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        total = param_size + buffer_size
        total_mb = total / (1024 * 1024)

        return {
            "model_params": f"{param_size / (1024*1024):.1f} MB" if param_size >= 1024*1024 else f"{param_size / 1024:.1f} KB",
            "model_buffers": f"{buffer_size / (1024*1024):.1f} MB" if buffer_size >= 1024*1024 else f"{buffer_size / 1024:.1f} KB",
            "total_model": f"{total_mb:.1f} MB" if total >= 1024*1024 else f"{total / 1024:.1f} KB",
            "estimated_inference_ram": "~1-2 GB" if total_mb > 100 else "~50-100 MB",
        }


def print_model_info(model: LightweightFaceNet):
    """Print model architecture summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total_params * 4 / (1024 * 1024)
    size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{total_params * 4 / 1024:.1f} KB"

    print("\n" + "=" * 50)
    print("Model Summary")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{size_str} (float32)")
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
