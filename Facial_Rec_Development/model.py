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

import cv2
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

try:
    from .ImageProcessor import ImagePreprocessor
except ImportError:  # Fallback for running as a standalone script
    from ImageProcessor import ImagePreprocessor


class _FaceDataset(Dataset):
    """Loads face images from paths on demand to avoid loading the full dataset into memory."""

    def __init__(self, paths: list[str], labels: list[int], preprocessor: ImagePreprocessor, align: bool = False):
        self.paths = paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.align = align

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        x = self.preprocessor(self.paths[i], align=self.align)
        return x.squeeze(0), self.labels[i]


def _make_conv_block(in_c: int, out_c: int, num_convs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
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
    Same interface as project: num_persons, embedding_dim, forward(), get_embedding().
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

        # 224 -> 112 -> 56 -> 28 -> 14 -> 7 with fewer channels
        self.conv_blocks = nn.Sequential(
            _make_conv_block(3, 32, 2),     # 224 -> 112
            _make_conv_block(32, 64, 2),    # 112 -> 56
            _make_conv_block(64, 128, 2),   # 56 -> 28
            _make_conv_block(128, 256, 2),  # 28 -> 14 -> 7
        )

        # Global pooling + small MLP head: 7x7x256 -> 256 -> embedding -> classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_proj = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(embedding_dim, num_persons)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        # x: [batch, 3, 224, 224]
        features = self.conv_blocks(x)              # [B, 256, 7, 7]
        pooled = self.global_pool(features)         # [B, 256, 1, 1]
        flat = pooled.view(pooled.size(0), -1)      # [B, 256]
        emb = self.embedding_proj(flat)             # [B, embedding_dim]
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
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
        # Single on-device database located at <this_folder>/database
        # Cached embeddings: {person_name: embedding}
        self._database_dir = Path(__file__).resolve().parent / "database"
        self._database_embeddings: Optional[dict[str, np.ndarray]] = None

    def _show_failed_image_and_wait(self, image_path: str, error_msg: str) -> None:
        """Display the image that failed to load and block until Escape is pressed."""
        img = cv2.imread(image_path)
        if img is None:
            print(f"    (Could not read image for display: {image_path})")
            return
        # Resize if very large so it fits on screen
        h, w = img.shape[:2]
        max_side = 800
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.putText(
            img, "Press ESC to continue", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.imshow("Load failed - press ESC", img)
        print(f"    Failed: {image_path}")
        print(f"    Error: {error_msg}")
        while True:
            if cv2.waitKey(100) == 27:  # Escape
                break
        cv2.destroyAllWindows()

    def _load_split(
        self,
        split_path: Path,
        person_names: list[str],
        debug_failures: bool = False,
    ) -> tuple[list[str], list[int]]:
        """Collect (path, label) for one split. If debug_failures=True, validate by loading and show failures; else just collect paths."""
        paths: list[str] = []
        labels: list[int] = []
        name_to_idx = {name: i for i, name in enumerate(person_names)}
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        total = 0
        for idx, person_name in enumerate(person_names):
            person_dir = split_path / person_name
            if not person_dir.is_dir():
                continue
            image_files = [
                f for f in person_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            image_files = sorted(image_files)
            for image_file in image_files:
                path = str(image_file)
                if debug_failures:
                    try:
                        self.preprocessor(path, align=False)
                        paths.append(path)
                        labels.append(name_to_idx[person_name])
                        total += 1
                    except Exception as e:
                        print(f"    Warning: Failed to load {image_file.name}: {e}")
                        self._show_failed_image_and_wait(path, str(e))
                else:
                    paths.append(path)
                    labels.append(name_to_idx[person_name])
                    total += 1
            if (idx + 1) % 50 == 0:
                print(f"  ... {total} images from {idx + 1} persons")

        return paths, labels

    def load_train_val_splits(
        self,
        data_dir: str,
        debug_failures: bool = False,
        validation_split: float = 0.2,
    ) -> tuple[list[str], list[int], list[str], list[int]]:
        """
        Load dataset from a single directory and create a random train/val split,
        matching the behavior of the voice model.

        Expected directory structure:

            data_dir/
                person_name/
                    img1.jpg
                    img2.jpg
                    ...
                another_person/
                    ...

        All images are first collected, then randomly split into training and
        validation subsets using the given validation_split ratio.

        Returns:
            train_paths, train_labels, val_paths, val_labels
        """
        data_path = Path(data_dir)
        if not data_path.is_dir():
            raise FileNotFoundError(f"Expected dataset directory at {data_path}")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # Person subdirectories under data_dir mirror the audio model's speaker dirs
        person_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        self.person_names = [d.name for d in person_dirs]
        n_persons = len(self.person_names)

        if n_persons == 0:
            raise ValueError(f"No person subdirectories found under {data_path}")

        if n_persons != self.model.num_persons:
            raise ValueError(
                f"{data_path} has {n_persons} persons but model was created with "
                f"num_persons={self.model.num_persons}. Use FaceRecognitionSystem(num_persons={n_persons})."
            )

        print(f"Found {n_persons} persons: {self.person_names[:5]}{'...' if n_persons > 5 else ''}")

        # Collect all image paths and labels
        all_paths: list[str] = []
        all_labels: list[int] = []
        name_to_idx = {name: i for i, name in enumerate(self.person_names)}
        total = 0

        for idx, person_dir in enumerate(person_dirs):
            image_files = [
                f for f in person_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            image_files = sorted(image_files)
            for image_file in image_files:
                path = str(image_file)
                if debug_failures:
                    try:
                        self.preprocessor(path, align=False)
                        all_paths.append(path)
                        all_labels.append(name_to_idx[person_dir.name])
                        total += 1
                    except Exception as e:
                        print(f"    Warning: Failed to load {image_file.name}: {e}")
                        self._show_failed_image_and_wait(path, str(e))
                else:
                    all_paths.append(path)
                    all_labels.append(name_to_idx[person_dir.name])
                    total += 1
            if (idx + 1) % 50 == 0:
                print(f"  ... {total} images from {idx + 1} persons")

        if not all_paths:
            raise ValueError(f"No image files found under {data_path}")

        # Random train/val split following the audio model pattern
        n_samples = len(all_paths)
        indices = torch.randperm(n_samples)
        n_val = int(n_samples * validation_split)

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_paths = [all_paths[i] for i in train_indices]
        train_labels = [all_labels[i] for i in train_indices]
        val_paths = [all_paths[i] for i in val_indices]
        val_labels = [all_labels[i] for i in val_indices]

        print(f"  Total images: {n_samples}")
        print(f"  Train: {len(train_paths)} samples, Val: {len(val_paths)} samples (split={validation_split:.2f})")

        return train_paths, train_labels, val_paths, val_labels

    def train(
        self,
        data_dir: str,
        epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 0.001,
        debug_failures: bool = False,
        num_workers: int = 0,
        validation_split: float = 0.2,
    ):
        """Train the model on the dataset using cross-entropy classification
        (same strategy as the voice model). Expects data_dir with train/ and
        val/ subdirs. Images are loaded on demand (streaming) to avoid OOM.
        If debug_failures=True, each failed image is shown during the
        path-collection phase; press ESC to continue."""
        # Warm start from existing best checkpoint if available
        best_path = Path(__file__).resolve().parent / "best_model.pt"
        if best_path.is_file():
            print(f"Found existing checkpoint at {best_path}, loading before training...")
            self.load(str(best_path))

        print("Loading dataset (path lists only, random train/val split)...")
        train_paths, train_labels, val_paths, val_labels = self.load_train_val_splits(
            data_dir,
            debug_failures=debug_failures,
            validation_split=validation_split,
        )

        train_dataset = _FaceDataset(train_paths, train_labels, self.preprocessor, align=False)
        val_dataset = _FaceDataset(val_paths, val_labels, self.preprocessor, align=False)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
        )

        print(f"Training samples: {len(train_paths)}, Validation: {len(val_paths)}")
        if len(val_paths) == 0:
            print("Warning: No validation samples (data/val/ empty or missing). Training will run but no val accuracy or best-model checkpoint.")

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

        best_val_acc = 0.0
        print("\nStarting training (streaming batches, cross-entropy classification)...")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            total_train_samples = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_x = augment(batch_x)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                batch_size_actual = batch_y.size(0)
                train_loss += loss.item() * batch_size_actual
                train_correct += (logits.argmax(1) == batch_y).sum().item()
                total_train_samples += batch_size_actual

            scheduler.step()
            if total_train_samples > 0:
                train_loss /= total_train_samples
                train_acc = train_correct / total_train_samples
            else:
                train_loss = float("nan")
                train_acc = float("nan")

            # Validation: standard classification accuracy on val_loader
            val_acc = 0.0
            val_loss = float("nan")
            if len(val_paths) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss_total = 0.0
                    val_correct = 0
                    total_val_samples = 0
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        logits = self.model(batch_x)
                        loss = criterion(logits, batch_y)
                        batch_size_actual = batch_y.size(0)
                        val_loss_total += loss.item() * batch_size_actual
                        val_correct += (logits.argmax(1) == batch_y).sum().item()
                        total_val_samples += batch_size_actual

                    if total_val_samples > 0:
                        val_loss = val_loss_total / total_val_samples
                        val_acc = val_correct / total_val_samples

            if len(val_paths) > 0 and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save(str(best_path))

            if len(val_paths) > 0:
                print(
                    f"Epoch {epoch+1:3d}: "
                    f"Train Loss={train_loss:.4f}, Acc={train_acc:.2%} | "
                    f"Val Loss={val_loss:.4f}, Acc={val_acc:.2%}"
                )
            else:
                print(
                    f"Epoch {epoch+1:3d}: "
                    f"Train Loss={train_loss:.4f}, Acc={train_acc:.2%} | "
                    "Val: N/A (no val data)"
                )

        print(
            "\nTraining complete!"
            + (f" Best validation accuracy: {best_val_acc:.2%}" if len(val_paths) > 0 else " (no validation set)")
        )
        if len(val_paths) > 0:
            self.load(str(best_path))

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

    def get_face_embedding_from_array(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from a BGR numpy array (e.g. camera frame or
        the output of ImagePreprocessor.capture_aligned_face_from_camera).
        """
        self.model.eval()

        with torch.no_grad():
            # capture_aligned_face_from_camera already returns an aligned face crop,
            # so we just convert the BGR array to a normalized tensor.
            img_tensor = self.preprocessor._array_to_tensor(img_bgr).to(self.device)
            embedding = self.model.get_embedding(img_tensor)

        return embedding.cpu().numpy().squeeze()

    # ------------------------------------------------------------------
    # Database-based identification API
    # ------------------------------------------------------------------
    def _build_database_embeddings(self, database_dir: str) -> dict[str, np.ndarray]:
        """
        Scan a database directory laid out as:

            database/
                person1/
                    img1.jpg
                    img2.jpg
                    ...
                person2/
                    ...

        and compute a single reference embedding per person (mean over that
        person's images). Returns a mapping person_name -> embedding (np.ndarray).
        """
        db_path = Path(database_dir)
        if not db_path.is_dir():
            raise FileNotFoundError(f"Database directory not found: {db_path}")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        person_to_embeddings: dict[str, list[np.ndarray]] = {}

        for person_dir in sorted(p for p in db_path.iterdir() if p.is_dir()):
            person_name = person_dir.name
            image_files = [
                f for f in person_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            if not image_files:
                continue

            embeddings: list[np.ndarray] = []
            for img_path in image_files:
                try:
                    emb = self.get_face_embedding(str(img_path))
                    embeddings.append(emb)
                except Exception as e:
                    print(f"Warning: failed to compute embedding for {img_path}: {e}")

            if embeddings:
                # Mean embedding for this person
                person_to_embeddings[person_name] = np.mean(
                    np.stack(embeddings, axis=0), axis=0
                )

        if not person_to_embeddings:
            raise ValueError(
                f"No valid face images found in database directory: {db_path}"
            )

        return person_to_embeddings

    def __call__(
        self,
        aligned_face_bgr: np.ndarray,
        threshold: float = 0.7,
    ) -> Optional[str]:
        """
        Run the model on an already aligned face image (BGR numpy array),
        compare its embedding against a database of stored embeddings, and
        return the most likely person or None.

        This is designed to be called with the output of
        ImagePreprocessor.capture_aligned_face_from_camera.

        Args:
            aligned_face_bgr: Aligned face crop as a BGR numpy array.
            threshold: Minimum cosine similarity required to accept a match.

        Returns:
            The person name (directory name under the fixed database directory)
            that best matches the query image, or None if no person is within
            the similarity margin.
        """
        # Lazily build and cache embeddings for the single, fixed database
        if self._database_embeddings is None:
            self._database_embeddings = self._build_database_embeddings(str(self._database_dir))

        db_embeddings = self._database_embeddings

        # Compute normalized query embedding from the aligned BGR image
        query_emb = self.get_face_embedding_from_array(aligned_face_bgr)
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        best_person: Optional[str] = None
        best_sim: float = -1.0

        for person_name, ref_emb in db_embeddings.items():
            ref_norm = ref_emb / (np.linalg.norm(ref_emb) + 1e-8)
            sim = float(np.dot(query_norm, ref_norm))
            if sim > best_sim:
                best_sim = sim
                best_person = person_name

        if best_person is None or best_sim < threshold:
            return None

        return best_person

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
    system = FaceRecognitionSystem(num_persons=480)
    # Automatically load pretrained weights from best_model.pt next to this file, if present.
    default_ckpt = Path(__file__).resolve().parent / "best_model.pt"
    if default_ckpt.is_file():
        system.load(str(default_ckpt))
    print_model_info(system.model)
    print("\nMemory usage:")
    for k, v in system.memory_usage().items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 50)