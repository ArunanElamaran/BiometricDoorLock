"""
Training script for face recognition model.
Configure the constants below, then run: python train.py
"""

from pathlib import Path

try:
    from .model import FaceRecognitionSystem, print_model_info
except ImportError:  # Fallback for running as a standalone script
    from model import FaceRecognitionSystem, print_model_info

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"  # Folder containing train/ and val/ (each with person subfolders)
OUTPUT_MODEL = "face_model.pt"  # Where to save the trained model
EPOCHS = 50
BATCH_SIZE = 256  # VGG-style model is large; reduce (e.g. to 4) if OOM during training
LEARNING_RATE = 0.001
DEBUG_FAILURES = False  # If True, show each failed image in a window and wait for ESC

# ============================================================


def _count_persons(data_dir: str) -> int:
    """Infer number of persons from union of data_dir/train/ and data_dir/val/ person subfolders."""
    data_path = Path(data_dir)
    train_path = data_path / "train"
    val_path = data_path / "val"
    if not train_path.is_dir():
        raise FileNotFoundError(f"Expected training split at {train_path}")
    train_names = {d.name for d in train_path.iterdir() if d.is_dir()}
    val_names = {d.name for d in val_path.iterdir() if d.is_dir()} if val_path.is_dir() else set()
    return len(train_names | val_names)


if __name__ == "__main__":
    num_persons = _count_persons(DATA_DIR)
    print(f"Detected {num_persons} persons in {DATA_DIR}/train/\n")
    system = FaceRecognitionSystem(num_persons=num_persons)
    print_model_info(system.model)

    system.train(
        data_dir=DATA_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        debug_failures=DEBUG_FAILURES,
    )

    system.save(OUTPUT_MODEL)
    print(f"\nDone. Model saved to {OUTPUT_MODEL}")
