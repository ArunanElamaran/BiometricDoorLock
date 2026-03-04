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

DATA_DIR = "data"  # Folder containing person subfolders (no explicit train/val split)
OUTPUT_MODEL = "face_model.pt"  # Where to save the trained model
EPOCHS = 50
BATCH_SIZE = 256  # VGG-style model is large; reduce (e.g. to 4) if OOM during training
LEARNING_RATE = 0.001
DEBUG_FAILURES = False  # If True, show each failed image in a window and wait for ESC
VALIDATION_SPLIT = 0.2  # Fraction of data reserved for validation (like audio model)

# ============================================================


def _count_persons(data_dir: str) -> int:
    """
    Infer number of persons from person subfolders directly under data_dir,
    mirroring how the audio model derives speaker classes.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Expected dataset directory at {data_path}")
    person_names = {d.name for d in data_path.iterdir() if d.is_dir()}
    return len(person_names)


if __name__ == "__main__":
    num_persons = _count_persons(DATA_DIR)
    print(f"Detected {num_persons} persons in {DATA_DIR}/\n")
    system = FaceRecognitionSystem(num_persons=num_persons)
    print_model_info(system.model)

    system.train(
        data_dir=DATA_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        debug_failures=DEBUG_FAILURES,
        validation_split=VALIDATION_SPLIT,
    )

    system.save(OUTPUT_MODEL)
    print(f"\nDone. Model saved to {OUTPUT_MODEL}")
