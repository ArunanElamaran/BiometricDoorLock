"""
Training script for face recognition model.
Configure the constants below, then run: python train.py
"""

from model import FaceRecognitionSystem, print_model_info

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"  # Folder containing person subfolders (each with face images)
OUTPUT_MODEL = "face_model.pt"  # Where to save the trained model
NUM_PERSONS = 5
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# ============================================================

if __name__ == "__main__":
    system = FaceRecognitionSystem(num_persons=NUM_PERSONS)
    print_model_info(system.model)

    system.train(
        data_dir=DATA_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        validation_split=VALIDATION_SPLIT,
    )

    system.save(OUTPUT_MODEL)
    print(f"\nDone. Model saved to {OUTPUT_MODEL}")
