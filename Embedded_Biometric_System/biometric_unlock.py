"""
Smart lock with local face + voice authentication.
Authentication runs on user input (e.g. button)
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path so we can import Facial_Rec_Development
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Facial_Rec_Development.model import FaceRecognitionSystem
from Facial_Rec_Development.ImageProcessor import ImagePreprocessor


class BiometricUnlock:
    """
    On-device dual-factor (face + voice) authentication for a smart lock.
    All inference and data live locally; unlock only if both models agree.
    """

    def __init__(self, database_path: str):
        """
        Inputs:
            database_path: Path to the on-device database storing
                           audio and image samples (and reference profiles)
                           for enrolled users.
        """
        self.database_path = database_path

        # Face pipeline: model + image preprocessor
        # NOTE: FaceRecognitionSystem expects its database at <project_root>/Facial_Rec_Development/database
        project_root = Path(__file__).resolve().parents[1]
        face_model_path = project_root / "Facial_Rec_Development" / "best_model.pt"

        # NOTE: num_persons must match the checkpoint; best_model.pt was trained with 480 identities.
        self.face_preprocessor = ImagePreprocessor(image_size=224)
        self.face_system = FaceRecognitionSystem(num_persons=480, image_size=224)

        # Load trained weights (assumes best_model.pt is present on the device)
        if face_model_path.is_file():
            self.face_system.load(str(face_model_path))

        # On boot: eagerly build and cache database embeddings so later recognition calls are fast.
        # This parses Facial_Rec_Development/database, computes per-person mean embeddings,
        # and stores them in self.face_system._database_embeddings.
        try:
            self.face_system._database_embeddings = self.face_system._build_database_embeddings(
                str(self.face_system._database_dir)
            )
        except Exception as e:  # noqa: BLE001
            # In an embedded system you'd likely log this and keep running
            print(f"Warning: failed to build face database embeddings: {e}")

    def on_user_request_authentication(self) -> None:
        """
        Entry point when the user requests access (e.g. button press).
        Captures sensor data, runs both models, and unlocks only if they agree.
        """
        face_user = self._run_face_model()
        voice_user = self._run_voice_model()

        if face_user is not None and voice_user is not None and face_user == voice_user:
            self._unlock()
        # else: reject (no unlock)

    def _run_face_model(self):
        """
        Run the face recognition model on the current camera frame.
        Returns the predicted user id/name if confident, else None.
        """
        # 1. Capture a stable, aligned face from the camera
        aligned_face = self.face_preprocessor.capture_aligned_face_from_camera()
        if aligned_face is None:
            return None

        # 2. Run the facial recognition system on the captured face.
        #    The FaceRecognitionSystem.__call__ performs the similarity check
        #    against the cached database embeddings with its default threshold.
        person = self.face_system(aligned_face)
        return person

    def run_face_recognition_loop(self, threshold: float = 0.7) -> None:
        """
        Continuously run camera-based face recognition:

            capture aligned face -> run model -> similarity check vs database
            -> print identified user or 'unidentifiable' -> repeat.

        Press Ctrl+C to exit the loop.
        """
        print("Starting face recognition loop. Press Ctrl+C to exit.")
        try:
            while True:
                aligned_face = self.face_preprocessor.capture_aligned_face_from_camera()
                if aligned_face is None:
                    continue

                person = self.face_system(aligned_face, threshold=threshold)
                if person is None:
                    print("User unidentifiable.")
                else:
                    print(f"Recognized user: {person}")
        except KeyboardInterrupt:
            print("\nFace recognition loop stopped by user.")

    def _run_voice_model(self):
        """
        Run the voice recognition model on the current audio segment.
        Returns the predicted user id/name if confident, else None.
        """
        raise NotImplementedError("Voice model not yet implemented")

    def _unlock(self) -> None:
        """
        Send control signal to the lock actuator (e.g. servo) to unlock.
        """
        raise NotImplementedError("Lock actuator not yet implemented")


def main() -> None:
    """
    Simple entry point to exercise the face-recognition pipeline end-to-end.

    - Initializes BiometricUnlock (which loads the face model and builds the database embeddings).
    - Starts an infinite loop that:
        * captures an aligned face from the camera
        * runs it through the model
        * performs the similarity check against the facial database
        * prints the recognized user or 'unidentifiable'
    """
    project_root = PROJECT_ROOT
    database_path = project_root / "Facial_Rec_Development" / "database"

    print(f"Using facial database at: {database_path}")
    unlock_system = BiometricUnlock(database_path=str(database_path))

    # Run continuous face recognition loop (Ctrl+C to stop)
    unlock_system.run_face_recognition_loop(threshold=0.7)


if __name__ == "__main__":
    main()
