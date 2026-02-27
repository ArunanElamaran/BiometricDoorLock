"""
Smart lock with local face + voice authentication.
Authentication runs on user input (e.g. button)
"""

from pathlib import Path

from Facial_Rec_Development import FaceRecognitionSystem, ImagePreprocessor


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
        face_model_path = project_root / "Facial_Rec_Development" / "face_model.pt"

        # TODO: if you change num_persons/image_size during training, make sure they match the checkpoint used here.
        self.face_preprocessor = ImagePreprocessor(image_size=224)
        self.face_system = FaceRecognitionSystem(num_persons=5, image_size=224)

        # Load trained weights (assumes face_model.pt is present on the device)
        if face_model_path.is_file():
            self.face_system.load(str(face_model_path))

        # On boot: cache all embeddings from Facial_Rec_Development/database so later recognition calls are fast.
        try:
            self.face_system.initialize_database_cache()
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

        # 2. Run the facial recognition system on the captured face
        person = self.face_system(aligned_face)
        return person

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
