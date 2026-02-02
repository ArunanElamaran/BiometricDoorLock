"""
Smart lock with local face + voice authentication.
Authentication runs on user input (e.g. button)
"""


class BiometricUnlock:
    """
    On-device dual-factor (face + voice) authentication for a smart lock.
    All inference and data live locally; unlock only if both models agree.
    """

    def __init__(self, database_path: str):
        """
        Args:
            database_path: Path to the on-device database storing
                           audio and image samples (and reference profiles)
                           for enrolled users.
        """
        self.database_path = database_path

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
        raise NotImplementedError("Face model not yet implemented")

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
