"""
Smart lock with local face + voice authentication.
Authentication runs on user input (e.g. button)
"""

from pathlib import Path
import sys
import time
from datetime import datetime

# Ensure project root is on sys.path so we can import Facial_Rec_Development
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Add Audio_Rec_Development for voice inference
AUDIO_REC_ROOT = PROJECT_ROOT / "Audio_Rec_Development" / "voicemodelece113"
if str(AUDIO_REC_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_REC_ROOT))

from Facial_Rec_Development.model import FaceRecognitionSystem
from Facial_Rec_Development.ImageProcessor import ImagePreprocessor

from lcd_uart_test import LCDUARTDisplay


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
            print(f"Warning: failed to build face database embeddings: {e}")

        # Voice pipeline: load model from Audio_Rec_Development
        voice_model_path = AUDIO_REC_ROOT / "best_model.pt"
        self.voice_system = None
        if voice_model_path.is_file():
            try:
                from infer import load_system
                self.voice_system = load_system(voice_model_path, None)
            except Exception as e:  # noqa: BLE001
                print(f"Warning: failed to load voice model: {e}")

        # Initialize LCD display
        self.lcd_display = LCDUARTDisplay(port="/dev/cu.usbmodem1101")

    def _run_face_model(self, aligned_face, threshold: float = 0.7):
        """
        Run the face recognition model on an already aligned face crop.
        Returns the predicted user id/name if confident, else None.
        """
        if aligned_face is None:
            return None
        # Single place where we call into FaceRecognitionSystem for identification.
        return self.face_system(aligned_face, threshold=threshold)

    def _run_voice_model(self):
        """
        Run the voice recognition model on the current audio segment.
        Returns the predicted user id/name if confident, else None.
        """
        if self.voice_system:
            import sounddevice as sd
            import soundfile as sf
            import torch
            import torchaudio.transforms as T
            record_rate, target_rate = 48000, 16000
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            tmp_path = str(PROJECT_ROOT / f"clip_{ts}.wav")
            rec = sd.rec(int(3 * record_rate), samplerate=record_rate, channels=1)
            sd.wait()
            if record_rate != target_rate:
                resampler = T.Resample(record_rate, target_rate)
                rec_16k = resampler(torch.from_numpy(rec).float().T.unsqueeze(0)).squeeze().numpy()
            else:
                rec_16k = rec
            sf.write(tmp_path, rec_16k, target_rate)
            try:
                speaker, confidence, _ = self.voice_system.predict(tmp_path)
                print(f"Voice: {speaker} ({confidence:.1%})")
                return speaker, confidence
            except Exception as e:  # noqa: BLE001
                print(f"Warning: failed to run voice model: {e}")
                return None, None

    def run_biometric_auth_loop(self, threshold: float = 0.7, use_pi_camera: bool = False) -> None:
        """
        Continuously run the full biometric pipeline (currently face-based; voice TBD):

            run camera until a face is detected
            -> run face model helper on the captured face
            -> perform similarity check vs database inside FaceRecognitionSystem
            -> call _unlock(user) or report that no one is identified

        Press Ctrl+C to exit the loop.
        """
        print("Starting biometric authentication loop. Press Ctrl+C to exit.")
        try:
            while True:
                # Run the camera until we capture an aligned face.
                self.lcd_display.send_message("Waiting for face...")
                aligned_face = self.face_preprocessor.capture_aligned_face_from_camera(
                    use_picamera=use_pi_camera
                )
                if aligned_face is None:
                    continue

                # Record 3 s from USB mic and run voice inference
                # self.lcd_display.send_message("Face detected\nRecording voice...")
                # speaker, confidence = self._run_voice_model()
                # self.lcd_display.send_message("Finished recording voice")
                speaker = "John Doe"

                # Run the face model helper on the captured face.
                person = self._run_face_model(aligned_face, threshold=threshold)
                if person is None:
                    self.lcd_display.send_message("INVALID User")
                    print("User unidentifiable.")
                else:
                    # Delegate handling of a successful identification to the unlock logic.
                    print(f"Face id: {person}\nVoice id: {speaker}")
                    self.lcd_display.send_message(f"Face id: {person}\nVoice id: {speaker}")
                    # self._unlock(person)
                time.sleep(3)

        except KeyboardInterrupt:
            print("\nFace recognition loop stopped by user.")

    def _unlock(self, user: str) -> None:
        """
        Send control signal to the lock actuator (e.g. servo) to unlock.
        """
        raise NotImplementedError(f"Lock actuator not yet implemented (recognized user: {user})")


def main() -> None:
    """
    Simple entry point to exercise the biometric authentication pipeline end-to-end.

    - Initializes BiometricUnlock (which loads the face model and builds the database embeddings).
    - Starts an infinite loop that:
        * runs the camera until a face is detected
        * runs the captured face through the model helper
        * performs the similarity check against the facial database
        * calls _unlock(user) or reports that no one is identified
    """
    project_root = PROJECT_ROOT
    database_path = project_root / "Facial_Rec_Development" / "database"

    print(f"Using facial database at: {database_path}")
    unlock_system = BiometricUnlock(database_path=str(database_path))

    # Run continuous biometric authentication loop (Ctrl+C to stop).
    # Set use_pi_camera=True on Raspberry Pi to use picamera2; keep False on a Mac/PC.
    unlock_system.run_biometric_auth_loop(threshold=0.7, use_pi_camera=False)


if __name__ == "__main__":
    main()
