#!/usr/bin/env python3
"""
Test the voice recording + inference pipeline (mirrors biometric_unlock.py lines 101-113).
Records 3 s from USB mic, saves to clip_{timestamp}.wav, runs voice inference.
"""

from pathlib import Path
from datetime import datetime

import sounddevice as sd
import soundfile as sf

# Output in voicemodelece113 dir; change to PROJECT_ROOT if you want BiometricDoorLock root
OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    print("Recording 3 seconds from mic... (speak now)")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tmp_path = str(OUTPUT_DIR / f"clip_{ts}.wav")

    rec = sd.rec(int(3 * 16000), samplerate=16000, channels=1)
    sd.wait()

    sf.write(tmp_path, rec, 16000)
    print(f"Saved to {tmp_path}")

    # Run inference
    try:
        from infer import load_system
        model_path = OUTPUT_DIR / "best_model.pt"
        if model_path.is_file():
            system = load_system(model_path, None)
            speaker, confidence, _ = system.predict(tmp_path)
            print(f"Voice: {speaker} ({confidence:.1%})")
        else:
            print("No best_model.pt found, skipping inference")
    except Exception as e:
        print(f"Inference failed: {e}")


if __name__ == "__main__":
    main()
