#!/usr/bin/env python3
"""
Test the voice recording + inference pipeline (mirrors biometric_unlock.py lines 101-113).
Records 3 s from USB mic, saves to clip_{timestamp}.wav, runs voice inference.
"""

from pathlib import Path
from datetime import datetime

import sounddevice as sd
import soundfile as sf
import torch
import torchaudio.transforms as T

# Output in voicemodelece113 dir; change to PROJECT_ROOT if you want BiometricDoorLock root
OUTPUT_DIR = Path(__file__).resolve().parent

# Many USB mics only support 44100 or 48000 Hz; model expects 16 kHz
RECORD_RATE = 44100
TARGET_RATE = 16000


def main() -> None:
    print("Recording 3 seconds from mic... (speak now)")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tmp_path = str(OUTPUT_DIR / f"clip_{ts}.wav")

    rec = sd.rec(int(3 * RECORD_RATE), samplerate=RECORD_RATE, channels=1)
    sd.wait()

    # Resample to 16 kHz for the voice model
    if RECORD_RATE != TARGET_RATE:
        resampler = T.Resample(RECORD_RATE, TARGET_RATE)
        rec_t = torch.from_numpy(rec).float().T.unsqueeze(0)  # [1, samples]
        rec_16k = resampler(rec_t).squeeze().numpy()
    else:
        rec_16k = rec

    sf.write(tmp_path, rec_16k, TARGET_RATE)
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
