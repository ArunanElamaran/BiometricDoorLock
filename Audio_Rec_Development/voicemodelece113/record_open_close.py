#!/usr/bin/env python3
"""
Record 'open' and 'close' voice samples for training. 5 recordings per word.
Saves to DATA_DIR/speaker/word/word_speaker_N.wav
"""

from pathlib import Path
from datetime import datetime

import sounddevice as sd
import soundfile as sf
import torch
import torchaudio.transforms as T

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "extra_ece113d_data"

WORDS = ["open", "close"]
RECORDINGS_PER_WORD = 5

RECORD_RATE = 48000
TARGET_RATE = 16000
DURATION_SEC = 3


def record_word():
    """Record DURATION_SEC from mic, return 16 kHz mono audio as numpy array."""
    rec = sd.rec(int(DURATION_SEC * RECORD_RATE), samplerate=RECORD_RATE, channels=1)
    sd.wait()
    if RECORD_RATE != TARGET_RATE:
        resampler = T.Resample(RECORD_RATE, TARGET_RATE)
        rec_t = torch.from_numpy(rec).float().T.unsqueeze(0)
        rec_16k = resampler(rec_t).squeeze().numpy()
    else:
        rec_16k = rec
    return rec_16k


SPEAKERS = ["sourish", "arunan"]


def main() -> None:
    print("Record 'open' and 'close' (5 times each)")
    print(f"Data dir: {DATA_DIR}\n")

    print("Who is speaking?")
    for i, s in enumerate(SPEAKERS, 1):
        print(f"  {i}) {s}")
    choice = input("Choose (1 or 2): ").strip()
    if choice == "1":
        speaker = SPEAKERS[0]
    elif choice == "2":
        speaker = SPEAKERS[1]
    else:
        print("Invalid choice. Exiting.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    speaker_dir = DATA_DIR / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRecording {RECORDINGS_PER_WORD} samples for each word.\n")

    for word in WORDS:
        word_dir = speaker_dir / word
        word_dir.mkdir(exist_ok=True)
        for i in range(RECORDINGS_PER_WORD):
            input(f"Press Enter, then say '{word}' ({i + 1}/{RECORDINGS_PER_WORD})... ")
            print("Recording...")
            audio = record_word()
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # ms precision
            out_path = word_dir / f"{word}_{speaker}_{ts}.wav"
            sf.write(str(out_path), audio, TARGET_RATE)
            print(f"  Saved: {out_path}\n")

    print(f"Done. Recorded {RECORDINGS_PER_WORD} samples each of {', '.join(WORDS)} for {speaker}.")


if __name__ == "__main__":
    main()
