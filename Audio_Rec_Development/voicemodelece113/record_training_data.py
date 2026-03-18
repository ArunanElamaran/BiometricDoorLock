#!/usr/bin/env python3
"""
Record voice samples for training. Prompts for speaker, then each word.
Saves to DATA_DIR/speaker/word/word_speaker_N.wav
"""

from pathlib import Path

import sounddevice as sd
import soundfile as sf
import torch
import torchaudio.transforms as T

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "ece113d_data"

# Words to record (matches training structure)
WORDS = ["close", "open", "hello", "start", "yes"]

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


def get_next_index(speaker_dir: Path, word: str, speaker: str) -> int:
    """Get next index for word_speaker_N.wav (or word_speaker_N.m4a)."""
    word_dir = speaker_dir / word
    if not word_dir.exists():
        return 1
    existing = list(word_dir.glob(f"{word}_{speaker}_*"))
    indices = []
    for f in existing:
        stem = f.stem
        try:
            n = int(stem.split("_")[-1])
            indices.append(n)
        except ValueError:
            pass
    return max(indices, default=0) + 1


def main() -> None:
    print("Voice training data recorder")
    print(f"Data dir: {DATA_DIR}\n")

    # List existing speakers or allow new
    if DATA_DIR.exists():
        existing = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
        if existing:
            print(f"Existing speakers: {', '.join(existing)}")
            for i, s in enumerate(existing, 1):
                print(f"  {i}) {s}")
            print(f"  {len(existing) + 1}) New speaker")
            choice = input("Choose speaker (number or name): ").strip().lower()
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(existing):
                    speaker = existing[idx - 1]
                else:
                    speaker = input("New speaker name: ").strip().lower()
            else:
                speaker = choice
        else:
            speaker = input("Speaker name (e.g. arunan, sourish): ").strip().lower()
    else:
        DATA_DIR.mkdir(parents=True)
        speaker = input("Speaker name (e.g. arunan, sourish): ").strip().lower()

    if not speaker:
        print("No speaker name entered. Exiting.")
        return

    speaker_dir = DATA_DIR / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRecording {len(WORDS)} words. Say each word when prompted.\n")

    for word in WORDS:
        input(f"Press Enter, then say '{word}'... ")
        print("Recording...")
        audio = record_word()
        idx = get_next_index(speaker_dir, word, speaker)
        word_dir = speaker_dir / word
        word_dir.mkdir(exist_ok=True)
        out_path = word_dir / f"{word}_{speaker}_{idx}.wav"
        sf.write(str(out_path), audio, TARGET_RATE)
        print(f"  Saved: {out_path}\n")

    print(f"Done. Recorded {len(WORDS)} words for {speaker}.")
    print(f"Run train.py to retrain. Set DATA_DIR in train.py to: {DATA_DIR}")


if __name__ == "__main__":
    main()
