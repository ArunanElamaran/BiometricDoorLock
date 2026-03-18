#!/usr/bin/env python3
"""
Record 'open' and 'close' voice samples for training. 5 recordings per word.
Saves to DATA_DIR/speaker/word/word_speaker_N.wav
"""

from pathlib import Path

import sounddevice as sd
import soundfile as sf
import torch
import torchaudio.transforms as T

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "ece113d_data"

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


def get_next_index(speaker_dir: Path, word: str, speaker: str) -> int:
    """Get next index for word_speaker_N.wav."""
    word_dir = speaker_dir / word
    if not word_dir.exists():
        return 1
    existing = list(word_dir.glob(f"{word}_{speaker}_*"))
    indices = []
    for f in existing:
        try:
            n = int(f.stem.split("_")[-1])
            indices.append(n)
        except ValueError:
            pass
    return max(indices, default=0) + 1


def main() -> None:
    print("Record 'open' and 'close' (5 times each)")
    print(f"Data dir: {DATA_DIR}\n")

    if DATA_DIR.exists():
        existing = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
        if existing:
            for i, s in enumerate(existing, 1):
                print(f"  {i}) {s}")
            print(f"  {len(existing) + 1}) New speaker")
            choice = input("Choose speaker (number or name): ").strip().lower()
            if choice.isdigit():
                idx = int(choice)
                speaker = existing[idx - 1] if 1 <= idx <= len(existing) else input("New speaker name: ").strip().lower()
            else:
                speaker = choice
        else:
            speaker = input("Speaker name: ").strip().lower()
    else:
        DATA_DIR.mkdir(parents=True)
        speaker = input("Speaker name: ").strip().lower()

    if not speaker:
        print("No speaker name. Exiting.")
        return

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
            idx = get_next_index(speaker_dir, word, speaker)
            out_path = word_dir / f"{word}_{speaker}_{idx}.wav"
            sf.write(str(out_path), audio, TARGET_RATE)
            print(f"  Saved: {out_path}\n")

    print(f"Done. Recorded {RECORDINGS_PER_WORD} samples each of {', '.join(WORDS)} for {speaker}.")


if __name__ == "__main__":
    main()
