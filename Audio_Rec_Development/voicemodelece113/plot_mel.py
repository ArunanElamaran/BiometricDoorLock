#!/usr/bin/env python3
"""Plot mel spectrograms for given audio files (same preprocessing as model input)."""

import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving

from model import VoiceRecognitionSystem

def plot_mel(audio_path: str, output_path: str, title: str):
    system = VoiceRecognitionSystem(num_speakers=2)
    mel_spec = system.preprocessor(audio_path)  # [1, n_mels, time]
    mel_np = mel_spec.squeeze().numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mel_np, aspect='auto', origin='lower', cmap='magma')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Mel bin')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Normalized amplitude')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    arunan_path = "ece113d_data/arunan/hello/New Recording 23.m4a"
    sourish_path = "ece113d_data/sourish/hello/hello_sourish_13.m4a"

    plot_mel(arunan_path, "mel_arunan.png", "Mel spectrogram — Arunan (hello)")
    plot_mel(sourish_path, "mel_sourish.png", "Mel spectrogram — Sourish (hello)")
