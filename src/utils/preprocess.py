import argparse
import os
import glob

import numpy as np
import torch
import torchaudio

def compute_spectogram(wav_path):
    torchaudio.set_audio_backend("soundfile")
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    
    stft = torch.stft(waveform, n_fft=512, hop_length=256, win_length=512, return_complex=True)
    
    magnitude = stft.abs()
    log_mag = 20 * torch.log10(magnitude + 1e-6)

    mean = log_mag.mean()
    std = log_mag.std()
    norm = (log_mag - mean) / std

    return norm.cpu().numpy()


def split_process(input_dir : str, output_dir : str):
    os.makedirs(output_dir, exist_ok= True)

    patterns = ["*.wav", "*.flac"]
    wav_paths = []

    for p in patterns:
        wav_paths.extend(glob.glob(os.path.join(input_dir, p)))
    wav_paths = sorted(wav_paths)

    for idx, wav_path in enumerate(wav_paths, 1):
        utt_id = os.path.basename(wav_path).rsplit(".", 1)[0]
        out_path = os.path.join(output_dir, f"{utt_id}.npy")

        if os.path.exists(out_path):
            continue
        
        spectograms = compute_spectogram(wav_path)

        np.save(out_path, spectograms)

        print(f"[{idx}/{len(wav_paths)}] saved {utt_id}.npy")



def main():
    parser = argparse.ArgumentParser(description="Preprocess audio to STFT spectrograms (.npy)")

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    arguments = parser.parse_args()

    split_process(arguments.input_dir, arguments.output_dir)

if __name__ == "__main__":
    main()
