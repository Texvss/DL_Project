import argparse
import os
import glob
import numpy as np
import torch
import torchaudio
import pickle

TARGET_T = 600

def compute_global_stats(input_dir: str) -> tuple:
    all_mags = []
    patterns = ["*.wav", "*.flac"]
    wav_paths = []
    for p in patterns:
        wav_paths.extend(glob.glob(os.path.join(input_dir, p)))
    if not wav_paths:
        raise FileNotFoundError(f"No audio files found in {input_dir}")
    for wav_path in wav_paths:
        try:
            waveform, sr = torchaudio.load(wav_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
            stft = torch.stft(
                waveform,
                n_fft=512,
                hop_length=256,
                win_length=512,
                window=torch.hann_window(512),
                return_complex=True
            )
            magnitude = stft.abs()
            log_mag = 20 * torch.log10(magnitude + 1e-6)
            all_mags.append(log_mag.cpu().numpy())
        except Exception as e:
            print(f"Warning: Failed to process {wav_path}: {e}")
    if not all_mags:
        raise ValueError(f"No valid spectrograms computed in {input_dir}")
    all_mags = np.concatenate(all_mags, axis=-1)
    global_mean = all_mags.mean()
    global_std = all_mags.std()
    return global_mean, global_std

def compute_spectrogram(wav_path: str, global_mean: float, global_std: float) -> np.ndarray:
    torchaudio.set_audio_backend("soundfile")
    try:
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        stft = torch.stft(
            waveform,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=torch.hann_window(512),
            return_complex=True
        )
        magnitude = stft.abs()
        log_mag = 20 * torch.log10(magnitude + 1e-6)
        norm = (log_mag - global_mean) / (global_std + 1e-6)
        return norm.cpu().numpy()
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        raise

def split_process(input_dir: str, output_dir: str, stats_file: str = None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if stats_file and os.path.exists(stats_file):
        with open(stats_file, 'rb') as f:
            global_mean, global_std = pickle.load(f)
        print(f"Loaded global stats from {stats_file}: mean={global_mean:.4f}, std={global_std:.4f}")
    else:
        global_mean, global_std = compute_global_stats(input_dir)
        print(f"Computed global stats: mean={global_mean:.4f}, std={global_std:.4f}")
        if stats_file:
            with open(stats_file, 'wb') as f:
                pickle.dump((global_mean, global_std), f)
            print(f"Saved global stats to {stats_file}")
    patterns = ["*.wav", "*.flac"]
    wav_paths = []
    for p in patterns:
        wav_paths.extend(glob.glob(os.path.join(input_dir, p)))
    wav_paths = sorted(wav_paths)
    if not wav_paths:
        raise FileNotFoundError(f"No audio files found in {input_dir}")
    print(f"Found {len(wav_paths)} audio files in {input_dir}")
    for idx, wav_path in enumerate(wav_paths, 1):
        utt_id = os.path.basename(wav_path).rsplit(".", 1)[0]
        out_path = os.path.join(output_dir, f"{utt_id}.npy")
        if os.path.exists(out_path):
            print(f"[{idx}/{len(wav_paths)}] Skipped {utt_id}.npy (already exists)")
            continue
        spectrogram = compute_spectrogram(wav_path, global_mean, global_std)
        c, f, t = spectrogram.shape
        if t < TARGET_T:
            pad_amt = TARGET_T - t
            spectrogram = np.pad(
                spectrogram,
                ((0, 0), (0, 0), (0, pad_amt)),
                mode='constant',
                constant_values=0
            )
        else:
            spectrogram = spectrogram[:, :, :TARGET_T]
        np.save(out_path, spectrogram)
        print(f"[{idx}/{len(wav_paths)}] saved {utt_id}.npy (shape={spectrogram.shape})")

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess audio to fixed-size STFT spectrograms (.npy)"
    )
    parser.add_argument("--input_dir", required=True, help="Directory with wav/flac files")
    parser.add_argument("--output_dir", required=True, help="Directory to save .npy files")
    parser.add_argument("--stats_file", default="global_stats.pkl", help="File to save/load global mean and std")
    args = parser.parse_args()
    split_process(args.input_dir, args.output_dir, args.stats_file)

if __name__ == "__main__":
    main()