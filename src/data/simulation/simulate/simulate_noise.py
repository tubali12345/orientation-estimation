import random
from wave import Wave_read

import numpy as np

from ..utils.audio_utils import load_audio


def get_random_noise(noise_list: list, max_len: int, sr: int) -> tuple[np.ndarray, int]:
    noise_path = np.random.choice(noise_list, size=1, replace=False).item()

    wav_r = Wave_read(noise_path)
    nchannels, samplewidth, samplerate, nframes, comptype, compname = wav_r.getparams()
    noise = load_audio(noise_path, sr)
    if noise is None:
        raise ValueError(f"Could not load noise file: {noise_path}")
    if noise.size != 0 and not (noise == noise[0]).all() and nframes >= 0.3 * sr:
        noise = _load_audio_from_wave(wav_r, max_len, nframes)
    return get_noise_delay(noise, max_len, sr)


def _load_audio_from_wave(wav_r: Wave_read, max_len: int, nframes: int) -> np.ndarray:
    start_sample = random.randint(0, max(0, nframes - max_len))
    num_samples = random.randint(0, max_len - 1)
    wav_r.setpos(start_sample)
    audio_bytes = wav_r.readframes(num_samples)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_np * (1.0 / (2**15))


def get_noise_delay(noise: np.ndarray, aud_len: int, sample_rate: int) -> tuple[np.ndarray, int]:
    delay = 0
    noise_len = len(noise)
    n_a_diff = noise_len - aud_len
    if n_a_diff > 0:
        # Cut out a piece of the noise of same length as audio
        start = random.randint(0, n_a_diff)
        noise = noise[start : start + aud_len]
    else:
        # Place the noise somewhere inside the audio
        a_n_diff = -n_a_diff
        start = random.randint(0, a_n_diff)
        delay = start / sample_rate

    return noise, int(delay)
