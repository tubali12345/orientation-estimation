import random
from wave import Wave_read

import numpy as np
import torchaudio
from scipy.signal import fftconvolve


def get_random_noise(noise_list: list, max_len: int, sr: int) -> np.ndarray:
    noise_path = np.random.choice(noise_list, size=1, replace=False).item()

    wav_r = Wave_read(noise_path)
    _, _, noise_sr, nframes, _, _ = wav_r.getparams()
    if noise_sr != sr:
        raise ValueError(f"Sample rate mismatch: {noise_sr} != {sr}")
    return _load_audio_from_wave(wav_r, noise_path, max_len, nframes)


def _load_audio_from_wave(wav_r: Wave_read, noise_path: str, max_len: int, nframes: int) -> np.ndarray:
    # select a random chunk from the noise that is exactly max_len long
    start_sample = random.randint(0, max(nframes - max_len, 0))
    audio_np = torchaudio.load(noise_path, frame_offset=start_sample, num_frames=max_len)[0].numpy().T
    num_channels = wav_r.getnchannels()
    if num_channels > 1:
        audio_np = audio_np.mean(axis=1)
    return audio_np


def render_diffuse_array_ambience_from_mono_ambience(mono_ambience_sig, samplerate, array_geom_xyz):
    """
    Renders a multichannel ambience signal for a specific array geometry of
    omnidirectional microphones from a monophonic ambience recording, under
    isotropic diffuse conditions.

    Args:
        mono_ambience_sig: [L,] monophonic ambience recording (1D numpy array)
        samplerate: int, sample rate in Hz
        array_geom_xyz: [M,3] numpy array of microphone positions in meters

    Returns:
        array_ambience_sigs: [L, M] numpy array, multichannel ambience recordings
    """
    c = 343.0
    nMics = array_geom_xyz.shape[0]
    lFilt = 1024
    nFFT = lFilt
    f = np.arange(0, nFFT // 2 + 1) * samplerate / nFFT
    k = 2 * np.pi * f / c

    # Compute pairwise distances
    D = np.zeros((nMics, nMics))
    for nm in range(nMics):
        D[:, nm] = np.sqrt(np.sum((array_geom_xyz[nm, :] - array_geom_xyz) ** 2, axis=1))

    # Compute isotropic diffuse-field coherence
    DCM = np.zeros((nMics, nMics, len(f)), dtype=np.complex128)
    for nb in range(len(f)):
        DCM[:, :, nb] = np.sinc(k[nb] * D / np.pi)

    # Eigendecomposition and mixing matrix
    C = np.zeros((nMics, nMics, len(f)), dtype=np.complex128)
    for nb in range(len(f)):
        eigvals, V = np.linalg.eigh(DCM[:, :, nb])
        C[:, :, nb] = np.diag(np.sqrt(np.abs(eigvals))) @ V.T

    # Hermitian symmetry for IFFT
    C_fft = np.concatenate((C, np.conj(C[:, :, -2:0:-1])), axis=2)
    C_ss = np.zeros_like(C_fft)
    k_0 = nFFT // 2
    C_ss[:, :, k_0] = C_fft[:, :, k_0]

    # Smooth upwards
    for kk in range(k_0 + 1, nFFT):
        R = C_ss[:, :, kk - 1] @ C_fft[:, :, kk].T
        W, _, Z = np.linalg.svd(R)
        Uf_kk = W @ Z
        C_ss[:, :, kk] = Uf_kk @ C_fft[:, :, kk]
    # Smooth downwards
    for kk in range(k_0 - 1, -1, -1):
        R = C_ss[:, :, kk + 1] @ C_fft[:, :, kk].T
        W, _, Z = np.linalg.svd(R)
        Uf_kk = W @ Z
        C_ss[:, :, kk] = Uf_kk @ C_fft[:, :, kk]

    # Mixing matrices
    MMtx = np.zeros((nMics, nMics, nFFT), dtype=np.complex128)
    for nb in range(nFFT):
        MMtx[:, :, nb] = C_ss[:, :, nb].T
    MMtx[:, :, 0] = np.real(MMtx[:, :, 0])

    # Decorrelate mono signal M times
    lDec = int(0.08 * samplerate)
    fDec = np.arange(0, lDec // 2 + 1) * samplerate / lDec
    HDec = np.exp(1j * (2 * np.pi * np.random.rand(len(fDec), nMics - 1)))
    HDec[0, :] = 1
    HDec[-1, :] = 1
    HDec_full = np.concatenate([HDec, np.conj(HDec[-2:0:-1, :])], axis=0)
    hDec = np.fft.ifft(HDec_full, axis=0).real

    # Decorrelated signals
    dec_mono_sigs = np.stack(
        [fftconvolve(mono_ambience_sig, hDec[:, i], mode="full")[: len(mono_ambience_sig)] for i in range(nMics - 1)],
        axis=1,
    )
    dec_mono_sigs = np.concatenate([mono_ambience_sig[:, None], dec_mono_sigs], axis=1)  # [L, M]

    # Create mixing filters from mixing matrices
    hMix = np.fft.ifft(MMtx, axis=2).real
    hMix = np.transpose(hMix, (2, 1, 0))  # [lFilt, nMics, nMics]
    hMix = np.fft.fftshift(hMix, axes=0)

    # Apply mixing filters
    L = len(mono_ambience_sig)
    array_ambience_sigs = np.zeros((L, nMics))
    for nm in range(nMics):
        # Sum over all decorrelated signals, each filtered by hMix
        sigs = [fftconvolve(dec_mono_sigs[:, i], hMix[:, i, nm], mode="full")[:L] for i in range(nMics)]
        array_ambience_sigs[:, nm] = np.sum(sigs, axis=0)

    return array_ambience_sigs
