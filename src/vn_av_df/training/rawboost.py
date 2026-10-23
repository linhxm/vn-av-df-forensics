"""RawBoost algo 3 (nhiễu cộng dừng, không phụ thuộc tín hiệu) cho BL-A XLS-R + AASIST.

Chép từ github.com/TakHemlata/SSL_Anti-spoofing (RawBoost.py, giấy phép MIT, © 2022 Hemlata Tak),
chỉ giữ thuật toán 3 mà script DF của họ dùng mặc định (`--algo 3`) và thêm `rng` để tái lập.
Tak et al., "RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker
Verification Anti-Spoofing", ICASSP 2022.
"""

import numpy as np

# Mặc định của main_SSL_DF.py.
DEFAULTS = dict(
    SNRmin=10,
    SNRmax=40,
    nBands=5,
    minF=20,
    maxF=8000,
    minBW=100,
    maxBW=1000,
    minCoeff=10,
    maxCoeff=100,
    minG=0,
    maxG=0,
)


def norm_wav(x, always):
    peak = np.amax(np.abs(x))
    if always or peak > 1:
        x = x / max(peak, 1e-12)
    return x


def notch_coeffs(rng, nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs):
    from scipy import signal

    b = 1
    for _ in range(nBands):
        fc = rng.uniform(minF, maxF)
        bw = rng.uniform(minBW, maxBW)
        c = int(rng.uniform(minCoeff, maxCoeff))
        if c / 2 == int(c / 2):
            c = c + 1
        f1, f2 = fc - bw / 2, fc + bw / 2
        f1 = 1 / 1000 if f1 <= 0 else f1
        f2 = fs / 2 - 1 / 1000 if f2 >= fs / 2 else f2
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs), b)
    gain = rng.uniform(minG, maxG)
    _, h = signal.freqz(b, 1, fs=fs)
    return pow(10, gain / 20) * b / np.amax(np.abs(h))


def filter_fir(x, b):
    from scipy import signal

    n = b.shape[0] + 1
    y = signal.lfilter(b, 1, np.pad(x, (0, n), "constant"))
    return y[int(n / 2) : int(y.shape[0] - n / 2)]


def ssi_additive_noise(x, rng, fs=16000, **options):
    o = {**DEFAULTS, **options}
    noise = rng.normal(0, 1, x.shape[0])
    b = notch_coeffs(
        rng,
        o["nBands"],
        o["minF"],
        o["maxF"],
        o["minBW"],
        o["maxBW"],
        o["minCoeff"],
        o["maxCoeff"],
        o["minG"],
        o["maxG"],
        fs,
    )
    noise = norm_wav(filter_fir(noise, b), True)
    snr = rng.uniform(o["SNRmin"], o["SNRmax"])
    noise = (
        noise / max(np.linalg.norm(noise, 2), 1e-12) * np.linalg.norm(x, 2) / 10.0 ** (0.05 * snr)
    )
    return (x + noise).astype(np.float32)
