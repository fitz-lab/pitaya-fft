#!/usr/bin/env python3
###############################################################################
#Mock tester for Red Pitaya Doppler-FFT pipeline.
###############################################################################
import time
import math
import numpy as np

###############################################################################
# Toggler for using mock or real Red Pitaya
###############################################################################
USE_MOCK_RP = True

try:
    import rp as _real_rp
    if not USE_MOCK_RP:
        rp = _real_rp
except Exception:
    # Fall back to mock automatically if real rp import fails
    USE_MOCK_RP = True

###############################################################################
# Mock Red Pitaya module
###############################################################################
if USE_MOCK_RP:
    class _FBuffer:
        """Minimal fBuffer replacement with indexing and slicing."""
        def __init__(self, N):
            self._buf = np.zeros(N, dtype=np.float32)
        def __getitem__(self, idx):
            return self._buf[idx]
        def __setitem__(self, idx, val):
            self._buf[idx] = val
        def __len__(self):
            return len(self._buf)
        def __iter__(self):
            return iter(self._buf)
        def __getslice__(self, i, j):
            return self._buf[i:j]
        def __array__(self, *args, **kwargs):
            return self._buf
        def __repr__(self):
            return f"_FBuffer(len={len(self._buf)})"
        def __copy__(self):
            out = _FBuffer(len(self._buf))
            out._buf[:] = self._buf
            return out
        def __reduce__(self):
            return (_FBuffer, (len(self._buf),))

    class _MockRP:
        # Constants used by the code
        RP_DEC_1   = 1
        RP_DEC_2   = 2
        RP_CH_1    = 1
        RP_CH_2    = 2

        # Trigger sources / states / channels
        RP_TRIG_SRC_EXT_PE = 101
        RP_TRIG_STATE_TRIGGERED = 201
        RP_T_CH_1 = 301

        # Mock config state
        _dec = RP_DEC_2
        _trig_src = RP_TRIG_SRC_EXT_PE
        _trig_level = 1e-7
        _trig_delay = 0
        _started = False

        # Synthetic data store (filled by tester)
        _synth_I = None   # shape: (NumPulses, N)
        _synth_Q = None   # shape: (NumPulses, N)
        _pulse_idx = 0

        # API used by the code
        @staticmethod
        def rp_Init(): pass
        @staticmethod
        def rp_AcqReset(): pass
        @staticmethod
        def rp_AcqSetDecimation(dec): _MockRP._dec = dec
        @staticmethod
        def rp_AcqSetTriggerLevel(ch, lvl): _MockRP._trig_level = lvl
        @staticmethod
        def rp_AcqSetTriggerDelay(dly): _MockRP._trig_delay = dly
        @staticmethod
        def rp_AcqStart(): _MockRP._started = True
        @staticmethod
        def rp_AcqSetTriggerSrc(src): _MockRP._trig_src = src
        @staticmethod
        def rp_AcqGetTriggerState():
            # Instantly "triggered" for the mock
            return (0, _MockRP.RP_TRIG_STATE_TRIGGERED)
        @staticmethod
        def rp_AcqGetBufferFillState():
            # Instantly "filled" for the mock
            return (0, True)
        @staticmethod
        def rp_AcqGetDataV(ch, start_idx, N, fbuff):
            # Copy the current pulse samples into fbuff
            k = _MockRP._pulse_idx
            if _MockRP._synth_I is None or _MockRP._synth_Q is None:
                raise RuntimeError("MockRP: synthetic data not initialized")
            if ch == _MockRP.RP_CH_1:
                src = _MockRP._synth_I[k, :]
            elif ch == _MockRP.RP_CH_2:
                src = _MockRP._synth_Q[k, :]
            else:
                raise ValueError("MockRP: invalid channel")
            try:
                fbuff._buf[:] = src
            except AttributeError:
                for i in range(N):
                    fbuff[i] = float(src[i])
            return 0
        @staticmethod
        def rp_Release(): pass

        # fBuffer factory
        @staticmethod
        def fBuffer(N): return _FBuffer(N)

    rp = _MockRP


###############################################################################
# Test configuration
###############################################################################
currentPRF = 4000
dec = rp.RP_DEC_2

trig_lvl = 1e-7
acq_trig_sour = getattr(rp, "RP_TRIG_SRC_EXT_PE", None)

N = 1250                        # samples per pulse (range bins)
trig_dly = int(N/2) - 1
NumPulses = 256                 # pulses for Doppler FFT

DOPPLER_HZ = 500.0              # expected Doppler frequency (peak we test for)
SNR_DB = 30.0                   # per-bin SNR for the tone
RANGE_BIN_PEAK = 10             # give one range bin a strong target
RNG_PROFILE_DB = -40.0

def _make_synthetic_iq(NumPulses, N, prf_hz, f_dop_hz, rng_bin_peak, snr_db, bg_db):
    k = np.arange(NumPulses, dtype=np.float32)
    phi = 2 * np.pi * (f_dop_hz / prf_hz) * k
    tone = np.exp(1j * phi).astype(np.complex64)

    amp_peak = 1.0
    amp_bg = 10.0 ** (bg_db / 20.0)  # amplitude ratio
    profile = np.full(N, amp_bg, dtype=np.float32)
    profile[rng_bin_peak] = amp_peak

    # Outer product to build (NumPulses, N) complex signal
    sig = tone[:, None] * profile[None, :]

    # Add white noise to reach desired SNR at the peak bin
    snr_lin = 10.0 ** (snr_db / 10.0)
    peak_power = (amp_peak ** 2)
    noise_power = peak_power / snr_lin
    noise_sigma = math.sqrt(noise_power / 2.0)  # per real/imag component

    noise = noise_sigma * (np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape)).astype(np.complex64)
    sig_noisy = sig + noise

    # Split into I/Q (float32)
    I = sig_noisy.real.astype(np.float32)
    Q = sig_noisy.imag.astype(np.float32)
    return I, Q

###############################################################################
# Test driver
###############################################################################
def main(plot=False):
    if USE_MOCK_RP:
        I, Q = _make_synthetic_iq(
            NumPulses, N, currentPRF, DOPPLER_HZ,
            RANGE_BIN_PEAK, SNR_DB, RNG_PROFILE_DB
        )
        rp._synth_I = I
        rp._synth_Q = Q
        rp._pulse_idx = 0

    rp.rp_Init()
    rp.rp_AcqReset()
    rp.rp_AcqSetDecimation(dec)
    rp.rp_AcqSetTriggerLevel(rp.RP_T_CH_1, trig_lvl)
    rp.rp_AcqSetTriggerDelay(trig_dly)

    fbuff1 = rp.fBuffer(N)
    fbuff2 = rp.fBuffer(N)

    pulsesI = np.empty((NumPulses, N), dtype=np.float32)
    pulsesQ = np.empty((NumPulses, N), dtype=np.float32)

    print("Acq_start (mock)" if USE_MOCK_RP else "Acq_start")
    rp.rp_AcqStart()
    if acq_trig_sour is not None:
        rp.rp_AcqSetTriggerSrc(acq_trig_sour)

    startT = time.time()
    FirstCaptureT = None

    for k in range(NumPulses):
        # Wait for trigger
        while True:
            state = rp.rp_AcqGetTriggerState()[1]
            if state == getattr(rp, "RP_TRIG_STATE_TRIGGERED", None):
                if FirstCaptureT is None:
                    FirstCaptureT = time.time()
                break

        # Wait for buffer fill
        while True:
            if rp.rp_AcqGetBufferFillState()[1]:
                break

        # Acquire N samples from both channels
        rp.rp_AcqGetDataV(rp.RP_CH_1, 0, N, fbuff1)
        rp.rp_AcqGetDataV(rp.RP_CH_2, 0, N, fbuff2)

        # Copy into arrays
        try:
            pulsesI[k, :] = fbuff1[:]
            pulsesQ[k, :] = fbuff2[:]
        except TypeError:
            for i in range(N):
                pulsesI[k, i] = fbuff1[i]
                pulsesQ[k, i] = fbuff2[i]

        # Advance mock pulse index
        if USE_MOCK_RP:
            rp._pulse_idx = min(rp._pulse_idx + 1, NumPulses - 1)

    endT = time.time()
    rp.rp_Release()

    # ====== FFT Processing ======
    complex_data = pulsesI + 1j * pulsesQ
    win = np.hanning(NumPulses).astype(np.float32)[:, None]
    windowed = complex_data * win
    doppler_fft = np.fft.fftshift(np.fft.fft(windowed, axis=0), axes=0)

    # Frequency bins
    frequency_bins = np.fft.fftfreq(NumPulses, d=1.0 / currentPRF)
    shifted_frequency_bins = np.fft.fftshift(frequency_bins)

    bin_idx = RANGE_BIN_PEAK
    eps = 1e-20
    power_spectrum = 10 * np.log10(np.maximum(np.abs(doppler_fft[:, bin_idx]) ** 2, eps))

    # ====== Test assertion: peak should land near DOPPLER_HZ ======
    raw_bin = int(np.round(DOPPLER_HZ * NumPulses / currentPRF))
    expected_idx = (raw_bin + NumPulses // 2) % NumPulses

    found_idx = int(np.argmax(power_spectrum))
    found_freq = shifted_frequency_bins[found_idx]

    print(f"\nExpected Doppler ~ {DOPPLER_HZ:.1f} Hz "
          f"(bin {expected_idx}), found {found_freq:.1f} Hz (bin {found_idx})")

    assert abs(found_idx - expected_idx) <= 1, (
        f"Peak bin off by more than 1: expected {expected_idx}, got {found_idx}"
    )

    # Timing diagnostics
    DeltaT = endT - startT
    Delta1 = (FirstCaptureT - startT) if FirstCaptureT is not None else float('nan')
    print("First Capture Time =", Delta1)
    print("Time to capture all pulses =", DeltaT)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(shifted_frequency_bins, power_spectrum)
        plt.title('Doppler Spectrum (Mock) @ range bin {}'.format(bin_idx))
        plt.xlabel('Doppler Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main(plot=True)
