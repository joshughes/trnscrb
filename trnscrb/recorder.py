"""Audio capture using sounddevice.

Supports mic-only or BlackHole 2ch (system audio) as input.
Records at 16 kHz mono — the sample rate Whisper expects.
"""
import threading
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile

SAMPLE_RATE = 16_000  # Whisper expects 16 kHz


class Recorder:
    def __init__(self, device: int | str | None = None):
        # device=None → system default input
        self.device = device
        self._recording = False
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    # ── public ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._frames = []
        self._recording = True
        # Query actual channel count — aggregate devices may have more than 1.
        # Opening with fewer channels than the device supports causes an AUHAL
        # error (-10863) when another app already has the device open.
        if self.device is not None:
            info = sd.query_devices(self.device)
            channels = min(info["max_input_channels"], 2)
        else:
            channels = 1
        self._channels = channels
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=SAMPLE_RATE,
            channels=channels,
            dtype="float32",
            callback=self._callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self) -> Path | None:
        """Stop recording and return the path to a temporary WAV file."""
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            frames = list(self._frames)

        if not frames:
            return None

        audio = np.concatenate(frames, axis=0)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mix multi-channel down to mono for Whisper
        audio = audio.flatten()
        audio_int16 = (audio * 32_767).astype(np.int16)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wavfile.write(tmp.name, SAMPLE_RATE, audio_int16)
        return Path(tmp.name)

    @property
    def is_recording(self) -> bool:
        return self._recording

    # ── helpers ─────────────────────────────────────────────────────────────

    def _callback(self, indata, frames, time_info, status):
        if self._recording:
            with self._lock:
                self._frames.append(indata.copy())

    # ── class-level utilities ────────────────────────────────────────────────

    @staticmethod
    def find_blackhole_device() -> int | None:
        for i, dev in enumerate(sd.query_devices()):
            if "BlackHole" in dev["name"] and dev["max_input_channels"] > 0:
                return i
        return None

    @staticmethod
    def find_device_by_name(name: str) -> int | None:
        """Find an input device by exact name. Returns None if not found."""
        for i, dev in enumerate(sd.query_devices()):
            if dev["name"] == name and dev["max_input_channels"] > 0:
                return i
        return None

    @staticmethod
    def list_input_devices() -> list[dict]:
        return [
            {"index": i, "name": dev["name"], "channels": dev["max_input_channels"]}
            for i, dev in enumerate(sd.query_devices())
            if dev["max_input_channels"] > 0
        ]
