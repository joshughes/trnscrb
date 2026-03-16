"""Audio capture using sounddevice.

Supports mic-only or BlackHole 2ch (system audio) as input.
Records at 16 kHz mono — the sample rate Whisper expects.

Audio is streamed directly to a temp WAV file via a drain thread so nothing
is buffered in memory. The callback enqueues raw chunks; the drain thread
converts to int16 and writes frames. On stop(), the file is closed and
returned immediately.
"""
import queue
import threading
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000  # Whisper expects 16 kHz
_WAV_HEADER_BYTES = 44


class Recorder:
    def __init__(self, device: int | str | None = None):
        # device=None → system default input
        self.device   = device
        self._recording = False
        self._stream:       sd.InputStream | None = None
        self._wav:          wave.Wave_write | None = None
        self._tmp_path:     Path | None = None
        self._queue:        queue.Queue = queue.Queue()
        self._drain_thread: threading.Thread | None = None
        self._channels = 1

    # ── public ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        # Query actual channel count — aggregate devices may have more than 1.
        # Opening with fewer channels than the device supports causes an AUHAL
        # error (-10863) when another app already has the device open.
        if self.device is not None:
            info = sd.query_devices(self.device)
            self._channels = min(info["max_input_channels"], 2)
        else:
            self._channels = 1

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self._tmp_path = Path(tmp.name)
        tmp.close()

        self._wav = wave.open(str(self._tmp_path), "wb")
        self._wav.setnchannels(1)   # always mono — mixed down in drain thread
        self._wav.setsampwidth(2)   # int16
        self._wav.setframerate(SAMPLE_RATE)

        self._recording = True
        self._drain_thread = threading.Thread(target=self._drain, daemon=True)
        self._drain_thread.start()

        self._stream = sd.InputStream(
            device=self.device,
            samplerate=SAMPLE_RATE,
            channels=self._channels,
            dtype="float32",
            callback=self._callback,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self) -> Path | None:
        """Stop recording, flush to disk, and return the WAV path."""
        self._recording = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Signal drain thread to finish and wait for it to flush
        self._queue.put(None)
        if self._drain_thread:
            self._drain_thread.join(timeout=10)
            self._drain_thread = None

        if self._wav:
            self._wav.close()
            self._wav = None

        path, self._tmp_path = self._tmp_path, None

        if path and path.exists() and path.stat().st_size > _WAV_HEADER_BYTES:
            return path

        if path:
            path.unlink(missing_ok=True)
        return None

    @property
    def is_recording(self) -> bool:
        return self._recording

    # ── helpers ─────────────────────────────────────────────────────────────

    def _callback(self, indata, frames, time_info, status):
        """Real-time audio callback — only enqueues, never does I/O."""
        if self._recording:
            self._queue.put(indata.copy())

    def _drain(self) -> None:
        """Background thread: dequeue chunks, mix to mono, write to WAV."""
        while True:
            chunk = self._queue.get()
            if chunk is None:
                break
            if chunk.ndim > 1:
                chunk = chunk.mean(axis=1)  # mix multi-channel down to mono
            pcm = (chunk * 32_767).clip(-32_768, 32_767).astype(np.int16)
            if self._wav:
                self._wav.writeframes(pcm.tobytes())

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
