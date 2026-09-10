"""
AngleBuffer

A small fixed-size moving-average buffer used to smooth noisy per-frame
head pose angles (pitch, yaw, roll) before they're used for calibration,
intent classification, or display. Reduces frame-to-frame jitter from the
face-mesh/solvePnP pipeline without adding much lag (tune `size` to trade
smoothness against responsiveness).

Usage:
    buffer = AngleBuffer(size=10)
    buffer.add([pitch, yaw, roll])
    pitch, yaw, roll = buffer.get_average()
"""

from collections import deque
import numpy as np


class AngleBuffer:
    def __init__(self, size=10):
        """
        Args:
            size (int): number of most recent samples to average over.
                Larger = smoother but laggier; smaller = more responsive
                but noisier.
        """
        if size < 1:
            raise ValueError("AngleBuffer size must be at least 1.")
        self.size = size
        self._buffer = deque(maxlen=size)

    def add(self, angles):
        """Add a new [pitch, yaw, roll] sample (any 3-element sequence)."""
        self._buffer.append(np.asarray(angles, dtype=np.float64))

    def get_average(self):
        """Return the mean [pitch, yaw, roll] over the current buffer.

        Returns:
            np.ndarray of shape (3,). If nothing has been added yet,
            returns [0.0, 0.0, 0.0].
        """
        if not self._buffer:
            return np.zeros(3, dtype=np.float64)
        return np.mean(np.vstack(self._buffer), axis=0)

    def clear(self):
        """Reset the buffer (e.g. after recalibration)."""
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)
