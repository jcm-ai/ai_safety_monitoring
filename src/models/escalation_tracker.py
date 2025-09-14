from __future__ import annotations
from collections import deque
from typing import Deque, Dict, Any, List

class EscalationTracker:
    """
    Tracks rolling risk trend using EWMA and slope over recent scores.
    Scores can be max risk from model outputs or policy-computed risk.
    """

    def __init__(self, ewma_alpha: float = 0.3, slope_window: int = 5, risk_floor: float = 0.05):
        self.alpha = ewma_alpha
        self.window = slope_window
        self.risk_floor = risk_floor
        self.history: Deque[float] = deque(maxlen=self.window)
        self.ewma = 0.0

    def update(self, risk_score: float) -> Dict[str, Any]:
        """
        Update tracker with new risk score and return current trend metrics.

        Args:
            risk_score: Float between 0 and 1

        Returns:
            Dict with ewma, slope, and history
        """
        r = max(risk_score, self.risk_floor)
        self.ewma = self.alpha * r + (1 - self.alpha) * self.ewma
        self.history.append(r)
        slope = self._slope(list(self.history))
        return {
            "ewma": self.ewma,
            "slope": slope,
            "history": list(self.history)
        }

    def _slope(self, arr: List[float]) -> float:
        """
        Compute linear slope of recent scores.

        Args:
            arr: List of floats

        Returns:
            Slope value
        """
        n = len(arr)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2
        y_mean = sum(arr) / n
        num = sum((i - x_mean) * (arr[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n)) or 1.0
        return num / den
