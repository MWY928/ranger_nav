#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure-Python temporal filter for categorical robot actions."""

from dataclasses import dataclass
import math
import threading
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class ActionFilterResult:
    """One action-filter update and the state used to make the decision."""

    action_id: int
    raw_argmax: int
    candidate_action: int
    switched: bool
    alpha: float
    filtered_probs: Tuple[float, ...]


class ActionProbabilityFilter:
    """EMA categorical probabilities and debounce changes in wall-clock time."""

    def __init__(
        self,
        action_count: int,
        tau_sec: float,
        switch_margin: float,
        switch_hold_sec: float,
        stop_hold_sec: float,
        stop_action_id: int = 0,
    ):
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        if not 0 <= stop_action_id < action_count:
            raise ValueError("stop_action_id is outside the action space")
        for name, value in (
            ("tau_sec", tau_sec),
            ("switch_margin", switch_margin),
            ("switch_hold_sec", switch_hold_sec),
            ("stop_hold_sec", stop_hold_sec),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "{} must be finite and non-negative".format(name)
                )

        self.action_count = int(action_count)
        self.tau_sec = float(tau_sec)
        self.switch_margin = float(switch_margin)
        self.switch_hold_sec = float(switch_hold_sec)
        self.stop_hold_sec = float(stop_hold_sec)
        self.stop_action_id = int(stop_action_id)
        self._lock = threading.RLock()

        self.filtered_probs: Optional[Tuple[float, ...]] = None
        self.stable_action: Optional[int] = None
        self.pending_action: Optional[int] = None
        self.pending_since_sec: Optional[float] = None
        self.last_update_sec: Optional[float] = None

    def reset(self, stable_action: Optional[int] = None) -> None:
        """Clear EMA history, optionally forcing the currently executed action."""
        if (
            stable_action is not None
            and not 0 <= stable_action < self.action_count
        ):
            raise ValueError("stable_action is outside the action space")
        with self._lock:
            self.filtered_probs = None
            self.stable_action = stable_action
            self.pending_action = None
            self.pending_since_sec = None
            self.last_update_sec = None

    def force_action(self, action_id: int) -> None:
        """Immediately make a safety action stable without filtering it."""
        self.reset(stable_action=int(action_id))

    def update(
        self, probabilities: Iterable[float], now_sec: float
    ) -> ActionFilterResult:
        probs = self._normalize_probabilities(probabilities)
        now_sec = float(now_sec)
        if not math.isfinite(now_sec):
            raise ValueError("now_sec must be finite")

        with self._lock:
            alpha = self._update_ema(probs, now_sec)
            raw_argmax = self._argmax(probs)
            candidate = self._argmax(self.filtered_probs)
            switched = False

            if self.stable_action is None:
                self.stable_action = candidate
                self._clear_pending()
            elif candidate == self.stable_action:
                self._clear_pending()
            elif (
                self.filtered_probs[candidate]
                < self.filtered_probs[self.stable_action] + self.switch_margin
            ):
                self._clear_pending()
            else:
                if candidate != self.pending_action:
                    self.pending_action = candidate
                    self.pending_since_sec = now_sec

                hold_sec = (
                    self.stop_hold_sec
                    if candidate == self.stop_action_id
                    else self.switch_hold_sec
                )
                if now_sec - self.pending_since_sec >= hold_sec:
                    self.stable_action = candidate
                    switched = True
                    self._clear_pending()

            return ActionFilterResult(
                action_id=self.stable_action,
                raw_argmax=raw_argmax,
                candidate_action=candidate,
                switched=switched,
                alpha=alpha,
                filtered_probs=self.filtered_probs,
            )

    def _normalize_probabilities(
        self, probabilities: Iterable[float]
    ) -> Tuple[float, ...]:
        probs = tuple(float(value) for value in probabilities)
        if len(probs) != self.action_count:
            raise ValueError(
                "expected {} action probabilities, got {}".format(
                    self.action_count, len(probs)
                )
            )
        if any(not math.isfinite(value) or value < 0.0 for value in probs):
            raise ValueError(
                "action probabilities must be finite and non-negative"
            )
        total = sum(probs)
        if total <= 0.0:
            raise ValueError("action probabilities must have a positive sum")
        return tuple(value / total for value in probs)

    def _update_ema(self, probs: Tuple[float, ...], now_sec: float) -> float:
        if self.filtered_probs is None or self.last_update_sec is None:
            self.filtered_probs = probs
            alpha = 1.0
        else:
            dt = max(0.0, now_sec - self.last_update_sec)
            alpha = (
                1.0 if self.tau_sec == 0.0 else -math.expm1(-dt / self.tau_sec)
            )
            self.filtered_probs = tuple(
                (1.0 - alpha) * previous + alpha * current
                for previous, current in zip(self.filtered_probs, probs)
            )
        self.last_update_sec = now_sec
        return alpha

    @staticmethod
    def _argmax(values: Tuple[float, ...]) -> int:
        return max(range(len(values)), key=values.__getitem__)

    def _clear_pending(self) -> None:
        self.pending_action = None
        self.pending_since_sec = None
