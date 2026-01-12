"""Elo rating system helpers for ATP match prediction."""

import numpy as np
from collections import defaultdict


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


class EloState:
    def __init__(self, base: float = 1500.0):
        self.base = base
        self.elo_overall = defaultdict(lambda: base)
        self.elo_surface = defaultdict(lambda: defaultdict(lambda: base))

    def get_overall(self, pid: int) -> float:
        return float(self.elo_overall[pid])

    def get_surface(self, pid: int, surface: str) -> float:
        return float(self.elo_surface[pid][surface])

    def update(
        self, winner: int, loser: int, surface: str, k: float
    ) -> tuple[float, float, float, float]:
        # overall
        ew = expected_score(self.elo_overall[winner], self.elo_overall[loser])
        delta_w = k * (1.0 - ew)
        delta_l = -delta_w
        self.elo_overall[winner] += delta_w
        self.elo_overall[loser] += delta_l

        # surface
        ews = expected_score(
            self.elo_surface[winner][surface], self.elo_surface[loser][surface]
        )
        delta_ws = k * (1.0 - ews)
        delta_ls = -delta_ws
        self.elo_surface[winner][surface] += delta_ws
        self.elo_surface[loser][surface] += delta_ls

        return float(delta_w), float(delta_l), float(delta_ws), float(delta_ls)
