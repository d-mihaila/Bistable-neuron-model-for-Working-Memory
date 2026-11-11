"""
board.py (navigation)

A size x size board with solid walls, one agent and one apple at a time. The
apple cells are drawn in advance from the seed, so every network meets the same
apples as long as it eats the same ones (an apple drawn under the agent is
skipped). Same random draws as snake/simple_navigation.NavBoard.
"""

import numpy as np

from network import HEADING_STEPS


class Board:
    def __init__(self, size=10, seed=None, start=None, apple=None, n_draws=2000):
        self.size = size
        self.rng = np.random.default_rng(seed)
        cells = [(x, y) for x in range(size) for y in range(size)]
        if start is None:
            self.pos = cells[int(self.rng.integers(len(cells)))]
            self.rng.integers(4)                 # (a heading draw in the earlier versions; kept so the apples match)
        else:
            self.pos = tuple(start)
        self._draws = [cells[i] for i in self.rng.integers(len(cells), size=n_draws)]
        self._k = 0
        self.apple = tuple(apple) if apple is not None else self.next_apple()
        self.eaten = 0
        self.collisions = 0
        self.steps = 0

    def next_apple(self):
        while True:
            c = self._draws[self._k % len(self._draws)]
            self._k += 1
            if c != self.pos:
                return c

    def inside(self, c):
        return 0 <= c[0] < self.size and 0 <= c[1] < self.size

    def step(self, k):
        """One step in compass direction k (0 E, 1 N, 2 W, 3 S). Into a wall the agent
        stays and it counts as a collision. Returns dict(moved, ate, collision)."""
        sx, sy = HEADING_STEPS[int(k)]
        new = (self.pos[0] + sx, self.pos[1] + sy)
        self.steps += 1
        if not self.inside(new):
            self.collisions += 1
            return dict(moved=False, ate=False, collision=True)
        self.pos = new
        if new == self.apple:
            self.eaten += 1
            self.apple = self.next_apple()
            return dict(moved=True, ate=True, collision=False)
        return dict(moved=True, ate=False, collision=False)
