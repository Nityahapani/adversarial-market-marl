"""
Prioritized experience replay buffer.

Used for auxiliary training of the belief transformer and MINE estimator,
allowing them to train on diverse historical flow samples rather than
only the most recent rollout.

Implements sum-tree-based O(log N) priority sampling.
Reference: Schaul et al. (2015). Prioritized Experience Replay.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np


class SumTree:
    """
    Binary sum-tree for efficient priority-based sampling.
    Leaf nodes store priorities; internal nodes store their subtree sum.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self._write_ptr = 0
        self._size = 0

    def update(self, idx: int, priority: float) -> None:
        """Update priority at leaf index idx."""
        tree_idx = idx + self.capacity
        self.tree[tree_idx] = priority
        self._propagate(tree_idx)

    def _propagate(self, tree_idx: int) -> None:
        parent = tree_idx // 2
        while parent >= 1:
            left = 2 * parent
            self.tree[parent] = self.tree[left] + self.tree[left + 1]
            parent //= 2

    def total(self) -> float:
        return float(self.tree[1])

    def sample(self, value: float) -> Tuple[int, float]:
        """Find the leaf whose cumulative priority contains `value`."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        leaf_idx = idx - self.capacity
        return leaf_idx, float(self.tree[idx])

    def add(self, priority: float) -> int:
        """Add a new entry with given priority. Returns leaf index."""
        idx = self._write_ptr
        self.update(idx, priority)
        self._write_ptr = (self._write_ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return idx

    @property
    def size(self) -> int:
        return self._size

    @property
    def max_priority(self) -> float:
        if self._size == 0:
            return 1.0
        return float(self.tree[self.capacity : self.capacity + self._size].max() + 1e-8)


class PrioritizedReplayBuffer:
    """
    Prioritized replay buffer for flow sequences and belief transformer training.

    Stores (flow_sequence, label) pairs with priority = |TD error| or
    belief prediction error, enabling focused retraining on hard examples.

    Args:
        capacity:  Maximum number of stored sequences.
        alpha:     Priority exponent (0 = uniform, 1 = full priority).
        beta:      IS correction exponent (annealed from beta_start to 1.0).
    """

    def __init__(
        self,
        capacity: int = 10_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing_steps: int = 1_000_000,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self._beta_start = beta
        self._beta_annealing_steps = beta_annealing_steps
        self._tree = SumTree(capacity)
        self._data: List[Any] = [None] * capacity
        self._step = 0

    def add(self, data: Any, priority: Optional[float] = None) -> None:
        """
        Add a data item with given priority.
        If priority is None, uses the current max priority.
        """
        p = priority if priority is not None else self._tree.max_priority
        idx = self._tree.add(p**self.alpha)
        self._data[idx] = data

    def sample(self, batch_size: int) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions.

        Returns:
            samples:    List of data items.
            indices:    Leaf indices for priority updates.
            weights:    Importance sampling correction weights, shape (B,).
        """
        if self._tree.size == 0:
            raise RuntimeError("Cannot sample from empty buffer.")

        indices = []
        priorities = []
        total = self._tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            val = np.random.uniform(lo, hi)
            idx, prio = self._tree.sample(val)
            indices.append(idx)
            priorities.append(prio)

        # Importance sampling weights
        current_beta = min(
            1.0,
            self._beta_start + (1.0 - self._beta_start) * self._step / self._beta_annealing_steps,
        )
        self._step += 1

        probs = np.array(priorities) / (total + 1e-8)
        weights = (self._tree.size * probs) ** (-current_beta)
        weights /= weights.max()

        samples = [self._data[i] for i in indices]
        return samples, np.array(indices), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities after computing new TD/belief errors."""
        for idx, prio in zip(indices, priorities):
            self._tree.update(int(idx), (float(prio) + 1e-6) ** self.alpha)

    def __len__(self) -> int:
        return self._tree.size
