# =============================================================================
# env/terrain_generator.py — Procedural chunk-based terrain generator
# =============================================================================
# Generates terrain as a sequence of (x, y) polygon vertices that can be
# consumed by Box2D edge chains.  Terrain is emitted in chunks, enabling
# an infinite rolling window as the agent moves forward.
#
# Terrain types (each generates a list of (x,y) points):
#   flat     – constant height baseline
#   uneven   – smooth Perlin-like noise via sinusoidal superposition
#   slope    – linear incline or decline
#   stairs   – discrete step-function ascent / descent
# =============================================================================

from __future__ import annotations

import math
import random
from enum import Enum
from typing import List, Tuple

import numpy as np

# Type alias for a list of 2-D world-space points (x, y)
TerrainPoints = List[Tuple[float, float]]


class TerrainType(Enum):
    FLAT    = "flat"
    UNEVEN  = "uneven"
    SLOPE   = "slope"
    STAIRS  = "stairs"


class TerrainChunk:
    """A single contiguous section of terrain with a known start/end x."""

    def __init__(
        self,
        points: TerrainPoints,
        terrain_type: TerrainType,
        x_start: float,
        x_end: float,
    ):
        self.points       = points        # list of (x, y) world vertices
        self.terrain_type = terrain_type
        self.x_start      = x_start
        self.x_end        = x_end

    def __repr__(self) -> str:
        return (
            f"TerrainChunk(type={self.terrain_type.value}, "
            f"x=[{self.x_start:.1f}, {self.x_end:.1f}])"
        )


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class TerrainGenerator:
    """
    Generates and manages a rolling window of terrain chunks.

    Parameters
    ----------
    chunk_length        : horizontal width (m) of each terrain chunk
    point_spacing       : horizontal distance between vertices (m)
    max_height_variation: peak-to-peak height noise for UNEVEN terrain (m)
    slope_range         : (min, max) slope angle in degrees for SLOPE terrain
    step_height         : riser height (m) per step in STAIRS terrain
    step_width          : tread width (m) per step in STAIRS terrain
    difficulty_level    : 0.0–1.0.  Controls curriculum (see schedule below)
    randomise_after_warmup: if True, randomly pick terrain type once
                            difficulty_level > WARMUP_THRESHOLD
    seed                : optional RNG seed for reproducibility
    """

    WARMUP_THRESHOLD = 0.3   # difficulty below this → always FLAT

    # Curriculum schedule: difficulty → allowed terrain types (in order)
    CURRICULUM = [
        (0.0,  [TerrainType.FLAT]),
        (0.3,  [TerrainType.FLAT, TerrainType.UNEVEN]),
        (0.55, [TerrainType.FLAT, TerrainType.UNEVEN, TerrainType.SLOPE]),
        (0.75, [TerrainType.FLAT, TerrainType.UNEVEN,
                TerrainType.SLOPE, TerrainType.STAIRS]),
    ]

    def __init__(
        self,
        chunk_length: float         = 20.0,
        point_spacing: float        = 0.5,
        max_height_variation: float = 1.5,
        slope_range: Tuple[float, float] = (-12.0, 12.0),
        step_height: float          = 0.4,
        step_width: float           = 1.2,
        difficulty_level: float     = 0.0,
        randomise_after_warmup: bool = True,
        base_y: float               = 0.0,
        seed: int | None            = None,
    ):
        self.chunk_length         = chunk_length
        self.point_spacing        = point_spacing
        self.max_height_variation = max_height_variation
        self.slope_range          = slope_range
        self.step_height          = step_height
        self.step_width           = step_width
        self.difficulty_level     = difficulty_level
        self.randomise_after_warmup = randomise_after_warmup
        self.base_y               = base_y
        self.rng                  = random.Random(seed)
        self.np_rng               = np.random.default_rng(seed)

        # State –– maintained across resets
        self._chunks: List[TerrainChunk] = []
        self._cursor_x: float = 0.0   # x-coord where next chunk begins
        self._cursor_y: float = base_y
        self._type_index: int = 0     # track current terrain type in curriculum

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, difficulty_level: float | None = None):
        """Clear all chunks and restart generation from x=0."""
        if difficulty_level is not None:
            self.difficulty_level = float(np.clip(difficulty_level, 0.0, 1.0))
        self._chunks.clear()
        self._cursor_x = 0.0
        self._cursor_y = self.base_y
        self._type_index = 0

    def set_difficulty(self, level: float):
        """Update difficulty (0.0 = easy, 1.0 = hardest)."""
        self.difficulty_level = float(np.clip(level, 0.0, 1.0))

    def generate_next_chunk(self) -> TerrainChunk:
        """Generate and store the next terrain chunk, returning it."""
        terrain_type = self._pick_terrain_type()
        x0 = self._cursor_x
        y0 = self._cursor_y

        if terrain_type == TerrainType.FLAT:
            points, y_end = self._gen_flat(x0, y0)
        elif terrain_type == TerrainType.UNEVEN:
            points, y_end = self._gen_uneven(x0, y0)
        elif terrain_type == TerrainType.SLOPE:
            points, y_end = self._gen_slope(x0, y0)
        else:  # STAIRS
            points, y_end = self._gen_stairs(x0, y0)

        x_end = points[-1][0]

        chunk = TerrainChunk(
            points=points,
            terrain_type=terrain_type,
            x_start=x0,
            x_end=x_end,
        )
        self._chunks.append(chunk)
        self._cursor_x = x_end
        self._cursor_y  = y_end
        return chunk

    def generate_initial_chunks(self, n: int = 5) -> List[TerrainChunk]:
        """Pre-generate n chunks (e.g. at reset time)."""
        return [self.generate_next_chunk() for _ in range(n)]

    def advance_if_needed(self, agent_x: float, lookahead: float = 40.0) -> List[TerrainChunk]:
        """
        If the furthest existing chunk is within `lookahead` metres of
        agent_x, generate new chunks.  Returns list of *new* chunks only.
        """
        new_chunks: List[TerrainChunk] = []
        while self._cursor_x - agent_x < lookahead:
            new_chunks.append(self.generate_next_chunk())
        return new_chunks

    def prune_behind(self, agent_x: float, tail: float = 20.0):
        """
        Remove chunks that end more than `tail` metres behind agent_x.
        Returns list of pruned chunks (caller can destroy Box2D bodies).
        """
        cutoff = agent_x - tail
        pruned = [c for c in self._chunks if c.x_end < cutoff]
        self._chunks = [c for c in self._chunks if c.x_end >= cutoff]
        return pruned

    @property
    def active_chunks(self) -> List[TerrainChunk]:
        return list(self._chunks)

    def all_points(self) -> TerrainPoints:
        """Flatten all active chunks into one sorted point list."""
        pts: TerrainPoints = []
        for chunk in self._chunks:
            pts.extend(chunk.points)
        return pts

    # ------------------------------------------------------------------
    # Terrain type selector (curriculum / random)
    # ------------------------------------------------------------------

    def _allowed_types(self) -> List[TerrainType]:
        allowed = [TerrainType.FLAT]
        for threshold, types in self.CURRICULUM:
            if self.difficulty_level >= threshold:
                allowed = types
        return allowed

    def _pick_terrain_type(self) -> TerrainType:
        # User requested infinite deterministic loop of flat -> uneven -> stairs.
        sequence = [TerrainType.FLAT, TerrainType.UNEVEN, TerrainType.STAIRS]
        chosen_type = sequence[self._type_index % len(sequence)]
        self._type_index += 1
        return chosen_type

    # ------------------------------------------------------------------
    # Per-type generators
    # Each returns (points: TerrainPoints, y_at_end: float)
    # ------------------------------------------------------------------

    def _linspace_x(self, x0: float) -> np.ndarray:
        n = max(2, int(self.chunk_length / self.point_spacing) + 1)
        return np.linspace(x0, x0 + self.chunk_length, n)

    def _gen_flat(self, x0: float, y0: float) -> Tuple[TerrainPoints, float]:
        xs = self._linspace_x(x0)
        points = [(float(x), y0) for x in xs]
        return points, y0

    def _gen_uneven(self, x0: float, y0: float) -> Tuple[TerrainPoints, float]:
        xs = self._linspace_x(x0)
        amp   = self.max_height_variation * self.difficulty_level
        # Superimpose 3 sinusoids with random phases/frequencies
        ys = np.zeros(len(xs))
        for _ in range(3):
            freq  = self.rng.uniform(0.1, 0.5)
            phase = self.rng.uniform(0, 2 * math.pi)
            ys   += self.rng.uniform(0.2, 0.5) * amp * np.sin(2 * math.pi * freq * (xs - x0) + phase)
        ys = y0 + ys
        points = [(float(x), float(y)) for x, y in zip(xs.tolist(), ys.tolist())]
        return points, float(ys[-1])

    def _gen_slope(self, x0: float, y0: float) -> Tuple[TerrainPoints, float]:
        min_deg, max_deg = self.slope_range
        angle   = self.rng.uniform(min_deg, max_deg)
        # Scale slope with difficulty
        angle  *= self.difficulty_level
        slope   = math.tan(math.radians(angle))
        xs      = self._linspace_x(x0)
        ys      = y0 + slope * (xs - x0)
        points  = [(float(x), float(y)) for x, y in zip(xs.tolist(), ys.tolist())]
        return points, float(ys[-1])

    def _gen_stairs(self, x0: float, y0: float) -> Tuple[TerrainPoints, float]:
        """Discrete step function — alternating horizontal/vertical segments."""
        step_h = self.step_height * self.difficulty_level
        direction = self.rng.choice([-1, 1])   # +1 ascent, -1 descent
        points: TerrainPoints = []
        x = x0
        y = y0
        points.append((x, y))
        while x < x0 + self.chunk_length:
            # Horizontal tread
            x_next = x + self.step_width
            points.append((x_next, y))
            # Vertical riser
            y += direction * step_h
            points.append((x_next, y))
            x = x_next
        return points, y
