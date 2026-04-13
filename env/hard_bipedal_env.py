# =============================================================================
# env/hard_bipedal_env.py — BipedalWalker-v3 with procedural terrain +
#                           external disturbances
# =============================================================================
# Strategy
# --------
# BipedalWalker stores its terrain as a list of Box2D edge-chain bodies
# (self.terrain) and a parallel list of edge normals/friction (self.fd_edge,
# self.fd_polygon).  We subclass the raw BipedalWalker class (not the
# Gymnasium wrapper) so we can override _generate_terrain() and step().
#
# Because Gymnasium wraps BipedalWalker with TimeLimit internally when you
# call gym.make(), we instead instantiate the unwrapped class directly and
# re-wrap it ourselves so the pipeline stays clean.
# =============================================================================

from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

from gymnasium.envs.box2d.bipedal_walker import (
    BipedalWalker, SCALE, TERRAIN_STEP, TERRAIN_HEIGHT, VIEWPORT_W, VIEWPORT_H
)

# ---------------------------------------------------------------------------
# Optional: import Box2D for applying forces.  If it isn't installed we skip
# the disturbance smoothly.
# ---------------------------------------------------------------------------
try:
    import Box2D.b2 as b2
    _BOX2D_AVAILABLE = True
except ImportError:
    _BOX2D_AVAILABLE = False

from env.terrain_generator import TerrainGenerator, TerrainType


# ---------------------------------------------------------------------------
# Hard BipedalWalker
# ---------------------------------------------------------------------------

class HardBipedalEnv(BipedalWalker):
    """
    BipedalWalker-v3 extended with:
      - Chunk-based procedural terrain generation (rolling window)
      - Curriculum scheduler (flat → uneven → slope → stairs)
      - External disturbance forces applied to the hull

    Parameters
    ----------
    enable_terrain_generation : bool   – activate procedural terrain
    enable_disturbance        : bool   – activate random force pushes
    terrain_difficulty_level  : float  – initial difficulty (0–1)
    chunk_length              : float  – width of each terrain chunk (m)
    max_height_variation      : float  – peak height variation for uneven (m)
    slope_range               : tuple  – (min°, max°) for slope terrain
    step_height               : float  – riser height for stairs (m)
    disturbance_force_range   : tuple  – lateral force (N) applied per push
    disturbance_frequency     : float  – probability per step of a push
    randomise_terrain         : bool   – randomly pick terrain type in curriculum
    seed                      : int    – RNG seed (passed to base class too)
    """

    # How many chunks to pre-generate at reset
    INITIAL_CHUNKS = 6
    # How far ahead to maintain chunk coverage (m)
    LOOKAHEAD      = 50.0
    # How far behind to keep (pruning threshold, m)
    TAIL_KEEP      = 25.0

    def __init__(
        self,
        render_mode: Optional[str]      = None,
        enable_terrain_generation: bool = True,
        enable_disturbance: bool        = True,
        terrain_difficulty_level: float = 0.0,
        chunk_length: float             = 20.0,
        max_height_variation: float     = 1.5,
        slope_range: Tuple[float, float] = (-12.0, 12.0),
        step_height: float              = 0.40,
        disturbance_force_range: Tuple[float, float] = (50.0, 200.0),
        disturbance_frequency: float    = 0.03,
        randomise_terrain: bool         = True,
        hardcore: bool                  = False,
        seed: Optional[int]             = None,
    ):
        # Pass render_mode and hardcore to the parent BipedalWalker
        super().__init__(render_mode=render_mode, hardcore=hardcore)

        # --- Feature flags ---
        self.enable_terrain_generation = enable_terrain_generation
        self.enable_disturbance        = enable_disturbance

        # --- Terrain ---
        self._difficulty = float(np.clip(terrain_difficulty_level, 0.0, 1.0))
        self._terrain_gen = TerrainGenerator(
            chunk_length         = chunk_length,
            max_height_variation = max_height_variation,
            slope_range          = slope_range,
            step_height          = step_height,
            difficulty_level     = self._difficulty,
            randomise_after_warmup = randomise_terrain,
            base_y               = TERRAIN_HEIGHT,
            seed                 = seed,
        )

        # --- Disturbances ---
        self._disturbance_force_range = disturbance_force_range
        self._disturbance_frequency   = disturbance_frequency
        self._dist_rng                = random.Random(seed)

        # Track current active terrain bodies so we can clean them up
        self._custom_terrain_bodies: list = []

    # ------------------------------------------------------------------
    # Curriculum helpers (call from training loop)
    # ------------------------------------------------------------------

    def set_difficulty(self, level: float):
        """Update terrain difficulty on-the-fly (e.g. after each iteration)."""
        self._difficulty = float(np.clip(level, 0.0, 1.0))
        self._terrain_gen.set_difficulty(self._difficulty)

    @property
    def difficulty(self) -> float:
        return self._difficulty

    # ------------------------------------------------------------------
    # Gymnasium reset – hook terrain generation here
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        # Let the parent class reset physics, spawn the hull, etc.
        obs, info = super().reset(seed=seed, options=options)

        if self.enable_terrain_generation and _BOX2D_AVAILABLE:
            # ── 1. Destroy the parent's default flat terrain bodies ──────────
            for body in self.terrain:
                try:
                    self.world.DestroyBody(body)
                except Exception:
                    pass
            self.terrain.clear()
            self.terrain_poly = []

            # ── 2. Destroy any leftover custom bodies from previous episode ──
            self._destroy_custom_terrain()

            # ── 3. Generate fresh custom terrain and build physics bodies ────
            self._terrain_gen.reset(difficulty_level=self._difficulty)
            self._terrain_gen.generate_initial_chunks(self.INITIAL_CHUNKS)
            self._build_terrain_bodies()
            self._update_terrain_poly()

        return obs, info

    # ------------------------------------------------------------------
    # Rendering — vertical camera tracking with solid terrain fill
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode is None:
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            return

        # Initialize screen/surf safely
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            from gymnasium.envs.box2d.bipedal_walker import VIEWPORT_W, VIEWPORT_H, SCALE
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        from gymnasium.envs.box2d.bipedal_walker import VIEWPORT_W, VIEWPORT_H, SCALE, TERRAIN_HEIGHT, TERRAIN_STEP
        from Box2D.b2 import circleShape
        import numpy as np

        # ------------------------------------------------------------------
        # Vertical camera offset
        # ------------------------------------------------------------------
        # The pygame surface is drawn with y=0 at the TOP, then flipped at
        # the end.  After flip: pre-flip y=0  → visible screen BOTTOM,
        #                        pre-flip y=VIEWPORT_H → visible screen TOP.
        #
        # offset_y_world shifts every world-y upward in pixel space (pre-flip)
        # so that the hull stays centred vertically even when terrain drops.
        # ------------------------------------------------------------------
        offset_y_world = 0.0
        if self.hull is not None:
            hull_drop = TERRAIN_HEIGHT - self.hull.position.y
            offset_y_world = float(np.clip(hull_drop,
                                           -VIEWPORT_H / SCALE / 2,
                                            VIEWPORT_H / SCALE * 2))

        # ------------------------------------------------------------------
        # The "floor" of each terrain polygon must reach the visible screen
        # BOTTOM after the vertical flip.  Pre-flip screen bottom = y=0.
        # In world units that means: (floor_world_y + offset_y_world)*SCALE = 0
        #   → floor_world_y = -offset_y_world
        # We subtract a small extra margin (2 world units) so the fill goes
        # slightly off-screen and leaves no gap regardless of rounding.
        # ------------------------------------------------------------------
        FLOOR_WORLD_Y = -offset_y_world - 2.0

        # Dynamically sized surface to handle horizontal scrolling
        self.surf = pygame.Surface(
            (VIEWPORT_W + max(0.0, self.scroll) * SCALE, VIEWPORT_H)
        )

        pygame.transform.scale(self.surf, (SCALE, SCALE))

        # Sky background
        pygame.draw.polygon(
            self.surf,
            color=(215, 215, 255),
            points=[
                (self.scroll * SCALE, 0),
                (self.scroll * SCALE + VIEWPORT_W, 0),
                (self.scroll * SCALE + VIEWPORT_W, VIEWPORT_H),
                (self.scroll * SCALE, VIEWPORT_H),
            ],
        )

        # Clouds
        if getattr(self, "cloud_poly", None):
            for poly, x1, x2 in self.cloud_poly:
                if x2 < self.scroll / 2 or x1 > self.scroll / 2 + VIEWPORT_W / SCALE:
                    continue
                pts = [(p[0] * SCALE + self.scroll * SCALE / 2,
                        (p[1] + offset_y_world) * SCALE) for p in poly]
                pygame.draw.polygon(self.surf, color=(255, 255, 255), points=pts)
                gfxdraw.aapolygon(self.surf, pts, (255, 255, 255))

        # ------------------------------------------------------------------
        # Terrain — draw each surface segment as a filled quad that extends
        # all the way to FLOOR_WORLD_Y so there is never a gap below it.
        #
        # terrain_poly entries have form:
        #   ((x1, y1), (x2, y2), ...)  — first two points are the TOP surface.
        #   We intentionally skip vertical riser segments (where x1 == x2)
        #   because those are stair edges, not horizontal treads, and drawing
        #   a zero-width quad for them causes the "thin-line" artefact.
        # ------------------------------------------------------------------
        for poly, color in self.terrain_poly:
            x1, y1 = poly[0]
            x2, y2 = poly[1]

            # Skip vertical riser segments (stair risers have the same x)
            if abs(x2 - x1) < 1e-4:
                continue

            # Horizontal culling
            if x2 < self.scroll or x1 > self.scroll + VIEWPORT_W / SCALE:
                continue

            # Build a filled quad: terrain surface on top → floor on bottom.
            # For a stair TREAD the top may be horizontal (y1 == y2) or
            # slightly sloped (slope / uneven).  Either way this draws solid.
            quad = [
                (x1 * SCALE,    (y1 + offset_y_world) * SCALE),
                (x2 * SCALE,    (y2 + offset_y_world) * SCALE),
                (x2 * SCALE,    (FLOOR_WORLD_Y + offset_y_world) * SCALE),
                (x1 * SCALE,    (FLOOR_WORLD_Y + offset_y_world) * SCALE),
            ]
            pygame.draw.polygon(self.surf, color=color, points=quad)
            gfxdraw.aapolygon(self.surf, quad, color)

        # Erroneous fill_bottom block removed; terrain quads already span down to exactly FLOOR_WORLD_Y.

        # Lidar
        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        if hasattr(self, "lidar") and i < 2 * len(self.lidar):
            single_lidar = self.lidar[i] if i < len(self.lidar) else self.lidar[len(self.lidar) - i - 1]
            if hasattr(single_lidar, "p1") and hasattr(single_lidar, "p2"):
                pygame.draw.line(
                    self.surf,
                    color=(255, 0, 0),
                    start_pos=(single_lidar.p1[0] * SCALE, (single_lidar.p1[1] + offset_y_world) * SCALE),
                    end_pos=(single_lidar.p2[0] * SCALE, (single_lidar.p2[1] + offset_y_world) * SCALE),
                    width=1,
                )

        # Bodies
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pos = trans * f.shape.pos
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=(pos[0] * SCALE, (pos[1] + offset_y_world) * SCALE),
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=(pos[0] * SCALE, (pos[1] + offset_y_world) * SCALE),
                        radius=f.shape.radius * SCALE,
                    )
                else:
                    path = []
                    for v in f.shape.vertices:
                        pt = trans * v
                        path.append((pt[0] * SCALE, (pt[1] + offset_y_world) * SCALE))
                    if len(path) > 2:
                        pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                        gfxdraw.aapolygon(self.surf, path, obj.color1)
                        path.append(path[0])
                        pygame.draw.polygon(self.surf, color=obj.color2, points=path, width=1)
                        gfxdraw.aapolygon(self.surf, path, obj.color2)
                    else:
                        pygame.draw.aaline(
                            self.surf, start_pos=path[0], end_pos=path[1], color=obj.color1
                        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (-self.scroll * SCALE, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            # standard Gymnasium horizontal scrolling slice
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )[:, -int(VIEWPORT_W):]

    # ------------------------------------------------------------------
    # Gymnasium step – rolling terrain + disturbances
    # ------------------------------------------------------------------

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # ---- rolling terrain window ----
        if self.enable_terrain_generation and _BOX2D_AVAILABLE and self.hull is not None:
            # hull.position is already in Box2D world-units (metres) — no SCALE needed
            agent_x = self.hull.position.x
            new_chunks = self._terrain_gen.advance_if_needed(agent_x, self.LOOKAHEAD)
            if new_chunks:
                self._build_terrain_bodies(new_chunks)
            pruned = self._terrain_gen.prune_behind(agent_x, self.TAIL_KEEP)
            if pruned:
                self._prune_terrain_bodies(pruned)

            if new_chunks or pruned:
                self._update_terrain_poly()

        # ---- external disturbance ----
        if self.enable_disturbance and _BOX2D_AVAILABLE and self.hull is not None:
            if self._dist_rng.random() < self._disturbance_frequency:
                self._apply_disturbance()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Box2D terrain body management
    # ------------------------------------------------------------------

    def _build_terrain_bodies(self, chunks=None):
        """
        Create Box2D static edge-chain bodies for the given chunks.
        Box2D uses raw world-unit coordinates — do NOT multiply by SCALE.
        If chunks is None, rebuild all chunks currently in the generator.
        """
        if not _BOX2D_AVAILABLE:
            return

        target_chunks = chunks if chunks is not None else self._terrain_gen.active_chunks

        for chunk in target_chunks:
            pts = chunk.points
            if len(pts) < 2:
                continue

            # Raw world coordinates — Box2D works in metres, SCALE is only for rendering
            body = self.world.CreateStaticBody()
            body.userData = {"chunk_id": id(chunk)}

            for i in range(len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i + 1]
                body.CreateEdgeFixture(
                    vertices=[(x1, y1), (x2, y2)],
                    friction=0.8,
                    density=0,
                )

            self._custom_terrain_bodies.append((id(chunk), body))

    def _prune_terrain_bodies(self, pruned_chunks):
        """Destroy Box2D bodies for chunks that scrolled behind the agent."""
        if not _BOX2D_AVAILABLE:
            return

        pruned_ids = {id(c) for c in pruned_chunks}
        remaining  = []
        for chunk_id, body in self._custom_terrain_bodies:
            if chunk_id in pruned_ids:
                self.world.DestroyBody(body)
            else:
                remaining.append((chunk_id, body))
        self._custom_terrain_bodies = remaining

    def _destroy_custom_terrain(self):
        """Destroy all custom terrain bodies (called at reset)."""
        if not _BOX2D_AVAILABLE:
            return
        for _, body in self._custom_terrain_bodies:
            try:
                self.world.DestroyBody(body)
            except Exception:
                pass
        self._custom_terrain_bodies.clear()
        self.terrain_poly = []

    def _update_terrain_poly(self):
        """
        Rebuild the visual polygon list for the pygame renderer.

        Key fix: stair chunks produce VERTICAL riser segments (same x, different y)
        as well as HORIZONTAL tread segments.  We store ALL consecutive pairs as
        separate entries.  The render() loop then skips any entry where x1 ≈ x2
        (the riser) so only the filled tread quads are drawn — eliminating the
        thin-line artefact that appeared during stair descent.
        """
        self.terrain_poly = []
        for chunk in self._terrain_gen.active_chunks:
            pts = chunk.points
            for i in range(len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i + 1]
                # Store as (top-left, top-right, bottom-right, bottom-left).
                # The render loop overrides the Y floor so these bottom values
                # are placeholders only — kept for structural completeness.
                poly = [
                    (x1, y1),
                    (x2, y2),
                    (x2, 0.0),  # placeholder floor — overridden in render()
                    (x1, 0.0),  # placeholder floor — overridden in render()
                ]
                # Alternate slightly to match original BipedalWalker grass texture
                color = (102, 153, 76) if i % 2 == 0 else (102, 140, 76)
                self.terrain_poly.append((poly, color))

    # ------------------------------------------------------------------
    # External disturbance
    # ------------------------------------------------------------------

    def _apply_disturbance(self):
        """
        Apply a random lateral impulse to the hull (torso) body.
        Direction is random (left / right), magnitude is drawn uniformly
        from disturbance_force_range.
        """
        if not _BOX2D_AVAILABLE or self.hull is None:
            return
        lo, hi   = self._disturbance_force_range
        force_x  = self._dist_rng.uniform(lo, hi) * self._dist_rng.choice([-1, 1])
        force_y  = 0.0          # purely lateral; vertical would be unrealistic
        self.hull.ApplyForceToCenter(
            force=(force_x, force_y),
            wake=True,
        )
