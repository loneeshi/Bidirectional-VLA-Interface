"""Perception for base clearance: map fusion + queries (embodiment- and task-agnostic) and simulated sensors.

Clean copies of flowsde/perception.py (2.5D height map, sensor levels L0-L5, depth + lidar fusion "La+Lb") and of the
3D height-banded map logic of flowsde/distill/v2/banded.py (copied, not imported).

  Grid / clearance_map   world-frame grid and Euclidean distance transform (flowsde/nav_scenes.py, copied)
  HeightMap              2.5D map: max surface height per cell; unknown = 0 = free (OPTIMISTIC); obstacles: H > thr.
                         fuse_points(pts) (depth: per-cell max of the frame, last observation wins), fuse_scan(...)
                         (planar lidar: hits become columns of assumed height, ray-traced free space clears cells).
  RaySensorSim           MuJoCo ray-cast depth camera / planar lidar mounted on the base (simulation only; on hardware
                         feed point clouds / scans to the maps directly).
  Perception             one sensor + map (+ fusion delay = map staleness)   [level presets LEVELS]
  Fused                  two sensors on one grid, H = max (e.g. "L2+L4": depth camera + 2D lidar)
  BandedMap              3D occupancy per height band (5 cm): overhangs (table tops, counter lips) no longer become
                         floor-to-top columns, so the base may partly drive under them.
Queries: base_clearance(xy) (EDT of the obstacle cells), sphere_query(P) (distance from robot collision spheres to the
nearest perceived obstacle column / voxel and the direction away from it).

Evidence (TUNING_GUIDE §6): COVERAGE matters more than resolution. On NavTrack, 10 cm cells, 48 x 36 rays or a 0.5 s
stale map keep 0.98 collision-free completion; dropping thin structures (< 3 cm poles) gives 0.48; a forward-only depth
camera misses fixtures beside / behind the robot in kitchens (cold-start collisions, 0.53); a 0.25 m planar lidar
misses low boxes (0.20-0.25 m). Depth + 360 deg lidar fused: 0.97 (NavTrack), 0.72-0.75 with 0 collisions (kitchens).
"""
import collections

import mujoco
import numpy as np

from flowsde_ctrl.math3d import Rz


# ---------------------------------------------------------------------------------------------------- grid
class Grid:
    def __init__(self, x0, y0, nx, ny, cell):
        self.x0, self.y0, self.nx, self.ny, self.cell = x0, y0, nx, ny, cell

    def idx(self, x, y):
        return (np.floor((np.asarray(x) - self.x0) / self.cell).astype(int), np.floor((np.asarray(y) - self.y0) / self.cell).astype(int))

    def centers(self):
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.cell; ys = self.y0 + (np.arange(self.ny) + 0.5) * self.cell
        return xs, ys


def clearance_map(H, cell, thresh=0.02):
    from scipy.ndimage import distance_transform_edt
    occ = H > thresh
    if not occ.any():
        return np.full(H.shape, 99.0, np.float32)
    return (distance_transform_edt(~occ) * cell).astype(np.float32)


def regrid(grid0, cell):
    """grid with cell size `cell` covering grid0 (x0, y0, nx, ny, cell0)."""
    x0, y0, nx, ny, cell0 = grid0
    f = cell / cell0
    nx2, ny2 = (nx, ny) if abs(f - 1) < 1e-9 else (int(np.ceil(nx / f - 1e-6)), int(np.ceil(ny / f - 1e-6)))
    return Grid(x0, y0, nx2, ny2, cell)


# ---------------------------------------------------------------------------------------------------- sensors (sim)
LEVELS = {
    "L0": dict(kind="exact", cell=0.02, noise=0.0),
    "L1": dict(kind="depth", cell=0.02, noise=0.005, thin=True, res=(128, 96), stale=0.0),
    "L2": dict(kind="depth", cell=0.05, noise=0.02, thin=False, res=(80, 60), stale=0.0),
    "L3": dict(kind="depth", cell=0.10, noise=0.05, thin=False, res=(64, 48), stale=0.5),
    "L4": dict(kind="lidar", cell=0.05, noise=0.01, z=0.25, n=720, rmax=6.0, assume_h=0.9, stale=0.0),
    "L5": dict(kind="none", cell=0.05, noise=0.0),
}
CAM = dict(pos=(0.05, 0.0, 1.45), pitch=np.radians(40.0), hfov=np.radians(87.0), vfov=np.radians(58.0), rmin=0.2, rmax=4.0)
RATE = 0.1


class RaySensorSim:
    """MuJoCo ray-cast sensor on the base. world: object with base_state() -> (cx, cy, th), base_origin(cx, cy, th)
    -> xy, base_z, and the simulator m / d. groups: geom groups visible to the sensor (uint8 mask of 6)."""

    def __init__(self, m, d, world, kind, c, groups, rng, cam=None):
        self.m, self.d, self.w, self.kind, self.c, self.groups, self.rng = m, d, world, kind, c, groups, rng
        self.cam = cam or CAM
        self.n_rays = 0
        if kind == "depth":
            W, Hh = c["res"]; cam = self.cam
            az = np.linspace(-cam["hfov"] / 2, cam["hfov"] / 2, W); el = np.linspace(-cam["vfov"] / 2, cam["vfov"] / 2, Hh)
            A, E = np.meshgrid(az, el, indexing="ij")
            dd = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
            p = cam["pitch"]; Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
            self.dirs_b = dd @ Ry.T
        elif kind == "lidar":
            a = np.linspace(-np.pi, np.pi, c["n"], endpoint=False)
            self.dirs_b = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1)

    def _cast(self, origin, dirs, rmax):
        n = len(dirs)
        gid = np.zeros(n, np.int32); dist = np.zeros(n)
        mujoco.mj_multiRay(self.m, self.d, origin.astype(np.float64), np.ascontiguousarray(dirs, np.float64).ravel(),
                           self.groups, 1, -1, gid, dist, n, rmax)
        self.n_rays += n
        return dist, gid

    def frame(self):
        w = self.w
        cx, cy, th = w.base_state()
        o = w.base_origin(cx, cy, th)
        R = Rz(th)
        if self.kind == "depth":
            cam = self.cam
            org = np.r_[o, w.base_z] + R @ np.array(cam["pos"])
            dirs = self.dirs_b @ R.T
            dist, gid = self._cast(org, dirs, cam["rmax"])
            hit = dist >= 0
            dz = dirs[:, 2]
            tf = np.where(dz < -1e-6, -org[2] / np.minimum(dz, -1e-6), np.inf)    # analytic floor plane z = 0
            dist = np.where(hit, dist, tf)
            ok = (dist >= cam["rmin"]) & (dist <= cam["rmax"])
            dist = dist + self.rng.normal(0, self.c["noise"], len(dist))
            pts = org[None] + dist[:, None] * dirs
            return ("pts", pts[ok])
        org = np.r_[o, self.c["z"]]
        dirs = self.dirs_b @ R.T
        dist, gid = self._cast(org, dirs, self.c["rmax"])
        hit = dist >= 0
        dist_n = np.where(hit, dist + self.rng.normal(0, self.c["noise"], len(dist)), self.c["rmax"])
        return ("scan", org, dirs, dist_n, hit)


# ---------------------------------------------------------------------------------------------------- 2.5D height map
class HeightMap:
    def __init__(self, grid, thr, assume_h=0.9, H=None):
        self.grid, self.thr, self.assume_h = grid, thr, assume_h
        self.H = np.zeros((grid.nx, grid.ny), np.float32) if H is None else H
        self.seen = np.zeros(self.H.shape, bool) if H is None else np.ones(self.H.shape, bool)
        self.version = 0; self._clr = None; self._clr_ver = -1

    def fuse(self, fr):
        g = self.grid
        if fr[0] == "pts":
            pts = fr[1]
            i, j = g.idx(pts[:, 0], pts[:, 1])
            ok = (i >= 0) & (j >= 0) & (i < g.nx) & (j < g.ny)
            i, j, z = i[ok], j[ok], np.maximum(pts[ok, 2], 0.0).astype(np.float32)
            if len(i) == 0:
                return
            lin = i * g.ny + j
            frame = np.full(g.nx * g.ny, -1.0, np.float32)
            np.maximum.at(frame, lin, z)
            upd = frame >= 0
            Hf = self.H.reshape(-1); Hf[upd] = frame[upd]
            self.seen.reshape(-1)[upd] = True
        else:
            _, org, dirs, dist, hit = fr
            step = 0.5 * g.cell
            L = np.maximum(dist - g.cell, 0.0)
            ns = (L / step).astype(int)
            mx = int(ns.max()) if len(ns) else 0
            if mx > 0:
                s = np.arange(mx)[None, :] * step
                valid = np.arange(mx)[None, :] < ns[:, None]
                P = org[None, None, :2] + s[..., None] * dirs[:, None, :2]
                P = P[valid]
                i, j = g.idx(P[:, 0], P[:, 1])
                ok = (i >= 0) & (j >= 0) & (i < g.nx) & (j < g.ny)
                self.H[i[ok], j[ok]] = 0.0; self.seen[i[ok], j[ok]] = True
            hp = org[None, :2] + dist[hit, None] * dirs[hit, :2]
            i, j = g.idx(hp[:, 0], hp[:, 1])
            ok = (i >= 0) & (j >= 0) & (i < g.nx) & (j < g.ny)
            self.H[i[ok], j[ok]] = self.assume_h; self.seen[i[ok], j[ok]] = True
        self.version += 1


def _clearance(H, grid, thr, cache_owner):
    if cache_owner._clr_ver != cache_owner.version or cache_owner._clr is None:
        cache_owner._clr = clearance_map(H, grid.cell, thr); cache_owner._clr_ver = cache_owner.version
    return cache_owner._clr


def base_clearance_q(clr, grid, xy):
    xy = np.atleast_2d(xy)
    i, j = grid.idx(xy[:, 0], xy[:, 1])
    i = np.clip(i, 0, grid.nx - 1); j = np.clip(j, 0, grid.ny - 1)
    return clr[i, j]


def sphere_query_H(H, grid, thr, P, rinf=0.35):
    """distance from world points P (n, 3) to the nearest obstacle COLUMN (cells with H > thr occupy [0, H]) and the
    unit direction away from it; (inf, 0) if none within rinf."""
    g = grid; c = g.cell
    out_d = np.full(len(P), np.inf); out_n = np.zeros((len(P), 3))
    r = int(np.ceil(rinf / c)) + 1
    for k, p in enumerate(P):
        i0, j0 = g.idx(p[0], p[1]); i0, j0 = int(i0), int(j0)
        a0, a1, b0, b1 = max(i0 - r, 0), min(i0 + r + 1, g.nx), max(j0 - r, 0), min(j0 + r + 1, g.ny)
        if a1 <= a0 or b1 <= b0:
            continue
        blk = H[a0:a1, b0:b1]
        occ = blk > thr
        if not occ.any():
            continue
        ii, jj = np.nonzero(occ)
        cxs = g.x0 + (ii + a0 + 0.5) * c; cys = g.y0 + (jj + b0 + 0.5) * c; hz = blk[ii, jj]
        qx = np.clip(p[0], cxs - c / 2, cxs + c / 2); qy = np.clip(p[1], cys - c / 2, cys + c / 2); qz = np.minimum(p[2], hz)
        dd = np.sqrt((p[0] - qx) ** 2 + (p[1] - qy) ** 2 + (p[2] - qz) ** 2)
        m = int(np.argmin(dd))
        out_d[k] = dd[m]
        v = p - np.array([qx[m], qy[m], qz[m]])
        nv = np.linalg.norm(v)
        out_n[k] = v / nv if nv > 1e-9 else np.array([0, 0, 1.0])
    return out_d, out_n


# ---------------------------------------------------------------------------------------------------- one sensor + map
class Perception:
    """level: a LEVELS key. world: see RaySensorSim (+ grid0 = (x0, y0, nx, ny, cell0) of the task region and, for L0,
    exact_H(grid) -> privileged height map). groups: dict(obs=group id, thin=group id) of the sim scene."""

    def __init__(self, world, level="L0", seed=0, groups=(3, 4), cam=None, **over):
        self.w = world
        self.level = level
        self.c = dict(LEVELS[level]); self.c.update(over)
        self.rng = np.random.default_rng([int(seed), 31337])
        kind = self.c["kind"]
        thr = 0.03 + 3 * self.c["noise"] if kind in ("depth",) else 0.02
        self.grid = regrid(world.grid0, self.c["cell"])
        H = world.exact_H(self.grid) if kind == "exact" else None
        self.map = HeightMap(self.grid, thr, self.c.get("assume_h", 0.9), H)
        self.thr = thr
        self.queue = collections.deque(); self.t_last = -1e9
        grp = np.zeros(6, np.uint8); grp[groups[0]] = 1
        if self.c.get("thin", True):
            grp[groups[1]] = 1
        self.sensor = RaySensorSim(world.m, world.d, world, kind, self.c, grp, self.rng, cam) if kind in ("depth", "lidar") else None

    @property
    def H(self):
        return self.map.H

    @property
    def version(self):
        return self.map.version

    @property
    def n_rays(self):
        return self.sensor.n_rays if self.sensor else 0

    def update(self, t):
        """call every control tick with time t (s): senses at RATE, fuses after the staleness delay."""
        if self.sensor is None:
            return
        if t - self.t_last >= RATE - 1e-9:
            self.t_last = t
            self.queue.append((t, self.sensor.frame()))
        st = self.c.get("stale", 0.0)
        while self.queue and t - self.queue[0][0] >= st - 1e-9:
            self.map.fuse(self.queue.popleft()[1])

    def base_clearance(self, xy):
        return base_clearance_q(_clearance(self.map.H, self.grid, self.thr, self.map), self.grid, xy)

    def sphere_query(self, P, rinf=0.35):
        return sphere_query_H(self.map.H, self.grid, self.thr, P, rinf)


class Fused:
    """'La+Lb': two sensors on the first one's grid; H = max of both maps."""

    def __init__(self, world, level, seed=0, **over):
        a, b = level.split("+")
        self.level = level
        self.p = [Perception(world, a, seed, **over), Perception(world, b, seed + 1, cell=LEVELS[a]["cell"])]
        self.grid = self.p[0].grid; self.thr = self.p[0].thr; self.c = self.p[0].c
        self._clr = None; self._clr_ver = None; self._Hc = None

    @property
    def H(self):
        return np.maximum(self.p[0].H, self.p[1].H)

    @property
    def version(self):
        return tuple(q.version for q in self.p)

    @property
    def n_rays(self):
        return sum(q.n_rays for q in self.p)

    def update(self, t):
        for q in self.p:
            q.update(t)

    def _clr_map(self):
        if self._clr_ver != self.version or self._clr is None:
            self._Hc = self.H
            self._clr = clearance_map(self._Hc, self.grid.cell, self.thr); self._clr_ver = self.version
        return self._clr

    def base_clearance(self, xy):
        return base_clearance_q(self._clr_map(), self.grid, xy)

    def sphere_query(self, P, rinf=0.35):
        self._clr_map()
        return sphere_query_H(self._Hc, self.grid, self.thr, P, rinf)


def make(world, level="L0", seed=0, **over):
    if level == "banded":
        return BandedMap(world, seed)
    if "+" in level:
        return Fused(world, level, seed, **over)
    return Perception(world, level, seed, **over)


# ---------------------------------------------------------------------------------------------------- 3D banded map
class BandedMap:
    """Occupancy per height band (dz, 0..zmax) per cell from depth (L2) + planar lidar (L4). Depth: every cell observed
    in a frame has its band column replaced by the bands of its points (points below the ground threshold are ground).
    Lidar: returns mark [0, assume_h], ray-traced free space clears the lidar layer. occupied = depth OR lidar; unknown =
    free. sphere_query gives true 3D clearance to the perceived voxels. (Logic copied from flowsde/distill/v2/banded.py.)
    Base clearance for the path planner uses the 2.5D top-band view (conservative)."""

    def __init__(self, world, seed=0, dz=0.05, zmax=2.0, depth="L2", lidar="L4"):
        self.dep = Perception(world, depth, seed)
        self.lid = Perception(world, lidar, seed + 1, cell=self.dep.c["cell"])
        self.grid = self.dep.grid; self.c = self.dep.c; self.thr = self.dep.thr
        self.dz, self.nb = dz, int(round(zmax / dz))
        g = self.grid
        self.occ_d = np.zeros((g.nx, g.ny, self.nb), bool)
        self.occ_l = np.zeros((g.nx, g.ny, self.nb), bool)
        self.nb_lid = int(np.ceil(self.lid.c["assume_h"] / dz))
        self.version = 0; self._clr = None; self._clr_ver = -1
        self.level = f"{depth}+{lidar}/banded"

    @property
    def n_rays(self):
        return self.dep.n_rays + self.lid.n_rays

    def _fuse_depth(self, pts):
        g = self.grid
        i, j = g.idx(pts[:, 0], pts[:, 1]); i, j = np.asarray(i), np.asarray(j)
        ok = (i >= 0) & (j >= 0) & (i < g.nx) & (j < g.ny)
        i, j, z = i[ok], j[ok], pts[ok, 2]
        if len(i) == 0:
            return
        self.occ_d[i, j, :] = False
        ob = z > self.thr
        k = np.clip((z[ob] / self.dz).astype(int), 0, self.nb - 1)
        self.occ_d[i[ob], j[ob], k] = True

    def _fuse_lidar(self, fr):
        _, org, dirs, dist, hit = fr
        g = self.grid
        step = 0.5 * g.cell
        L = np.maximum(dist - g.cell, 0.0)
        ns = (L / step).astype(int)
        mx = int(ns.max()) if len(ns) else 0
        if mx > 0:
            s = np.arange(mx)[None, :] * step
            valid = np.arange(mx)[None, :] < ns[:, None]
            P = (org[None, None, :2] + s[..., None] * dirs[:, None, :2])[valid]
            i, j = g.idx(P[:, 0], P[:, 1]); i, j = np.asarray(i), np.asarray(j)
            ok = (i >= 0) & (j >= 0) & (i < g.nx) & (j < g.ny)
            self.occ_l[i[ok], j[ok], :] = False
        hp = org[None, :2] + dist[hit, None] * dirs[hit, :2]
        i, j = g.idx(hp[:, 0], hp[:, 1]); i, j = np.asarray(i), np.asarray(j)
        ok = (i >= 0) & (j >= 0) & (i < g.nx) & (j < g.ny)
        self.occ_l[i[ok], j[ok], :self.nb_lid] = True

    def update(self, t):
        for p in (self.dep, self.lid):
            if t - p.t_last >= RATE - 1e-9:
                p.t_last = t
                fr = p.sensor.frame()
                if fr[0] == "pts":
                    self._fuse_depth(fr[1])
                else:
                    self._fuse_lidar(fr)
                self.version += 1

    @property
    def H(self):
        occ = self.occ_d | self.occ_l
        top = np.where(occ.any(2), (self.nb - np.argmax(occ[:, :, ::-1], axis=2)) * self.dz, 0.0)
        return top.astype(np.float32)

    def base_clearance(self, xy):
        if self._clr_ver != self.version or self._clr is None:
            self._clr = clearance_map(self.H, self.grid.cell, self.thr); self._clr_ver = self.version
        return base_clearance_q(self._clr, self.grid, xy)

    def sphere_query(self, P, rinf=0.35):
        g = self.grid; c = g.cell
        out_d = np.full(len(P), np.inf); out_n = np.zeros((len(P), 3))
        r = int(np.ceil(rinf / c)) + 1
        for k, p in enumerate(P):
            i0, j0 = g.idx(p[0], p[1]); i0, j0 = int(i0), int(j0)
            a0, a1, b0, b1 = max(i0 - r, 0), min(i0 + r + 1, g.nx), max(j0 - r, 0), min(j0 + r + 1, g.ny)
            if a1 <= a0 or b1 <= b0:
                continue
            kz0 = max(int((p[2] - rinf) / self.dz), 0); kz1 = min(int((p[2] + rinf) / self.dz) + 1, self.nb)
            if kz1 <= kz0:
                continue
            blk = self.occ_d[a0:a1, b0:b1, kz0:kz1] | self.occ_l[a0:a1, b0:b1, kz0:kz1]
            if not blk.any():
                continue
            ii, jj, kk = np.nonzero(blk)
            cxs = g.x0 + (ii + a0 + 0.5) * c; cys = g.y0 + (jj + b0 + 0.5) * c
            z0 = (kk + kz0) * self.dz; z1 = z0 + self.dz
            qx = np.clip(p[0], cxs - c / 2, cxs + c / 2); qy = np.clip(p[1], cys - c / 2, cys + c / 2); qz = np.clip(p[2], z0, z1)
            dd = np.sqrt((p[0] - qx) ** 2 + (p[1] - qy) ** 2 + (p[2] - qz) ** 2)
            m = int(np.argmin(dd))
            out_d[k] = dd[m]
            v = p - np.array([qx[m], qy[m], qz[m]]); nv = np.linalg.norm(v)
            out_n[k] = v / nv if nv > 1e-9 else np.array([0, 0, 1.0])
        return out_d, out_n
