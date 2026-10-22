"""Sensor-only, finite-horizon clearance checks against calibrated robot envelopes.

This module owns no obstacle history. ``EpisodeMemory`` owns records, provenance,
expiry and checkpointing. The adapter supplies synchronized robot RGB-D/FK and
episode-relative odometry; neither scene meshes nor simulator poses are accepted.
An accepted report means no conflict in the observed swept envelopes, not a
collision-free dynamics guarantee. Missing, self-occluded and expired views are
unknown, never free space. Sampling robot cameras is an adapter responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math

import numpy as np

from .eef_tools import RobotFK, transform, vector


OPTICAL_TO_CAMERA = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
FINGERS = frozenset(('l_gripper_finger_link', 'r_gripper_finger_link'))


def _float32_hold_resolution(initial_pose, poses, bounds):
    """Bound total corner displacement by one robot-local float32 ULP.

    This is an observation admission resolution, not evidence of free space.
    Compare every endpoint to command entry (never accumulate per-step credit).
    Robot-local coordinates prevent a large odometry origin inflating the bound.
    """
    corners = _corners(bounds)
    initial = corners @ initial_pose[:3, :3].T + initial_pose[:3, 3]
    resolution = np.linalg.norm(np.spacing(np.abs(initial).astype(np.float32)).astype(float),axis=1)
    return all(np.all(np.linalg.norm(corners @ p[:3,:3].T+p[:3,3]-initial,axis=1)<=resolution)
               for p in poses)


def _shared_finger_hold_path(states):
    """Recognize only interpolation to the SDK-encoded measured mean.

    This classifies a shared-actuator hold, not arbitrary opening/closing or
    evidence of free space. Spatial containment is checked separately.
    """
    q = np.asarray([s[0][13:15] for s in states], float)
    if len(q) < 2 or np.any(q[0] < 0) or np.any(q[0] > .05):
        return False
    lo, hi = np.float32(-.01), np.float32(.05)
    center, scale = np.float32(.5)*(hi+lo), np.float32(.5)*(hi-lo)
    mean = np.mean(q[0].astype(np.float32), dtype=np.float32)
    target = float(center + scale*np.clip((mean-center)/scale, np.float32(-1), np.float32(1)))
    if not np.array_equal(q[-1], [target, target]):
        return False
    delta = np.array([target, target])-q[0]
    length = float(delta@delta)
    if length == 0:
        return np.array_equal(q, np.broadcast_to(q[0], q.shape))
    progress = (q-q[0])@delta/length
    # Roundoff in interpolation algebra only; not a physical distance allowance.
    eps = 32*np.finfo(float).eps
    return bool(np.all(progress >= -eps) and np.all(progress <= 1+eps)
                and np.all(np.diff(progress) >= -eps)
                and np.allclose(q, q[0]+progress[:, None]*delta, atol=eps, rtol=0))


@dataclass(frozen=True)
class ClearanceConfig:
    voxel_m: float = .025
    finger_voxel_m: float = .005
    precision_reach_m: float = .08
    tile_pixels: int = 32
    max_occupied_per_camera: int = 32768
    margin_m: float = .015
    depth_margin_m: float = .02
    min_depth_m: float = .08
    max_depth_m: float = 5.
    path_translation_m: float = .02
    path_rotation_rad: float = .025
    path_joint_delta: float = .025
    coverage_cell_m: float = .10
    # None uses the calibrated lowest base/wheel envelope as a narrow search
    # band. A plane still needs dense, spatially spread current depth support.
    support_plane_odom_z_m: float | None = None
    support_plane_tolerance_m: float = .003
    tight_near_voxels: bool = False
    local_support_floor: bool = False
    s11_report: bool = False

    def __post_init__(self):
        values = [getattr(self, k) for k in self.__dataclass_fields__
                  if k not in ('support_plane_odom_z_m', 'tight_near_voxels', 'local_support_floor', 's11_report')]
        if type(self.local_support_floor) is not bool:
            raise ValueError('local_support_floor must be boolean')
        if type(self.s11_report) is not bool:
            raise ValueError('s11_report must be boolean')
        if type(self.tight_near_voxels) is not bool:
            raise ValueError('tight_near_voxels must be boolean')
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('positive finite clearance settings required')
        if (not isinstance(self.tile_pixels, int) or self.tile_pixels < 1
                or self.min_depth_m >= self.max_depth_m):
            raise ValueError('invalid depth sampling settings')
        if self.support_plane_odom_z_m is not None and not math.isfinite(self.support_plane_odom_z_m):
            raise ValueError('invalid calibrated support plane')


def odometry_transform(odometry):
    """Accept only the adapter's relative odometry or its local_from_base matrix."""
    if isinstance(odometry, dict):
        allowed = {'dx_m', 'dy_m', 'dyaw_rad'}
        if set(odometry) != allowed:
            raise ValueError('relative odometry requires dx_m, dy_m, dyaw_rad')
        x, y, yaw = (float(odometry[k]) for k in ('dx_m', 'dy_m', 'dyaw_rad'))
        if not np.isfinite([x, y, yaw]).all():
            raise ValueError('nonfinite relative odometry')
        out = np.eye(4)
        out[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        out[:2, 3] = [x, y]
        return out
    return transform(odometry)


def _corners(bounds):
    lo, hi = np.asarray(bounds, float)
    return np.array(list(itertools.product(*zip(lo, hi))))


def _inside(points, pose, bounds, margin=0.):
    local = (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]
    return np.all((local >= bounds[0] - margin) & (local <= bounds[1] + margin), axis=-1)


def _signed_distance(points, pose, bounds):
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    center = bounds.mean(axis=0)
    delta = np.abs(local - center) - (bounds[1] - bounds[0]) / 2
    # Fixed three-dimensional vectors: avoid generic row reductions for every
    # body at every swept pose, preserving the original arithmetic order.
    outside = np.maximum(delta, 0.)
    squared = outside * outside
    length = np.sqrt((squared[:, 0] + squared[:, 1]) + squared[:, 2])
    inside = np.maximum(np.maximum(delta[:, 0], delta[:, 1]), delta[:, 2])
    return length + np.minimum(inside, 0.)


def _low_support_components(points, reliable, expected, tolerance, *, image_boundary=False):
    """Observed low planar patches, not proof of traversability.

    Return separate pixel masks and measured upper heights. Never extrapolate
    a height band over other pixels. This diagnostic detector does not itself
    remove occupied records or authorize motion.
    """
    from scipy.ndimage import label, binary_erosion
    candidates = reliable & (np.abs(points[..., 2]-expected) <= .025)
    remaining = candidates.copy()
    patches = []
    # At most four data-derived modes; no case-specific absolute heights.
    for _ in range(4):
        heights = points[..., 2][remaining]
        if len(heights) < 128:
            break
        bins, counts = np.unique(np.floor(heights/tolerance).astype(np.int64), return_counts=True)
        mode = bins[np.argmax(counts)]
        seed = heights[np.floor(heights/tolerance).astype(np.int64) == mode]
        height = float(np.median(seed))
        band = remaining & (np.abs(points[..., 2]-height) <= tolerance)
        remaining[band] = False
        components, count = label(band)
        for index in range(1, count+1):
            mask = components == index
            cloud = points[mask]
            if len(cloud) < 128:
                continue
            # Principal extents reject a long narrow rail even when diagonal.
            xy = cloud[:, :2]-cloud[:, :2].mean(axis=0)
            _, axes = np.linalg.eigh(xy.T@xy/len(xy))
            if np.min(np.ptp(xy@axes, axis=0)) < .30:
                continue
            design = np.column_stack((xy, np.ones(len(xy))))
            plane = np.linalg.lstsq(design, cloud[:, 2], rcond=None)[0]
            residual = cloud[:, 2]-design@plane
            if np.linalg.norm(plane[:2]) > .01 or np.max(np.abs(residual)) > tolerance:
                continue
            # Keep silhouette, mixed-depth and unreliable neighbours occupied.
            # Image boundaries contain no extra pixels to classify. Allowing
            # one-sided support here never creates evidence outside the image;
            # unreliable/self-occluded neighbours inside it remain excluded.
            interior = binary_erosion(mask, structure=np.ones((3, 3)), border_value=int(image_boundary))
            if interior.sum() < 128:
                continue
            patches.append((interior, float(np.max(points[..., 2][interior])+tolerance)))
    return patches


def _local_ground_mask(points, reliable, expected, tolerance):
    """Select the lowest broad measured support patch, not every low surface.

    A higher rug/step/rail cannot become ground merely by dominating the image.
    Separate components retain their own evidence; no mask is extrapolated
    across occlusion, image borders or unobserved space.
    """
    from scipy.ndimage import label
    patches = []
    mask = np.zeros(reliable.shape, bool)
    h, w = reliable.shape
    # Overlapping image windows keep a distant surface from determining the
    # near-ground estimate. Fit only observed, robot-filtered low pixels.
    for window in (128, 256):
        stride = window//2
        ys = sorted(set(range(0, max(1, h-window+1), stride)) | {max(0, h-window)})
        xs = sorted(set(range(0, max(1, w-window+1), stride)) | {max(0, w-window)})
        for y, x in itertools.product(ys, xs):
            cloud = points[y:y+window, x:x+window]
            valid = reliable[y:y+window, x:x+window] & (np.abs(cloud[..., 2]-expected) <= .025)
            if valid.sum() < 128:
                continue
            heights = cloud[..., 2][valid]
            bins, counts = np.unique(np.floor(heights/tolerance).astype(np.int64), return_counts=True)
            # Seed several measured modes, then refine in plane coordinates;
            # a fixed height bin must not permanently split a sloped patch.
            for mode in bins[np.argsort(counts)[-4:]]:
                seed = float(np.median(heights[np.floor(heights/tolerance) == mode]))
                selected = valid & (np.abs(cloud[..., 2]-seed) <= tolerance)
                for _ in range(3):
                    values = cloud[selected]
                    if len(values) < 128:
                        break
                    center = values[:, :2].mean(axis=0)
                    design = np.column_stack((values[:, :2]-center, np.ones(len(values))))
                    plane = np.linalg.lstsq(design, values[:, 2], rcond=None)[0]
                    if np.linalg.norm(plane[:2]) > .01:
                        break
                    residual = cloud[..., 2] - ((cloud[..., :2]-center)@plane[:2]+plane[2])
                    refined = valid & (np.abs(residual) <= tolerance)
                    if np.array_equal(refined, selected):
                        break
                    selected = refined
                else:
                    values = cloud[selected]
                if len(values) < 128 or np.linalg.norm(plane[:2]) > .01:
                    continue
                components, count = label(selected)
                for index in range(1, count+1):
                    region = components == index
                    values = cloud[region]
                    if len(values) < 128:
                        continue
                    xy = values[:, :2]-values[:, :2].mean(axis=0)
                    _, axes = np.linalg.eigh(xy.T@xy/len(xy))
                    if np.min(np.ptp(xy@axes, axis=0)) < .30:
                        continue
                    # Classify each measured inlier itself. Eroding the ground
                    # mask would turn its ground-side boundary into a fictitious
                    # obstacle. Raised furniture pixels remain outliers, and
                    # unreliable pixels cannot enter selected at all. Free-space
                    # tiles independently require all pixels to be reliable.
                    if region.sum() >= 128:
                        height = float(np.median(cloud[..., 2][region]))
                        # Both the calibrated support estimate and the measured
                        # patch have a tolerance. A broad raised platform alone
                        # must not redefine the robot's support height.
                        if abs(height-expected) <= 2*tolerance:
                            patches.append((y, x, window, region, height))
    if not patches:
        return mask, None
    lowest = min(height for _, _, _, _, height in patches)
    for y, x, window, interior, height in patches:
        if height <= lowest + tolerance:
            mask[y:y+window, x:x+window] |= interior
    return mask, float(np.max(points[..., 2][mask]))


def _measured_voxel_envelopes(points, size, uncertainty):
    """Enclose every measured point plus explicit sensor uncertainty.

    Preserve one record per voxel without treating its arbitrary grid center
    as a measured obstacle location. No point or furniture class is removed.
    """
    keys, inverse = np.unique(np.floor(points/size).astype(np.int64), axis=0, return_inverse=True)
    low = np.full((len(keys), 3), np.inf)
    high = np.full((len(keys), 3), -np.inf)
    np.minimum.at(low, inverse, points)
    np.maximum.at(high, inverse, points)
    centers = (low+high)/2
    radii = np.zeros(len(keys))
    np.maximum.at(radii, inverse, np.linalg.norm(points-centers[inverse], axis=1))
    return centers, radii+uncertainty


def _clip_convex(points, normal, offset):
    """Vertices of a convex polyhedron intersected with n.x <= offset.

    All vertex-pair crossings include every real edge crossing. Their convex
    hull therefore represents the clipped body; diagonal crossings cannot add
    space outside it. No sampling claim is used to fill missing coverage.
    """
    distances = points @ normal-offset
    # Classification and intersection must use the same plane. A positive
    # distance admitted by an epsilon can put both ends on the outside while
    # the interpolation below extrapolates beyond the original polyhedron.
    inside = distances <= 0.
    if inside.all():
        return points
    if not inside.any():
        return np.empty((0, 3))
    a, b = points[inside], points[~inside]
    da, db = distances[inside], distances[~inside]
    t = da[:, None]/(da[:, None]-db[None, :])
    crossings = a[:, None, :]+t[..., None]*(b[None, :, :]-a[:, None, :])
    vertices = np.unique(np.round(np.concatenate([a, crossings.reshape(-1, 3)]), 12), axis=0)
    if len(vertices) <= 4:
        return vertices
    # Removing interior vertices only bounds clipping cost, not sensor data.
    from scipy.spatial import ConvexHull, QhullError
    centered = vertices-vertices[0]
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular > 1e-10))
    if rank <= 1:
        if rank == 0:
            return vertices[:1]
        coordinate = centered@axes[0]
        return vertices[[coordinate.argmin(), coordinate.argmax()]]
    try:
        hull = ConvexHull(centered@axes[:rank].T)
    except QhullError:
        return None  # Numerical uncertainty is not free coverage.
    return vertices[hull.vertices]


def _subtract_box(polyhedron, pose, bounds, margin=0.):
    """Partition a convex body's portion outside one robot-entry OBB."""
    local = (polyhedron-pose[:3, 3])@pose[:3, :3]
    lo, hi = bounds[0]-margin, bounds[1]+margin
    if np.any(local.max(axis=0) < lo-1e-10) or np.any(local.min(axis=0) > hi+1e-10):
        return [polyhedron]
    if np.all(local >= lo-1e-10) and np.all(local <= hi+1e-10):
        return []
    remaining, outside = polyhedron, []
    for axis in range(3):
        for sign, limit in ((1., hi[axis]), (-1., -lo[axis])):
            normal = sign*pose[:3, axis]
            offset = limit+normal@pose[:3, 3]
            extends_outside = np.max(remaining@normal-offset) > 1e-10
            part = _clip_convex(remaining, -normal, -offset)
            remaining = _clip_convex(remaining, normal, offset)
            if part is None or remaining is None:
                return None
            if extends_outside and len(part):
                outside.append(part)
            if not len(remaining):
                return outside
    return outside


def _voxel_keys(points, size):
    """Deduplicate packed integer keys, without lexicographically sorting XYZ."""
    if not len(points):
        return np.empty((0, 3), dtype=np.int64)
    keys = np.floor(points/size).astype(np.int64)
    low = keys.min(axis=0)
    shape = keys.max(axis=0)-low+1
    if int(shape[0])*int(shape[1])*int(shape[2]) >= 2**62:
        raise ValueError('robot sensor voxel coordinate range')
    relative = keys-low
    packed = (relative[:, 0]*shape[1]+relative[:, 1])*shape[2]+relative[:, 2]
    _, indices = np.unique(packed, return_index=True)
    return keys[indices]


def target_surface_component(points, valid, pixel, center, rotation, half_extents, *, tolerance_m=.0025):
    """A seed-connected observed planar surface, never an envelope-wide mask.

    Fit locally, grow on the complete image, THEN validate the whole component
    against the candidate estimate. Clipping first would silently cut a nearby
    cabinet/table into a purported target surface. Nonplanar/ambiguous views
    return no ownership and retain all obstacles.
    """
    from scipy.ndimage import label
    points=np.asarray(points,float);valid=np.asarray(valid)
    if (points.ndim!=3 or points.shape[2]!=3 or valid.shape!=points.shape[:2]
            or valid.dtype!=bool or not 0<tolerance_m<=.003):
        raise ValueError('Invalid target surface sensor input')
    h,w=valid.shape;u,v=pixel
    radius=8
    if type(u) is not int or type(v) is not int or not (radius<=u<w-radius and radius<=v<h-radius):
        raise ValueError('Interior source pixel required')
    empty=np.zeros((h,w),bool)
    reliable=valid & np.isfinite(points).all(axis=2)
    if not reliable[v,u]:return empty
    patch=points[v-radius:v+radius+1,u-radius:u+radius+1][reliable[v-radius:v+radius+1,u-radius:u+radius+1]]
    # Millimetre-quantized near-range depth makes a five-pixel plane fit
    # unstable. Use a physical surface patch, and reject a mostly missing one.
    if len(patch)<.8*(2*radius+1)**2:return empty
    centroid=patch.mean(axis=0);_,s,axes=np.linalg.svd(patch-centroid,full_matrices=False)
    if s[1]<1e-5 or s[2]/math.sqrt(len(patch))>tolerance_m/3:return empty
    normal=axes[-1]
    on_plane=reliable & (np.abs((points-centroid)@normal)<=tolerance_m)
    components,_=label(on_plane)
    index=components[v,u]
    if not index:return empty
    selected=components==index
    if not 20<=int(selected.sum())<=32768:return empty
    center=vector(center,3);rotation=np.asarray(rotation,float);half=vector(half_extents,3)
    if rotation.shape!=(3,3) or np.any(half<=0):raise ValueError('Invalid target surface envelope')
    local=(points[selected]-center)@rotation
    # Validate, do not crop: a component leaking onto furniture loses ownership.
    if np.any(np.abs(local)>half+tolerance_m):return empty
    return selected


def supported_target_component(points, valid, pixel, center, rotation, half_extents, gravity_up, support=None):
    """Seed-connected raised surface above an independently observed support.

    The support is fitted OUTSIDE the estimated object, and is excluded only
    from ownership, never from obstacle memory. Grow without clipping to the
    envelope; any connected spill beyond it invalidates the whole component.
    This is a sensor hypothesis, not a simulator object-identity query.
    """
    from collections import deque
    points=np.asarray(points,float);valid=np.asarray(valid)
    if points.shape!=valid.shape+(3,) or valid.dtype!=bool or valid.ndim!=2:
        raise ValueError('Invalid supported target sensor input')
    center=vector(center,3);rotation=np.asarray(rotation,float);half=vector(half_extents,3)
    up=vector(gravity_up,3)
    if (rotation.shape!=(3,3) or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6)
            or not np.isclose(np.linalg.det(rotation),1.) or np.any(half<=0)
            or not np.isclose(np.linalg.norm(up),1.)):
        raise ValueError('Invalid supported target frame')
    h,w=valid.shape;u,v=pixel;empty=np.zeros(valid.shape,bool)
    if type(u) is not int or type(v) is not int or not(0<=u<w and 0<=v<h):
        raise ValueError('Invalid supported target pixel')
    reliable=valid & np.isfinite(points).all(axis=2)
    if not reliable[v,u]:return empty
    heights=points@up;seed=points[v,u];seed_height=heights[v,u]
    outside=np.any(np.abs((points-center)@rotation)>half+.005,axis=2)
    around=np.linalg.norm(points-seed,axis=2)<.30
    candidates=reliable & outside & around & (heights<seed_height-.008) & (heights>seed_height-.15)
    values=heights[candidates]
    if len(values)<128:return empty
    bins=np.floor(values/.0025).astype(np.int64);keys,counts=np.unique(bins,return_counts=True)
    height=None
    for key in keys[np.argsort(counts)[::-1]][:8]:
        samples=points[candidates & (np.abs(heights-(key+.5)*.0025)<=.0025)]
        if len(samples)<128:continue
        plane=float(np.median(samples@up));residual=samples@up-plane
        _,singular,_=np.linalg.svd(samples-samples.mean(axis=0),full_matrices=False)
        if (np.max(np.abs(residual))<=.003 and singular[1]/math.sqrt(len(samples))>=.025
                and np.ptp(samples@up)<=.005):
            height=plane;break
    if height is None:return empty
    eligible=reliable & (heights>height+.004)
    if not eligible[v,u]:return empty
    side_height=2*float(np.abs(rotation.T@up)@half)
    selected=empty.copy();selected[v,u]=True;pending=deque([(v,u)]);count=1
    while pending:
        y,x=pending.popleft()
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if not(0<=yy<h and 0<=xx<w) or selected[yy,xx] or not eligible[yy,xx]:continue
            delta=points[yy,xx]-points[y,x]
            vertical=float(delta@up)
            # A grazing view can sample a thin vertical side in only one
            # pixel. Its height jump is not a lateral gap to another object.
            # Keep the half-fine-voxel lateral bound and the proposed object's
            # finite height; support pixels and any component spill still fail.
            side_connected=(np.linalg.norm(delta-vertical*up)<=.0025
                            and abs(vertical)<=side_height)
            if np.linalg.norm(delta)>.008 and not side_connected:continue
            if np.any(np.abs((points[yy,xx]-center)@rotation)>half+.0025):return empty
            selected[yy,xx]=True;pending.append((yy,xx));count+=1
            if count>32768:return empty
    if count>=20 and support is not None:
        support.update(height=float(height),mask=reliable & around & (np.abs(heights-height)<=.003))
    return selected if count>=20 else empty


def wholly_owned_voxels(points, selected, voxel_m):
    """Return keys only when EVERY measured contributor belongs to the surface.

    A mixed target/furniture voxel remains an obstacle, even if its centre falls
    inside the target envelope. Input points include all reliable occupied pixels.
    """
    points=np.asarray(points,float);selected=np.asarray(selected)
    if points.ndim!=2 or points.shape[1]!=3 or selected.shape!=(len(points),) or selected.dtype!=bool:
        raise ValueError('Invalid surface voxel membership')
    if not np.isfinite(points).all() or not 0<voxel_m<=.1:raise ValueError('Invalid surface voxel scale')
    keys=np.floor(points/voxel_m).astype(np.int64)
    unique,inverse=np.unique(keys,axis=0,return_inverse=True)
    rejected=np.zeros(len(unique),bool)
    np.logical_or.at(rejected,inverse,~selected)
    return unique[~rejected]


def owned_surface_mask(records, attachment):
    """Membership in exact retained source voxels, not proximity to an OBB."""
    grants={r['geometry_id']:r for r in (attachment or {}).get('surface_memberships',[])}
    pieces=[]
    for row in records:
        if row.get('status')!='occupied' or row.get('stale') or row.get('expired'):continue
        size=len(row['centers_odom_m']) if row.get('kind')=='voxel_cloud' else 1
        owned=np.zeros(size,bool);grant=grants.get(row['id'])
        if grant is not None:
            if row['kind']!='voxel_cloud' or grant['evidence']!=row['evidence']:
                raise ValueError('Surface ownership source mismatch')
            indices=np.asarray(grant['indices'],int)
            if np.any(indices<0) or np.any(indices>=size):raise ValueError('Surface ownership index')
            owned[indices]=True
        pieces.append(owned)
    return np.concatenate(pieces) if pieces else np.empty(0,bool)


def surface_memberships(records, points, valid, selected, evidence):
    """Map a raw-pixel surface mask onto only unmixed, exact source voxels."""
    points=np.asarray(points,float);valid=np.asarray(valid);selected=np.asarray(selected)
    if valid.dtype!=bool or selected.dtype!=bool or valid.shape!=selected.shape or points.shape!=valid.shape+(3,):
        raise ValueError('Surface source shape')
    if np.any(selected&~valid):raise ValueError('Invalid surface pixels cannot own geometry')
    digest=hashlib.sha256(np.asarray(selected.shape,dtype='<i4').tobytes()+np.packbits(selected).tobytes()).hexdigest()
    rows=[];cache={}
    for row in records:
        if row.get('kind')!='voxel_cloud' or row.get('status')!='occupied' or row['evidence']!=evidence:continue
        radii=np.asarray(row['radii_m'],float)
        if not len(radii) or not np.all(radii==radii[0]):raise ValueError('Uniform source voxel scale required')
        voxel=float(2*radii[0]/math.sqrt(3))
        if voxel>.1:continue  # coarse, ambiguous distant surfaces retain all obstacles
        if voxel not in cache:
            keys=wholly_owned_voxels(points[valid],selected[valid],voxel)
            cache[voxel]={tuple(key) for key in keys}
        centers=np.asarray(row['centers_odom_m'])
        keys=np.rint(centers/voxel-.5).astype(np.int64)
        if not np.allclose((keys+.5)*voxel,centers,atol=1e-10,rtol=0):raise ValueError('Source voxel lattice mismatch')
        indices=[i for i,key in enumerate(keys) if tuple(key) in cache[voxel]]
        if indices:rows.append(dict(geometry_id=row['id'],indices=indices,evidence=dict(evidence),mask_sha256=digest))
    return rows


def occupied_arrays(records):
    """Transient numeric view; EpisodeMemory remains the only persistent owner."""
    points, radii, sources = [], [], []
    for row in records:
        if row.get('status') != 'occupied' or row.get('stale') or row.get('expired'):
            continue
        if row.get('kind') == 'voxel_cloud':
            p, r = np.asarray(row['centers_odom_m'], float), np.asarray(row['radii_m'], float)
            if p.ndim != 2 or p.shape[1] != 3 or r.shape != (len(p),):
                raise ValueError('invalid remembered voxel cloud')
        elif row.get('kind') in ('point', 'ray'):
            p = vector(row.get('point_odom_m', row.get('end_odom_m')), 3)[None, :]
            r = np.asarray([row.get('radius_m', 0.)], float)
        else:
            continue
        if not np.isfinite(p).all() or not np.isfinite(r).all() or np.any(r < 0):
            raise ValueError('invalid remembered sensor geometry')
        points.append(p); radii.append(r); sources.extend([row]*len(p))
    return (np.concatenate(points) if points else np.empty((0, 3)),
            np.concatenate(radii) if radii else np.empty(0), sources)


class RobotGeometry(dict):
    """Validated link-local envelopes and their complete robot source pin."""
    @classmethod
    def from_manifest(cls, value):
        if (not isinstance(value, dict) or value.get('robot_only') is not True
                or value.get('contract') != 'fetch-link-local-collision-aabb-v1'
                or not isinstance(value.get('source_files'), dict)
                or not isinstance(value.get('links'), dict) or not value['links']):
            raise ValueError('robot-only collision manifest required')
        hashes = [value.get('urdf_sha256')] + list(value['source_files'].values())
        if any(not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
            raise ValueError('invalid robot source hashes')
        if value['source_files'].get('fetch.urdf') != value['urdf_sha256']:
            raise ValueError('robot URDF source pin mismatch')
        pin = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        result = cls({name: dict(bounds=record['bounds'], sha256=pin)
                      for name, record in value['links'].items()})
        result.urdf_sha256 = value['urdf_sha256']
        result.source_files = dict(value['source_files'])
        result.manifest_sha256 = pin
        return result

    @staticmethod
    def base_convex_vertices(root, manifest):
        return RobotGeometry.link_convex_vertices(root,manifest,'base_link')

    @staticmethod
    def link_convex_vertices(root, manifest, name):
        """Own calibrated robot geometry only; never a scene collision mesh."""
        from pathlib import Path
        import xml.etree.ElementTree as ET
        import trimesh
        from scipy.spatial.transform import Rotation
        from scipy.spatial import ConvexHull
        if name not in {'base_link','gripper_link',*FINGERS}:raise ValueError('Unsupported calibrated convex link')
        root=Path(root).resolve()
        urdf=root/'fetch.urdf'
        if hashlib.sha256(urdf.read_bytes()).hexdigest()!=manifest['urdf_sha256']:
            raise ValueError('Robot URDF hash mismatch')
        link=ET.parse(urdf).getroot().find("link[@name='%s']"%name)
        vertices=[]
        for collision in link.findall('collision'):
            mesh=collision.find('geometry/mesh')
            if mesh is None:raise ValueError('Expected pinned base mesh')
            path=(root/mesh.get('filename').replace('package://','')).resolve()
            if not path.is_relative_to(root):raise ValueError('Robot geometry path escaped root')
            expected=manifest['source_files'].get(path.relative_to(root).as_posix())
            if expected!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('Robot mesh hash mismatch')
            v=np.asarray(trimesh.load(path,force='mesh',process=False).vertices)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            origin=collision.find('origin')
            if origin is not None:
                v=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).apply(v)+np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            vertices.append(v)
        v=np.concatenate(vertices)
        # A convex hull contains the full triangulated body, including faces
        # between vertices. It removes AABB corner voids, not real robot parts.
        return v[ConvexHull(v).vertices]


class SensorClearance:
    """Stateless geometry consumer. ``geometry`` must be a robot-only pin.

    A pin maps link names to {bounds:[[min],[max]], sha256:<collision source hash>}.
    Bounds include each URDF collision origin/scale in the named link frame.
    The caller is responsible for checking the pin against the installed robot.
    ``fk.links(qpos)`` supplies all link poses relative to the current base.
    """
    def __init__(self, fk, memory, *, geometry, config=None):
        self.fk, self.memory = fk, memory
        self.config = config or ClearanceConfig()
        if not isinstance(self.config, ClearanceConfig):
            self.config = ClearanceConfig(**self.config)
        if not isinstance(geometry, dict) or not geometry:
            raise ValueError('calibrated robot-only geometry required')
        self.geometry = {}
        self.convex_planes = {}
        self.convex_vertices = {}
        # Opt-in CPU diagnostic sink; never included in model/tool feedback.
        self.coverage_diagnostics = None
        for name, record in geometry.items():
            bounds = np.asarray(record['bounds'], float)
            digest = record['sha256']
            if (not isinstance(name, str) or bounds.shape != (2, 3)
                    or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0])
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError('invalid robot geometry pin')
            self.geometry[name] = dict(bounds=bounds.copy(), sha256=digest)
            if 'convex_vertices' in record:
                from scipy.spatial import ConvexHull
                vertices=np.asarray(record['convex_vertices'],float)
                if (name not in {'base_link','gripper_link',*FINGERS} or vertices.ndim!=2 or vertices.shape[1]!=3
                        or not np.isfinite(vertices).all() or np.any(vertices<bounds[0]-1e-9)
                        or np.any(vertices>bounds[1]+1e-9)):
                    raise ValueError('Invalid calibrated base convex hull')
                self.convex_planes[name]=ConvexHull(vertices).equations
                self.convex_vertices[name]=vertices.copy()

    def _body_gaps(self, points, radii, pose, bounds, name, padding=0.):
        gaps=_signed_distance(points,pose,bounds)-radii-self.config.margin_m
        planes=self.convex_planes.get(name)
        if planes is not None:
            indices=np.flatnonzero(gaps<=padding)
            for ids in np.array_split(indices,max(1,math.ceil(len(indices)/512))):
                local=(points[ids]-pose[:3,3])@pose[:3,:3]
                gaps[ids]=(local@planes[:,:3].T+planes[:,3]).max(axis=1)-radii[ids]-self.config.margin_m
        return gaps

    def _body_translation_hits(self, points, radii, before, after, bounds, name):
        hits=self._translation_hits(points,radii,before,after,bounds,self.config.margin_m)
        planes=self.convex_planes.get(name)
        if planes is None or not hits.any():return hits
        # Intersect each sphere-center ray with the hull's expanded halfspaces.
        # Expansion is an outer bound on the sphere/hull Minkowski sum.
        ids=np.flatnonzero(hits);normal=planes[:,:3]
        delta=(after[:3,3]-before[:3,3])@before[:3,:3]
        slope=normal@delta
        for chunk in np.array_split(ids,max(1,math.ceil(len(ids)/512))):
            local=(points[chunk]-before[:3,3])@before[:3,:3]
            offset=local@normal.T+planes[:,3]-radii[chunk,None]-self.config.margin_m
            lower=np.zeros(len(chunk));upper=np.ones(len(chunk));valid=np.ones(len(chunk),bool)
            for j,s in enumerate(slope):
                if s>0:lower=np.maximum(lower,offset[:,j]/s)
                elif s<0:upper=np.minimum(upper,offset[:,j]/s)
                else:valid &= offset[:,j]<=0
            hits[chunk]=valid & (lower<=upper)
        return hits

    def _body_se2_hits(self, points, radii, before, after, name):
        """Conservative base hull sweep, including simultaneous turn/translation.

        Interpolated base translation is affine. Each rotating vertex differs
        from its endpoint chord by at most r*angle**2/8. This avoids padding the
        entire body by forward travel when yaw is merely nonzero.
        """
        from scipy.spatial import ConvexHull
        vertices=self.convex_vertices[name]
        relative=np.linalg.inv(before)@after
        if not np.allclose(relative[2],[0,0,1,0],atol=1e-10,rtol=0):
            raise ValueError('Base convex sweep requires planar odometry')
        angle=abs(math.atan2(relative[1,0],relative[0,0]))
        arc_bound=float(np.linalg.norm(vertices,axis=1).max())*angle**2/8
        endpoints=np.concatenate((vertices,vertices@relative[:3,:3].T+relative[:3,3]))
        local_points=(points-before[:3,3])@before[:3,:3]
        padding=np.asarray(radii)+self.config.margin_m+arc_bound
        # Every interpolated vertex lies within the endpoint AABB enlarged by
        # arc_bound. Sphere uncertainty and margin enlarge each axis as well.
        # Reject only points outside this conservative swept outer bound.
        candidates=np.flatnonzero(np.all(local_points>=endpoints.min(axis=0)-padding[:,None],axis=1)
                                  & np.all(local_points<=endpoints.max(axis=0)+padding[:,None],axis=1))
        hits=np.zeros(len(points),bool)
        if not len(candidates):return hits
        hull=ConvexHull(endpoints)
        planes=hull.equations
        for ids in np.array_split(candidates,max(1,math.ceil(len(candidates)/512))):
            local=local_points[ids]
            gaps=(local@planes[:,:3].T+planes[:,3]).max(axis=1)
            hits[ids]=gaps<=radii[ids]+self.config.margin_m+arc_bound
        return hits

    def _articulated_curve_bound(self, before, after, body, vertices, mount_offset=0.):
        """Joint-linear vertex/chord error from the calibrated ancestor chain.

        For a product of rotations and affine translations, ||x''(t)|| is
        bounded by L*A^2 + 2*A*V. The endpoint chord error on t=[0,1] is at
        most that bound /8. L bounds the chain length, A the summed absolute
        angular changes, and V the summed prismatic travel. Base translation
        is affine; planar base yaw participates in A. No scene data is used.
        """
        if type(self.fk) is not RobotFK:raise ValueError('Calibrated FK required for joint curve bound')
        qa,oa=before;qb,ob=after
        ya,yb=math.atan2(oa[1,0],oa[0,0]),math.atan2(ob[1,0],ob[0,0])
        angular=abs(math.atan2(math.sin(yb-ya),math.cos(yb-ya)))
        length=float(np.linalg.norm(vertices,axis=1).max())+float(mount_offset);travel=0.
        parents={child:(name,kind,parent,origin) for name,kind,parent,child,origin,_ in self.fk.joints}
        link='gripper_link' if body=='carried_object' else body;seen=set()
        while link!='base_link':
            if link in seen or link not in parents:raise ValueError('Robot chain missing')
            seen.add(link);name,kind,link,origin=parents[link]
            length+=float(np.linalg.norm(origin[:3,3]))
            if kind=='fixed':continue
            i=self.fk.index[name];delta=abs(float(qb[i]-qa[i]))
            if kind=='prismatic':
                length+=max(abs(float(qa[i])),abs(float(qb[i])));travel+=delta
            elif kind in ('revolute','continuous'):angular+=delta
            else:raise ValueError('Unsupported robot joint')
        return (length*angular*angular+2*angular*travel)/8.

    def _body_articulated_hits(self, points, radii, before, after, before_pose, after_pose, bounds, name, attachment=None):
        from scipy.spatial import ConvexHull
        vertices=self.convex_vertices.get(name,_corners(bounds))
        mount_offset=0. if name!='carried_object' else float(np.linalg.norm(attachment['center_gripper_m']))
        error=self._articulated_curve_bound(before,after,name,vertices,mount_offset)
        relative=np.linalg.inv(before_pose)@after_pose
        endpoints=np.concatenate((vertices,vertices@relative[:3,:3].T+relative[:3,3]))
        local=(points-before_pose[:3,3])@before_pose[:3,:3]
        padding=np.asarray(radii)+self.config.margin_m+error
        ids=np.flatnonzero(np.all(local>=endpoints.min(axis=0)-padding[:,None],axis=1)
                          &np.all(local<=endpoints.max(axis=0)+padding[:,None],axis=1))
        hits=np.zeros(len(points),bool)
        if not len(ids):return hits
        planes=ConvexHull(endpoints).equations
        for chunk in np.array_split(ids,max(1,math.ceil(len(ids)/512))):
            gaps=(local[chunk]@planes[:,:3].T+planes[:,3]).max(axis=1)
            hits[chunk]=gaps<=padding[chunk]
        return hits

    def _boxes(self, qpos, odometry, attachment=None):
        qpos = vector(qpos, 15)
        base = odometry_transform(odometry)
        links = self.fk.links(qpos)
        if set(self.geometry) - set(links):
            raise ValueError('calibrated robot link missing from FK')
        boxes={name: (base @ transform(links[name]), row['bounds'])
               for name, row in self.geometry.items()}
        if attachment is not None:
            mount=np.eye(4);mount[:3,:3]=attachment['rotation_gripper'];mount[:3,3]=attachment['center_gripper_m']
            half=np.asarray(attachment['half_extents_m'])
            # Estimate and closure each lie within the unchanged 2mm/.02rad
            # alignment gate. Cover their relative frame difference as well
            # as the declared visual uncertainty; do not silently treat them
            # as identical gripper poses.
            radius=np.linalg.norm(attachment['center_gripper_m'])+np.linalg.norm(half)
            frame_error=.004+2*math.sin(.02)*radius
            extent=half+attachment['uncertainty_m']+frame_error
            boxes['carried_object']=(base@transform(links['gripper_link'])@transform(mount),np.array([-extent,extent]))
        return boxes

    def ingest_observation(self, observation_id, qpos, odometry, cameras, *,
                           control_step, calibration_sha256):
        """Extract occupied voxels and conservative free frusta, without storing.

        Every valid depth pixel contributes before voxel reduction, including thin
        edges. A free tile requires *every* pixel valid and outside self envelopes;
        its far plane is the nearest depth in the tile, with a margin. Ambiguous
        points inside robot AABBs are omitted but never create free rays behind the
        robot. The registered frame hash/step is validated by EpisodeMemory.
        """
        if (not isinstance(observation_id, str) or not observation_id
                or isinstance(control_step, bool) or int(control_step) != control_step or control_step < 0
                or not isinstance(calibration_sha256, str) or len(calibration_sha256) != 64
                or any(c not in '0123456789abcdef' for c in calibration_sha256)):
            raise ValueError('invalid synchronized clearance observation')
        if set(cameras) != {'head', 'hand'}:
            raise ValueError('head and hand robot depth required')
        cfg = self.config
        base = odometry_transform(odometry)
        boxes = self._boxes(qpos, base)
        records = []
        for camera, values in cameras.items():
            depth = np.asarray(values['depth_mm'])
            K = np.asarray(values['intrinsics'], float)
            mount = transform(values['camera_in_base'])
            if (depth.ndim != 2 or min(depth.shape) < 2 or not np.issubdtype(depth.dtype, np.integer)
                    or K.shape != (3, 3) or not np.isfinite(K).all()
                    or K[0, 0] <= 0 or K[1, 1] <= 0
                    or not np.allclose(K[2], [0, 0, 1])):
                raise ValueError('invalid raw RGB-D calibration')
            evidence = dict(observation_id=observation_id, camera=camera,
                            control_step=int(control_step), calibration_sha256=calibration_sha256)
            T = base @ mount
            R = T[:3, :3] @ OPTICAL_TO_CAMERA
            z = depth.astype(float) / 1000.
            valid = (z >= cfg.min_depth_m) & (z <= cfg.max_depth_m)
            vv, uu = np.indices(depth.shape)
            optical = np.stack((uu, vv, np.ones_like(uu)), axis=-1) @ np.linalg.inv(K).T
            points = (optical * z[..., None]) @ R.T + T[:3, 3]
            flat = points.reshape(-1, 3)
            self_ambiguous = np.zeros(len(flat), dtype=bool)
            near_body = np.zeros(len(flat), dtype=bool)
            finger_near = np.zeros(len(flat), dtype=bool)
            for pose, bounds in boxes.values():
                shell = _corners([bounds[0]-cfg.depth_margin_m, bounds[1]+cfg.depth_margin_m]) @ pose[:3, :3].T + pose[:3, 3]
                low, high = shell.min(axis=0), shell.max(axis=0)
                near_body |= ((flat[:, 0] >= low[0]-.35) & (flat[:, 0] <= high[0]+.35) &
                              (flat[:, 1] >= low[1]-.35) & (flat[:, 1] <= high[1]+.35) &
                              (flat[:, 2] >= low[2]-.35) & (flat[:, 2] <= high[2]+.35))
                candidates = np.flatnonzero(valid.ravel() & ~self_ambiguous &
                    (flat[:, 0] >= low[0]) & (flat[:, 0] <= high[0]) &
                    (flat[:, 1] >= low[1]) & (flat[:, 1] <= high[1]) &
                    (flat[:, 2] >= low[2]) & (flat[:, 2] <= high[2]))
                self_ambiguous[candidates] = _inside(flat[candidates], pose, bounds, cfg.depth_margin_m)
            for name in ('gripper_link', 'l_gripper_finger_link', 'r_gripper_finger_link'):
                if name not in boxes:
                    continue
                pose, bounds = boxes[name]
                shell = _corners(bounds) @ pose[:3, :3].T + pose[:3, 3]
                lo, hi = shell.min(axis=0)-cfg.precision_reach_m, shell.max(axis=0)+cfg.precision_reach_m
                finger_near |= np.all((flat >= lo) & (flat <= hi), axis=1)
            self_ambiguous = self_ambiguous.reshape(z.shape)
            reliable = valid & ~self_ambiguous
            # Calibrated robot support height restricts this to the ground near
            # the wheels. A desk/cabinet top cannot become a floor merely by
            # being horizontal. No plane is extrapolated into unobserved tiles.
            floor_expected = cfg.support_plane_odom_z_m
            if floor_expected is None:
                floor_expected = min(float((_corners(b) @ p[:3, :3].T+p[:3, 3])[:, 2].min())
                    for name, (p, b) in boxes.items() if name == 'base_link' or 'wheel' in name)
            floor_candidates = points[reliable & (np.abs(points[..., 2]-floor_expected) <= .025)]
            floor_mask = np.zeros(z.shape, bool)
            floor_plane = None
            if cfg.local_support_floor:
                floor_mask, floor_plane = _local_ground_mask(points, reliable, floor_expected,
                                                             cfg.support_plane_tolerance_m)
            elif len(floor_candidates) >= 128:
                estimate = float(np.median(floor_candidates[:, 2]))
                supported = floor_candidates[np.abs(floor_candidates[:, 2]-estimate) <= cfg.support_plane_tolerance_m]
                if (len(supported) >= 128 and np.ptp(supported[:, 0]) >= .15
                        and np.ptp(supported[:, 1]) >= .15):
                    floor_plane = estimate
                    floor_mask = reliable & (np.abs(points[..., 2]-estimate) <= cfg.support_plane_tolerance_m)
            obstacle_pixels = reliable & ~floor_mask
            occupied = points[obstacle_pixels]
            if len(occupied):
                # Integer voxel centres and a circumsphere preserve every point;
                # averages/strided pixels would erase thin furniture edges.
                ranges = np.linalg.norm(occupied-base[:3, 3], axis=1)
                fine = finger_near.reshape(z.shape)[obstacle_pixels]
                protected = (ranges <= 1.5) | near_body.reshape(z.shape)[obstacle_pixels]
                fine_keys = _voxel_keys(occupied[fine], cfg.finger_voxel_m)
                near_points = occupied[protected & ~fine]
                near_keys = _voxel_keys(near_points, cfg.voxel_m)
                # Near furniture is never inflated merely to fit a record cap.
                # Capacity exhaustion is explicit unknown rather than a false
                # contact inferred from an enormous voxel around a close edge.
                if len(near_keys)+len(fine_keys) > cfg.max_occupied_per_camera:
                    raise ValueError('near-field sensor geometry exceeds memory capacity')
                groups = [(near_keys, cfg.voxel_m, 'near'), (fine_keys, cfg.finger_voxel_m, 'finger')]
                remaining = cfg.max_occupied_per_camera-len(near_keys)-len(fine_keys)
                far_points = occupied[~protected & ~fine]
                if len(far_points):
                    if remaining < 8:
                        raise ValueError('sensor geometry needs room for far evidence')
                    voxel = max(.075, cfg.voxel_m)
                    keys = _voxel_keys(far_points, voxel)
                    while len(keys) > remaining:
                        voxel *= 1.25
                        keys = _voxel_keys(far_points, voxel)
                    groups.append((keys, voxel, 'far'))
                for keys, voxel, level in groups:
                    centers = (keys + .5) * voxel
                    radius = math.sqrt(3.) * voxel / 2
                    radii = np.full(len(centers), radius)
                    if cfg.tight_near_voxels and level == 'near':
                        centers, radii = _measured_voxel_envelopes(near_points, voxel, cfg.depth_margin_m)
                    if len(centers):
                        records.append(dict(id=f'occupied:{observation_id}:{camera}:{level}', status='occupied',
                            kind='voxel_cloud', centers_odom_m=centers.tolist(),
                            radii_m=radii.tolist(), evidence=dict(evidence)))
            tile = cfg.tile_pixels
            h, w = depth.shape
            for v in range(0, h, tile):
                for u in range(0, w, tile):
                    v1, u1 = min(h, v + tile), min(w, u + tile)
                    if not reliable[v:v1, u:u1].all():
                        continue
                    far = float(z[v:v1, u:u1].min()) - cfg.depth_margin_m
                    support = (floor_plane is not None and floor_mask[v:v1, u:u1].all())
                    if support:
                        far = float(z[v:v1, u:u1].max())
                    if far <= cfg.min_depth_m:
                        continue
                    row = dict(id=f'free:{observation_id}:{camera}:{u}:{v}',
                        status='free', kind='frustum', origin_odom_m=T[:3, 3].tolist(),
                        rotation_odom_from_optical=R.tolist(), intrinsics=K.tolist(),
                        pixel_bounds=[max(0., u-.5), max(0., v-.5), min(w-1., u1-.5), min(h-1., v1-.5)], near_m=cfg.min_depth_m, far_m=far,
                        evidence=dict(evidence))
                    if support:
                        row['support_plane_odom_z_m'] = floor_plane+cfg.support_plane_tolerance_m
                    records.append(row)
        return records

    def associate_carried_surface(self, observation_id, qpos, odometry, cameras, *, control_step):
        """Associate a seed-connected surface with an estimated/possible attachment.

        Expected rigid motion is a tracking hypothesis, not an is_grasped signal.
        Missing projection, depth disagreement, or a leaking surface retains all
        points as ordinary obstacles. No volume-wide removal or free rays.
        """
        attachment=self.memory.attachment()
        if (attachment is None or attachment['status'] not in ('estimated','possibly_attached')
                or 'surface_seed_gripper_m' not in attachment):return 0
        binding=self.memory.observation_binding(observation_id)
        calibration={name:{key:np.asarray(value).tolist() for key,value in camera.items()
            if key!='depth_mm'} for name,camera in cameras.items()}
        canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
        if (binding['control_step']!=control_step or binding['odometry']!=odometry
                or hashlib.sha256(canonical(calibration)).hexdigest()!=binding['calibration_sha256']):
            raise ValueError('Carried surface frame binding mismatch')
        B=odometry_transform(odometry);links=self.fk.links(qpos);G=B@transform(links['gripper_link'])
        if attachment['status']=='estimated':
            if 'estimate_gripper_odom' in attachment:G=transform(attachment['estimate_gripper_odom'])
            elif control_step!=attachment['control_step']:return 0
        expected=G[:3,:3]@np.asarray(attachment['surface_seed_gripper_m'])+G[:3,3]
        boxes=self._boxes(qpos,odometry);records=self.memory.geometry(step=control_step)
        grants=[];supports=[]
        for camera,values in cameras.items():
            raw=np.asarray(values['depth_mm'],dtype='<i4')
            digest=hashlib.sha256(canonical(dict(shape=list(raw.shape),encoding='little-endian-int32-mm'))+raw.tobytes()).hexdigest()
            if digest!=binding['sensor_sha256'].get(camera+'_depth'):raise ValueError('Carried surface raw depth mismatch')
            K=np.asarray(values['intrinsics']);T=B@transform(values['camera_in_base']);R=T[:3,:3]@OPTICAL_TO_CAMERA
            optical=(expected-T[:3,3])@R
            if optical[2]<=self.config.min_depth_m:continue
            uv=K@optical;u,v=np.rint(uv[:2]/uv[2]).astype(int)
            h,w=raw.shape
            if not (8<=u<w-8 and 8<=v<h-8):continue
            z=raw.astype(float)/1000.
            if abs(z[v,u]-optical[2])>min(.01,attachment['uncertainty_m']):continue
            y,x=np.indices(raw.shape)
            points=(np.stack((x,y,np.ones_like(x)),axis=-1)@np.linalg.inv(K).T*z[...,None])@R.T+T[:3,3]
            valid=(z>=self.config.min_depth_m)&(z<=self.config.max_depth_m)
            for pose,bounds in boxes.values():
                # Surface association excludes the robot with a depth-noise
                # shell, not the broader ambiguity shell used for free-space
                # extraction. Geometry ingestion itself is unchanged; a label
                # cannot restore omitted pixels or create free space.
                valid&=~_inside(points.reshape(-1,3),pose,bounds,min(.0025,self.config.depth_margin_m)).reshape(raw.shape)
            grip=(points-G[:3,3])@G[:3,:3]
            selected=target_surface_component(grip,valid,[int(u),int(v)],attachment['center_gripper_m'],
                attachment['rotation_gripper'],np.asarray(attachment['half_extents_m'])+attachment['uncertainty_m'])
            support={}
            selected |= supported_target_component(grip,valid,[int(u),int(v)],attachment['center_gripper_m'],
                attachment['rotation_gripper'],np.asarray(attachment['half_extents_m'])+attachment['uncertainty_m'],
                G[:3,:3].T@np.array([0.,0.,1.]),support)
            if not selected.any():continue
            # A read-only sequence can produce several IDs for identical raw
            # evidence. Reuse ownership only for byte-identical calibrated data
            # in exactly the same odometry frame, never by spatial proximity.
            sources={r['evidence']['observation_id']:r['evidence'] for r in records
                if r.get('kind')=='voxel_cloud' and r['evidence']['camera']==camera}
            for oid,evidence in sources.items():
                old=self.memory.observation_binding(oid)
                if (old['calibration_sha256']!=binding['calibration_sha256'] or old['odometry']!=odometry
                        or old['sensor_sha256'].get(camera+'_depth')!=digest):continue
                already={r['geometry_id'] for r in attachment.get('surface_memberships',[])}
                grants.extend(g for g in surface_memberships(records,points,valid,selected,evidence)
                              if g['geometry_id'] not in already)
                if support:
                    memberships=surface_memberships(records,points,valid,support['mask'],evidence)
                    if memberships:
                        mount=np.eye(4);mount[:3,:3]=attachment['rotation_gripper'];mount[:3,3]=attachment['center_gripper_m']
                        supports.append(dict(height_odom_m=float(support['height']+G[2,3]),
                            anchor_object_odom=(G@mount).tolist(),memberships=memberships))
        if grants:self.memory.record_surface_memberships(grants,candidate_id=attachment['candidate_id'],step=control_step)
        for support in supports:self.memory.record_attachment_support(support,candidate_id=attachment['candidate_id'],step=control_step)
        return sum(len(r['indices']) for r in grants)

    @staticmethod
    def _report(status, scope, *, body='', clearance=None, step=None, reason=None, accepted=False):
        report = dict(accepted=bool(accepted), clearance_status=status,
                      clearance_body=body, clearance_scope=scope,
                      reason=reason or ('sensor_clearance_' + status))
        if step is not None:
            report['clearance_evidence_step'] = int(step)
        if clearance is not None and math.isfinite(clearance):
            report['clearance_distance_m'] = float(clearance)
        return report

    def _records(self, step):
        records = self.memory.geometry(step=step)
        if isinstance(records, dict):
            records = records.get('records', [])
        if not isinstance(records, (list, tuple)):
            raise ValueError('memory geometry must return sensor records')
        current, old = [], []
        for record in records:
            evidence = record.get('evidence', {})
            observed = evidence.get('control_step', -1)
            if not isinstance(observed, int) or observed < 0 or observed > step:
                raise ValueError('future or invalid geometry evidence')
            if record.get('stale') or record.get('expired'):
                old.append(record)
            elif record.get('status') in ('occupied', 'free'):
                current.append(record)
        return current, old

    @staticmethod
    def _free_groups(records):
        groups = {}
        for row in records:
            if row.get('status') != 'free' or row.get('kind') != 'frustum':
                continue  # A radius-bearing free ray is not evidence between rays.
            key = json.dumps([row['origin_odom_m'], row['rotation_odom_from_optical'],
                              row['intrinsics'], row['evidence']['observation_id'],
                              row['evidence']['camera']], separators=(',', ':'))
            if key not in groups:
                groups[key] = dict(origin=vector(row['origin_odom_m'], 3),
                    R=np.asarray(row['rotation_odom_from_optical'], float),
                    K=np.asarray(row['intrinsics'], float), tiles=[])
            groups[key]['tiles'].append(row)
        for group in groups.values():
            group['bounds'] = np.array([r['pixel_bounds'] for r in group['tiles']], float)
            group['near'] = np.array([r['near_m'] for r in group['tiles']], float)
            group['far'] = np.array([r['far_m'] for r in group['tiles']], float)
            group['support_plane'] = np.array([r.get('support_plane_odom_z_m', -np.inf) for r in group['tiles']], float)
        return list(groups.values())

    @staticmethod
    def _covered(corners, groups):
        """Cover each convex cell using complete image tiles from one view.

        Checking only sample points would miss unknown holes. We require every
        tile touched by the projected bounding rectangle to exist and its far
        plane to exceed the furthest cell corner. This is conservative for the
        complete cell, including its interior.
        """
        result = np.zeros(len(corners), dtype=bool)
        for group in groups:
            remaining = np.flatnonzero(~result)
            if not len(remaining):
                break
            optical = (corners[remaining] - group['origin']) @ group['R']
            zmin, zmax = optical[..., 2].min(axis=1), optical[..., 2].max(axis=1)
            projection = optical @ group['K'].T
            uv = projection[..., :2] / np.maximum(projection[..., 2:], 1e-12)
            low, high = uv.min(axis=1), uv.max(axis=1)
            bounds, near, far = group['bounds'], group['near'], group['far']
            for i, index in enumerate(remaining):
                if zmin[i] <= 0:
                    continue
                # Closed cell projection must lie within an unbroken union of
                # non-overlapping source pixel rectangles. Each record covers
                # pixel centres from u0 through u1-1, so half-pixel borders are
                # conservatively clipped at the image edge by the extractor.
                left, top = low[i]
                right, bottom = high[i]
                valid = (near <= zmin[i]) & (far >= zmax[i]) & (group['support_plane'] <= corners[index, :, 2].min()+1e-9)
                overlap = valid & (bounds[:, 0] <= right) & (bounds[:, 2] >= left) & (bounds[:, 1] <= bottom) & (bounds[:, 3] >= top)
                rectangles = bounds[overlap]
                if not len(rectangles):
                    continue
                # Slice at every intersecting tile edge: interval unions then
                # establish coverage, without counting overlaps twice.
                xs = sorted(set([left, right] + [float(x) for x in rectangles[:, [0, 2]].ravel() if left < x < right]))
                complete = True
                for a, b in zip(xs[:-1], xs[1:]):
                    mid = (a + b) / 2
                    spans = sorted((max(top, r[1]), min(bottom, r[3])) for r in rectangles if r[0] <= mid <= r[2] and r[3] >= top and r[1] <= bottom)
                    covered_to = top
                    for c, d in spans:
                        if c > covered_to + 1e-9:
                            break
                        covered_to = max(covered_to, d)
                    if covered_to < bottom - 1e-9:
                        complete = False
                        break
                if complete and len(xs) > 1:
                    result[index] = True
        return result

    def _cells(self, pose, bounds, padding, support_plane=None):
        bounds = np.array([bounds[0] - padding, bounds[1] + padding])
        axes = [np.linspace(bounds[0, j], bounds[1, j],
                  max(1, int(math.ceil((bounds[1, j] - bounds[0, j]) / self.config.coverage_cell_m))) + 1)
                for j in range(3)]
        cells = []
        for i, j, k in itertools.product(*(range(len(a) - 1) for a in axes)):
            lo = [axes[0][i], axes[1][j], axes[2][k]]
            hi = [axes[0][i+1], axes[1][j+1], axes[2][k+1]]
            cells.append(_corners([lo, hi]) @ pose[:3, :3].T + pose[:3, 3])
        cells = np.asarray(cells)
        if support_plane is not None:
            cells = cells[cells[..., 2].max(axis=1) > support_plane]
            cells[..., 2] = np.maximum(cells[..., 2], support_plane)
        return cells

    def _covered_union(self, cells, groups, entry_boxes):
        """Require free evidence only outside the robot's entry volume.

        A boundary cell can be partly self-occluded and partly newly entered.
        Exact convex subtraction splits these portions before checking source
        view coverage. An unresolved numerical split or excessive fragmentation
        fails closed; it never turns an entire mixed cell into known free space.
        """
        covered = np.zeros(len(cells), bool)
        diagnostic_start = len(self.coverage_diagnostics) if self.coverage_diagnostics is not None else 0
        for pose, bounds, margin in entry_boxes:
            covered |= _inside(cells, pose, bounds, margin).all(axis=1)
        pending = np.flatnonzero(~covered)
        if len(pending):
            covered[pending] = self._covered(cells[pending], groups)
        for index in np.flatnonzero(~covered):
            pieces = [cells[index]]
            for pose, bounds, margin in entry_boxes:
                new_pieces = []
                for piece in pieces:
                    outside = _subtract_box(piece, pose, bounds, margin)
                    if outside is None:
                        new_pieces = None
                        break
                    new_pieces.extend(outside)
                    if len(new_pieces) > 128:
                        new_pieces = None
                        break
                if new_pieces is None:
                    pieces = None
                    break
                pieces = new_pieces
                if not pieces:
                    break
            if pieces is not None:
                covered[index] = all(self._covered(piece[None, :, :], groups)[0] for piece in pieces)
            if (not covered[index] and self.coverage_diagnostics is not None
                    and len(self.coverage_diagnostics)-diagnostic_start < 8
                    and len(self.coverage_diagnostics) < 64):
                unknown = None if pieces is None else [p.tolist() for p in pieces
                    if not self._covered(p[None, :, :], groups)[0]]
                self.coverage_diagnostics.append(dict(cell_vertices_odom_m=cells[index].tolist(),
                    unknown_fragments_odom_m=unknown, numerical_split_failed=pieces is None))
        return covered

    def _translation_segment(self, before, after, body=None):
        """Certify affine link trajectories, never infer them from endpoint FK.

        Constant base rotation plus only prismatic joint changes gives constant
        link rotations and affine translations throughout joint interpolation.
        An opaque FK port can certify only unchanged joints. Equal endpoint
        rotations alone do not exclude an intermediate revolute excursion.
        """
        qa, oa = before; qb, ob = after
        if not np.array_equal(oa[:3, :3], ob[:3, :3]):
            return False
        changed = np.flatnonzero(qa != qb)
        if body is not None and type(self.fk) is RobotFK:
            # Certify the body's own chain. A head rotation does not rotate a
            # finger on the separate arm branch. Endpoint rotations alone are
            # insufficient: every changed ancestor must be prismatic.
            parents = {child: (name, parent) for name, _, parent, child, _, _ in self.fk.joints}
            ancestors, seen, link = set(), set(), 'gripper_link' if body=='carried_object' else body
            while link != 'base_link':
                if link in seen or link not in parents:
                    return False
                seen.add(link)
                joint, link = parents[link]
                if joint in self.fk.index:
                    ancestors.add(self.fk.index[joint])
            changed = [i for i in changed if i in ancestors]
        if not len(changed):
            return True
        return (type(self.fk) is RobotFK and all(
            self.fk.limits[self.fk.order[i]][2] == 'prismatic' for i in changed))

    @staticmethod
    def _translation_hits(points, radii, before_pose, after_pose, bounds, margin):
        """Segment/slab intersection against radius-expanded link-local OBB.

        Expanding each face by the occupied voxel radius encloses its sphere.
        The test retains thin obstacles between endpoints without extending
        the path backwards or sideways by the translation length.
        """
        local = (points-before_pose[:3, 3]) @ before_pose[:3, :3]
        velocity = -(after_pose[:3, 3]-before_pose[:3, 3]) @ before_pose[:3, :3]
        low = bounds[0]-margin-radii[:, None]
        high = bounds[1]+margin+radii[:, None]
        enter, leave = np.zeros(len(points)), np.ones(len(points))
        for axis in range(3):
            if velocity[axis] == 0:
                valid = (local[:, axis] >= low[:, axis]) & (local[:, axis] <= high[:, axis])
                leave[~valid] = -1.
            else:
                a = (low[:, axis]-local[:, axis])/velocity[axis]
                b = (high[:, axis]-local[:, axis])/velocity[axis]
                enter = np.maximum(enter, np.minimum(a, b))
                leave = np.minimum(leave, np.maximum(a, b))
        return enter <= leave

    def _translation_cells(self, before_pose, after_pose, bounds, support_plane=None):
        # The hull of corresponding endpoint cells is the exact swept convex
        # prism for affine translation. A cell may have sixteen vertices; the
        # convex clipping/coverage routines do not assume an eight-corner OBB.
        cells = self._cells(before_pose, bounds, self.config.margin_m)
        delta = after_pose[:3, 3]-before_pose[:3, 3]
        cells = np.concatenate([cells, cells+delta], axis=1)
        if support_plane is not None:
            cells = cells[cells[..., 2].max(axis=1) > support_plane]
            cells[..., 2] = np.maximum(cells[..., 2], support_plane)
        return cells

    @staticmethod
    def _contact_mask(points, body, contact_region, radii=None):
        if contact_region is None or body not in FINGERS:
            return np.zeros(len(points), bool)
        if not all(contact_region.get(k) is True for k in ('confirmed', 'locked', 'localization_executed')):
            raise ValueError('finger contact requires confirmed executed locked grasp')
        pose = np.eye(4)
        pose[:3, :3] = np.asarray(contact_region['rotation_odom'], float)
        pose[:3, 3] = vector(contact_region['center_odom_m'], 3)
        pose = transform(pose)
        extents = vector(contact_region['half_extents_m'], 3)
        # Outer ceiling from the pinned Fetch finger span (.060 x .0263 m)
        # and its 0..0.05 m prismatic opening. The runner additionally derives
        # the narrower current inner-pad intersection from calibrated FK.
        if np.any(extents <= 0) or np.any(extents > np.array([.030001, .050001, .013151])):
            raise ValueError('locked contact bounds exceed calibrated finger corridor')
        local = (points-pose[:3, 3])@pose[:3, :3]
        uncertainty = 0. if radii is None else np.asarray(radii)[:, None]
        return np.all(np.abs(local)+uncertainty <= extents, axis=1)

    def _supported_contact_mask(self, records, states, attachment):
        """Only source-labelled support contact for a bounded vertical lift.

        This is not free space. Robot links and unlabelled furniture retain
        their normal checks. The object's original support anchor never moves
        with repeated servo ticks or re-estimates of the same candidate.
        """
        points,_,_=occupied_arrays(records);empty=np.zeros(len(points),bool)
        if attachment is None or 'support_plane' not in attachment or type(self.fk) is not RobotFK:return empty
        support=attachment['support_plane'];anchor=transform(support['anchor_object_odom'])
        mount=np.eye(4);mount[:3,:3]=attachment['rotation_gripper'];mount[:3,3]=attachment['center_gripper_m']
        vertices=_corners(np.array([-np.asarray(attachment['half_extents_m']),attachment['half_extents_m']]))
        initial_bottom=float((vertices@anchor[:3,:3].T+anchor[:3,3])[:,2].min())
        if abs(initial_bottom-support['height_odom_m'])>attachment['uncertainty_m']+.004:return empty
        poses=[B@self.fk.links(q)['gripper_link']@mount for q,B in states]
        # Once the nominal object is wholly above its observed support plane,
        # lateral distance from the old contact point is not penetration. Bound
        # the entire articulated curve, not just its endpoints. This applies
        # only to the original source-labelled support samples; adjacent
        # furniture and every robot link still receive their normal checks.
        bottoms=[float((vertices@P[:3,:3].T+P[:3,3])[:,2].min()) for P in poses]
        above_support=all(z>support['height_odom_m'] for z in bottoms)
        for i in range(1,len(states)):
            error=self._articulated_curve_bound(states[i-1],states[i],'carried_object',vertices,np.linalg.norm(mount[:3,3]))
            above_support=above_support and min(bottoms[i-1],bottoms[i])-error>support['height_odom_m']
        if above_support:
            return owned_surface_mask(records,dict(surface_memberships=support['memberships']))
        distances=[float(np.linalg.norm(P[:2,3]-anchor[:2,3])) for P in poses]
        # Contact can perturb the measured entry beyond the fixed anchor band.
        # Admit only a return into that SAME band, within the existing estimate
        # uncertainty. Do not shift the anchor or grant a new travel allowance.
        recovering=distances[0]>.004
        if recovering and (distances[0]>.004+attachment['uncertainty_m'] or distances[-1]>.004):return empty
        for i,P in enumerate(poses):
            error=0. if i==0 else self._articulated_curve_bound(states[i-1],states[i],'carried_object',vertices,np.linalg.norm(mount[:3,3]))
            angle=math.acos(float(np.clip((np.trace(anchor[:3,:3].T@P[:3,:3])-1)/2,-1,1)))
            lateral_ok=distances[i]+error<=.004
            if recovering and i==0:lateral_ok=True
            elif recovering and i and distances[i-1]>.004:
                # Curve deviation <=4*error*t*(1-t), so this endpoint
                # inequality bounds every intermediate radius by entry radius.
                lateral_ok=distances[i]+4*error<=distances[i-1]
            if (not lateral_ok or angle>.04
                    or (vertices@P[:3,:3].T+P[:3,3])[:,2].min()-error<initial_bottom-.004):return empty
        return owned_surface_mask(records,dict(surface_memberships=support['memberships']))

    def _near_contact_clear_mask(self, records, states, candidate, name):
        """Certify finger clearance to the candidate's local support samples.

        These samples are NOT deleted or treated as target-owned. For this
        stationary-base, near-target path only, check the calibrated finger
        hull and full articulated sweep with the original voxel uncertainty
        plus one millimetre (the raw depth encoding unit). Ordinary navigation,
        other links, and unlabelled furniture retain the normal planning margin.
        """
        points,radii,_=occupied_arrays(records);clear=np.zeros(len(points),bool)
        if (candidate is None or name not in FINGERS|{'gripper_link'} or name not in self.convex_planes
                or type(self.fk) is not RobotFK or 'support_plane' not in candidate):return clear
        if any(not np.array_equal(B,states[0][1]) for _,B in states[1:]):return clear
        support=candidate['support_plane'];anchor=transform(support['anchor_object_odom'])
        poses=[B@self.fk.links(q)[name] for q,B in states]
        # Both approach and departure can need source-bound near-contact
        # clearance. A lift ending outside the local region must not revoke
        # the geometric certificate for its initial separation from the table.
        # The complete path is still checked below; this grants no contact.
        if min(np.linalg.norm(P[:3,3]-anchor[:3,3]) for P in (poses[0],poses[-1]))>.15:return clear
        # Restrict the reduced planning buffer to source-labelled support under
        # this object. A nearby cabinet or an unlabelled point is never eligible.
        local=(points-anchor[:3,3])@anchor[:3,:3]
        eligible=(owned_surface_mask(records,candidate) if name=='gripper_link'
                  else owned_surface_mask(records,dict(surface_memberships=support['memberships'])))
        finger_reach=float(np.linalg.norm(self.convex_vertices[name],axis=1).max())
        eligible &= np.all(np.abs(local[:,:2])+radii[:,None]<=np.asarray(candidate['half_extents_m'])[:2]+finger_reach,axis=1)
        ids=np.flatnonzero(eligible)
        if not len(ids):return clear
        import copy
        from dataclasses import replace
        probe=copy.copy(self);probe.config=replace(self.config,margin_m=.001)
        bounds=self.geometry[name]['bounds'];p=points[ids];r=radii[ids]
        safe=np.ones(len(ids),bool)
        for P in poses:safe &= probe._body_gaps(p,r,P,bounds,name)>0
        for i in range(1,len(states)):
            safe &= ~probe._body_articulated_hits(p,r,states[i-1],states[i],poses[i-1],poses[i],bounds,name)
        clear[ids]=safe
        return clear

    def _check(self, states, control_step, scope, allow_separating, contact_region=None,
               head_observation_contract=False, attachment=None, target_contact=None,
               observed_arm_only=False):
        if isinstance(control_step, bool) or int(control_step) != control_step or control_step < 0:
            raise ValueError('invalid clearance control step')
        records, stale = self._records(int(control_step))
        boxes = [self._boxes(q, odom,attachment) for q, odom in states]
        if observed_arm_only:
            # A diagnostic occupied-evidence layer. Self masking at ingestion
            # still uses the complete robot. Fingers/palm, carried objects,
            # base and head require their own acceptance; no claim about them.
            names={'shoulder_pan_link','shoulder_lift_link','upperarm_roll_link',
                   'elbow_flex_link','forearm_roll_link','wrist_flex_link','wrist_roll_link'}
            boxes=[{k:v for k,v in b.items() if k in names} for b in boxes]
            if not boxes[0]:raise ValueError('Observed arm sweep requires calibrated arm links')
        initial = boxes[0]
        head_names = {'head_pan_link', 'head_tilt_link'}
        if head_observation_contract:
            # One shared gripper actuator cannot hold two slightly different
            # measured finger coordinates exactly. Permit only representational
            # scale balancing during an otherwise head-only observation. Keep
            # checking ALL observed conflicts below, and retain unknown status.
            fixed = [i for i in range(15) if i not in (4,6,13,14)]
            q0,b0=states[0]
            if all(np.array_equal(q[fixed],q0[fixed]) and np.array_equal(b,b0) for q,b in states):
                inv=np.linalg.inv(b0)
                shared_hold = _shared_finger_hold_path(states)
                for name in FINGERS & initial.keys():
                    p,bounds=initial[name]
                    if _float32_hold_resolution(inv@p,[inv@sample[name][0] for sample in boxes[1:]],bounds):
                        head_names.add(name)
                    elif (shared_hold and all(self._translation_segment(a,b,name)
                                              for a,b in zip(states,states[1:]))):
                        # The existing entry exemption already reserves this
                        # margin envelope. A shared-actuator hold may use it,
                        # but may not leave it or mint allowance per subsegment.
                        # All occupied-evidence checks below still use the full
                        # margin. This remains an UNKNOWN observation exception,
                        # not a free-space or dynamics-safety certificate.
                        corners = _corners(bounds)
                        local = [((corners@s[name][0][:3,:3].T+s[name][0][:3,3])-p[:3,3])@p[:3,:3]
                                 for s in boxes[1:]]
                        if all(np.all(c >= bounds[0]-self.config.margin_m)
                               and np.all(c <= bounds[1]+self.config.margin_m) for c in local):
                            head_names.add(name)
        if not records:
            moving = {name for sample in boxes[1:] for name in initial
                      if not np.allclose(sample[name][0], initial[name][0], atol=1e-10, rtol=0)}
            result = self._report('stale_evidence' if stale else 'coverage_insufficient', scope,
                                body=next(iter(sorted(moving)), ''), reason='sensor_clearance_refresh_required')
            if observed_arm_only:
                result.update(clearance_stop_required=False,clearance_complete=False)
                return result
            if not stale and head_observation_contract and moving and moving <= head_names:
                result.update(accepted=True, clearance_observation_exception=True,
                              reason='sensor_clearance_bounded_head_observation')
            return result
        observed_step = max(r['evidence']['control_step'] for r in records)
        points, radii, occupied = occupied_arrays(records)
        attached_surface=owned_surface_mask(records,attachment or target_contact)
        support_contact=self._supported_contact_mask(records,states,attachment)
        near_contact={name:self._near_contact_clear_mask(records,states,attachment or target_contact,name)
                      for name in FINGERS|{'gripper_link'} if name in initial}
        groups = self._free_groups(records)
        support_planes = [r['support_plane_odom_z_m'] for r in records
                          if r['status']=='free' and 'support_plane_odom_z_m' in r]
        support_plane = None
        if support_planes and max(support_planes)-min(support_planes) <= 2*self.config.support_plane_tolerance_m:
            support_plane = max(support_planes)
        # Pad sampled FK segments by the greatest corresponding corner change.
        # This is a finite proposed-path check, not a proof of servo dynamics.
        swept_padding = [{name: 0. for name in initial}]
        translations = [{}]+[{name: self._translation_segment(a, b, name) for name in initial}
                            for a, b in zip(states, states[1:])]
        for before, after in zip(boxes, boxes[1:]):
            distance = {}
            for name in initial:
                p, b = before[name]; n, _ = after[name]
                c = _corners(b)
                distance[name] = float(np.linalg.norm(c @ p[:3, :3].T + p[:3, 3] - c @ n[:3, :3].T - n[:3, 3], axis=1).max())
            swept_padding.append(distance)
        initial_distance = {}
        last_distance = {}
        separating = False
        worst = (float('inf'), '')
        for name, (pose, bounds) in initial.items():
            initial_distance[name] = self._body_gaps(points,radii,pose,bounds,name)
            last_distance[name] = initial_distance[name]
        unknown_bodies = set()
        first_rejection_body = None
        coverage_tasks = []
        for index, state_boxes in enumerate(boxes[1:], 1):
            for name, (pose, bounds) in state_boxes.items():
                initial_pose, initial_bounds = initial[name]
                moved = not np.allclose(pose, initial_pose, atol=1e-10, rtol=0)
                if not moved:
                    continue
                gaps = self._body_gaps(points,radii,pose,bounds,name,swept_padding[index][name])
                # The carried candidate must not fall back to the legacy
                # spatial contact corridor: furniture inside that corridor
                # still has no target ownership. Preserve legacy profiles.
                permitted_contact = (np.zeros(len(points),bool) if attachment is not None or target_contact is not None
                    else self._contact_mask(points, name, contact_region, radii))
                if name=='carried_object' or name in FINGERS:
                    # Only source-labelled target surface voxels may contact
                    # fingers/the attached object. Other robot links must still
                    # avoid the target, and all unlabelled furniture is retained.
                    permitted_contact |= attached_surface
                if name=='carried_object':permitted_contact |= support_contact
                if name in near_contact:permitted_contact |= near_contact[name]
                active = ~permitted_contact
                # Expected calibrated wheel/support contact is not a furniture
                # exception. Only points whose entire voxel lies in a narrow
                # calibrated ground band may be excluded, for ground bodies.
                plane = self.config.support_plane_odom_z_m
                if plane is not None and (name == 'base_link' or 'wheel' in name):
                    active &= np.abs(points[:, 2] - plane) + radii > self.config.support_plane_tolerance_m
                if active.any():
                    i = np.flatnonzero(active)[np.argmin(gaps[active])]
                    worst = min(worst, (float(gaps[i]), name))
                starting_overlap = initial_distance[name] <= 0
                # Equal endpoint distances do not exclude a collision along a
                # rotation arc (the closest approach can be at its midpoint).
                approaching_sweep = gaps <= swept_padding[index][name]
                if name=='base_link' and name in self.convex_planes and not translations[index][name]:
                    approaching_sweep=self._body_se2_hits(points,radii,boxes[index-1][name][0],pose,name)
                elif translations[index][name]:
                    approaching_sweep = self._body_translation_hits(points, radii,
                        boxes[index-1][name][0], pose, bounds, name)
                elif type(self.fk) is RobotFK and (name in self.convex_planes or name=='carried_object'):
                    approaching_sweep=self._body_articulated_hits(points,radii,states[index-1],states[index],
                        boxes[index-1][name][0],pose,bounds,name,attachment)
                newly_colliding = ((gaps <= 0) | approaching_sweep) & ~starting_overlap
                worsening = (gaps <= 0) & (gaps < last_distance[name] - 1e-6) & starting_overlap
                violation = active & (newly_colliding | worsening | ((gaps <= 0) & (not allow_separating)))
                if violation.any():
                    i = np.flatnonzero(violation)[np.argmin(gaps[violation])]
                    if (translations[index][name] or name in self.convex_planes) and newly_colliding[i]:
                        # A slab intersection establishes conflict, not exact
                        # Euclidean penetration depth. Zero is the conservative
                        # contact bound if both endpoint gaps remain positive.
                        effective_gap = min(0., float(gaps[i]), float(last_distance[name][i]))
                    else:
                        effective_gap = gaps[i]-swept_padding[index][name] if newly_colliding[i] else gaps[i]
                    return self._report('observed_conflict', scope, body=name, clearance=effective_gap,
                        step=occupied[i]['evidence']['control_step'], reason='sensor_clearance_observed_conflict')
                if index == len(boxes) - 1 and (active & starting_overlap).any():
                    unresolved = active & starting_overlap & (gaps <= initial_distance[name] + 1e-6)
                    if unresolved.any():
                        return self._report('observed_conflict', scope, body=name, clearance=float(gaps[unresolved].min()),
                                            step=observed_step, reason='sensor_clearance_not_separating')
                    separating = True
                last_distance[name] = gaps
                coverage_tasks.append((name, pose, bounds, swept_padding[index][name], index))
        if observed_arm_only:
            # Absence of an observed conflict is NOT proof that the path is free.
            # Keep unknown explicit; the caller may not call this safe admission.
            result=self._report('coverage_not_evaluated',scope,body=worst[1],
                clearance=worst[0],step=observed_step,reason='sensor_clearance_no_observed_arm_conflict')
            result.update(clearance_stop_required=False,clearance_complete=False)
            return result
        # Scan occupied evidence over the complete path first, so an earlier
        # unknown patch never hides a later observed conflict. Coverage then
        # needs only one counterexample for rejection, not repeated expensive
        # clipping of the same unknown body at every proposed sample.
        for name, pose, bounds, padding, index in coverage_tasks:
            if name in unknown_bodies:
                continue
            support = support_plane if (name=='base_link' or 'wheel' in name) else None
            if translations[index][name]:
                cells = self._translation_cells(boxes[index-1][name][0], pose, bounds, support)
            else:
                cells = self._cells(pose, bounds, self.config.margin_m + padding, support)
            # Space occupied by the robot at command entry is not required
            # to be visible through the robot. Newly entered cells are.
            entry_boxes = [(p0, b0, self.config.margin_m) for p0, b0 in initial.values()]
            if name in FINGERS and contact_region is not None:
                self._contact_mask(np.empty((0, 3)), name, contact_region)
                contact_pose = np.eye(4)
                contact_pose[:3, :3] = np.asarray(contact_region['rotation_odom'], float)
                contact_pose[:3, 3] = vector(contact_region['center_odom_m'], 3)
                extents = vector(contact_region['half_extents_m'], 3)
                entry_boxes.append((contact_pose, np.array([-extents, extents]), 0.))
            diagnostic_start = len(self.coverage_diagnostics) if self.coverage_diagnostics is not None else 0
            covered = self._covered_union(cells, groups, entry_boxes) if len(cells) else np.ones(0, bool)
            if self.coverage_diagnostics is not None:
                for row in self.coverage_diagnostics[diagnostic_start:]:
                    row.update(body=name, segment_index=index,
                        sweep_mode='certified_translation_prism' if translations[index][name] else 'rotational_padding_fallback',
                        previous_pose_odom=boxes[index-1][name][0].tolist(),
                        next_pose_odom=pose.tolist(), link_bounds_m=bounds.tolist(),
                        configured_margin_m=self.config.margin_m,
                        original_isotropic_padding_m=padding)
            if not covered.all():
                unknown_bodies.add(name)
                if not head_observation_contract or name not in head_names:
                    if first_rejection_body is None:
                        first_rejection_body = name
                    # Opt-in CPU diagnostics inspect later bodies too. Normal
                    # admission keeps its original short circuit and report.
                    if self.coverage_diagnostics is None:
                        break
        if unknown_bodies:
            result = self._report('coverage_insufficient', scope,
                body=first_rejection_body or next(iter(sorted(unknown_bodies))),
                clearance=worst[0], step=observed_step, reason='sensor_clearance_refresh_required')
            if head_observation_contract and unknown_bodies <= head_names:
                result.update(accepted=True, clearance_observation_exception=True,
                              reason='sensor_clearance_bounded_head_observation')
            return result
        return self._report('no_observed_conflict', scope, body=worst[1], clearance=worst[0],
            step=observed_step, reason='sensor_clearance_separating' if separating else 'sensor_clearance_observed_path', accepted=True)

    def check_base(self, qpos, odometry, turn_rad, forward_m, control_step, *,
                   allow_separating=True, scope='base_turn_then_forward', contact_region=None,
                   attachment_candidate=None, report_only=False):
        """Sweeps all links: turn in place, then forward at the requested yaw."""
        q = vector(qpos, 15)
        base = odometry_transform(odometry)
        if not np.isfinite([turn_rad, forward_m]).all():
            raise ValueError('nonfinite base command')
        states = [(q, base)]
        n = max(1, int(math.ceil(abs(turn_rad) / self.config.path_rotation_rad)))
        for angle in np.linspace(0., turn_rad, n+1)[1:]:
            relative = odometry_transform(dict(dx_m=0., dy_m=0., dyaw_rad=float(angle)))
            states.append((q, base @ relative))
        final_turn = states[-1][1]
        n = max(1, int(math.ceil(abs(forward_m) / self.config.path_translation_m)))
        for distance in np.linspace(0., forward_m, n+1)[1:]:
            relative = odometry_transform(dict(dx_m=float(distance), dy_m=0., dyaw_rad=0.))
            states.append((q, final_turn @ relative))
        if report_only:
            return self._joint_path_report(states, control_step)
        return self._check(states, control_step, scope, allow_separating, contact_region,
                           attachment=self._attachment_for(attachment_candidate))

    def _attachment_for(self, candidate):
        if candidate is None:
            return None
        attachment=self.memory.attachment()
        if (attachment is None or attachment['candidate_id']!=candidate
                or attachment['status'] not in ('estimated','possibly_attached')):
            raise ValueError('Missing candidate-bound estimated attachment')
        return attachment

    def check_joint_path(self, qpath, odometry, control_step, *, scope='proposed_joint_path',
                         allow_separating=True, contact_region=None, allow_observation_unknown=False,
                         attachment_candidate=None, target_contact_candidate=None,
                         observed_arm_only=False, report_only=False):
        """Interpolated proposed joints only; not an assertion about later IK."""
        attachment=self._attachment_for(attachment_candidate)
        target_contact=None
        if target_contact_candidate is not None:
            target_contact=self.memory.contact_attachment()
            if target_contact is None or target_contact['candidate_id']!=target_contact_candidate:
                raise ValueError('Current confirmed target surface contact required')
        qpath = [vector(q, 15) for q in qpath]
        if not qpath:
            raise ValueError('joint path requires measured start')
        raw_odom = None if isinstance(odometry, dict) else np.asarray(odometry, float)
        if raw_odom is not None and raw_odom.ndim == 3:
            if raw_odom.shape != (len(qpath), 4, 4):
                raise ValueError('each joint node requires synchronized relative odometry')
            odoms = [odometry_transform(p) for p in raw_odom]
        else:
            odoms = [odometry_transform(odometry)] * len(qpath)
        states = [(qpath[0], odoms[0])]
        for a, b, oa, ob in zip(qpath, qpath[1:], odoms, odoms[1:]):
            if not np.array_equal(a[:3], b[:3]):
                raise ValueError('base motion must use relative odometry sweep')
            ya, yb = math.atan2(oa[1, 0], oa[0, 0]), math.atan2(ob[1, 0], ob[0, 0])
            dyaw = math.atan2(math.sin(yb-ya), math.cos(yb-ya))
            if not np.allclose(oa[2], [0, 0, 1, 0]) or not np.allclose(ob[2], [0, 0, 1, 0]):
                raise ValueError('moving odometry path must be planar SE2')
            n = max(1, int(math.ceil(np.max(np.abs(b[3:] - a[3:])) / self.config.path_joint_delta)),
                    int(math.ceil(np.linalg.norm(ob[:2, 3]-oa[:2, 3]) / self.config.path_translation_m)),
                    int(math.ceil(abs(dyaw) / self.config.path_rotation_rad)))
            for t in range(1, n+1):
                f = t/n
                # A stationary base must remain bit-identical. Re-encoding
                # its rotation through atan2/cos invents tiny base movement,
                # invalidating both the head-only and affine-link contracts.
                p = (oa.copy() if np.array_equal(oa,ob) else
                     odometry_transform(dict(dx_m=float((1-f)*oa[0, 3]+f*ob[0, 3]),
                     dy_m=float((1-f)*oa[1, 3]+f*ob[1, 3]), dyaw_rad=ya+f*dyaw)))
                states.append((a + (b-a)*f, p))
        if report_only:
            return self._joint_path_report(states, control_step)
        joints = np.stack(qpath)
        head_contract = (allow_observation_unknown
                         and np.max(np.ptp(joints[:, [4, 6]], axis=0)) <= .10+1e-9
                         and all(np.array_equal(p, odoms[0]) for p in odoms))
        return self._check(states, control_step, scope, allow_separating, contact_region,
                           head_observation_contract=head_contract,attachment=attachment,target_contact=target_contact,
                           observed_arm_only=observed_arm_only)

    def _joint_path_report(self, states, control_step):
        """Complete advisory scan; unknown cells never become free space.

        Distances include sensor voxel radius and the configured robot margin.
        This reports conservative link envelopes, not mesh penetration depth.
        Segment indices refer to this interpolated path (zero-based).
        """
        if self.config.s11_report:
            return self._s11_joint_path_report(states, control_step)
        records, stale = self._records(control_step)
        points,radii,_ = occupied_arrays(records)
        groups=self._free_groups(records)
        boxes=[self._boxes(q,b) for q,b in states]
        inverse=np.linalg.inv(states[0][1]);rows=[]
        # Exact broad phase only: a sphere enclosing the link AABB provides a
        # lower bound on AABB gaps. Include every convex-refinement candidate
        # as well: its halfspace score is not Euclidean distance outside a hull.
        # Refine all points that can tie or beat an evaluated upper bound.
        from scipy.spatial import cKDTree
        tree=cKDTree(points) if len(points) else None
        max_radius=float(np.max(radii)) if len(points) else 0.
        for name in boxes[0]:
            best=None;count=unknown=0;previous_key=None;cached=None
            for i,body in enumerate(boxes):
                p,bounds=body[name]
                if i:
                    old=boxes[i-1][name][0];c=_corners(bounds)
                    padding=float(np.linalg.norm(c@p[:3,:3].T+p[:3,3]-c@old[:3,:3].T-old[:3,3],axis=1).max())
                else:padding=0.
                key=(p.tobytes(),np.asarray(bounds).tobytes(),padding)
                if key!=previous_key:
                    j=None;distance=None
                    if tree is not None:
                        center=np.mean(bounds,axis=0)@p[:3,:3].T+p[:3,3]
                        radius=float(np.linalg.norm((bounds[1]-bounds[0])/2))
                        seed=int(tree.query(center)[1])
                        upper=float(self._body_gaps(points[seed:seed+1],radii[seed:seed+1],p,bounds,name,padding)[0])
                        cutoff=max(upper,padding) if name in self.convex_planes else upper
                        reach=max(0.,radius+max_radius+self.config.margin_m+cutoff)+1e-10
                        ids=np.asarray(sorted(tree.query_ball_point(center,reach)),dtype=int)
                        if not len(ids):ids=np.array([seed])
                        gaps=self._body_gaps(points[ids],radii[ids],p,bounds,name,padding)
                        winner=int(np.argmin(gaps));j=int(ids[winner]);distance=float(gaps[winner])
                    cells=self._cells(p,bounds,self.config.margin_m+padding,None)
                    covered=self._covered(cells,groups) if len(cells) else np.ones(0,bool)
                    cached=(j,distance,len(cells),int((~covered).sum()))
                    previous_key=key
                j,distance,ncells,nunknown=cached
                if j is not None and (best is None or distance<best[0]):
                    best=(distance,max(0,i-1),points[j])
                # Report-only coverage does not exempt the entry robot volume
                # or perform expensive convex subtraction. A whole cell must
                # have sensor free-space evidence; partial cells stay unknown.
                count+=ncells;unknown+=nunknown
            fraction=unknown/count if count else (1. if not records else 0.)
            rows.append(dict(link=name,min_clearance_m=None if best is None else best[0],
                segment_index=None if best is None else best[1],
                nearest_point_base_m=None if best is None else (inverse[:3,:3]@best[2]+inverse[:3,3]).tolist(),
                unobserved_fraction=float(fraction),
                status='unknown' if stale or fraction>0 or best is None else 'observed'))
        return dict(accepted=True,readonly=True,reason='clearance_report',steps_used=0,
            path_clearance=dict(frame='command_entry_base_link',links=rows,
                path_samples=len(states),tracking_error='unknown',stale_evidence=bool(stale),
                coverage_basis='fraction_of_swept_cells_without_complete_sensor_coverage_upper_bound'))

    def _s11_covered_union(self, cells, groups, entry_boxes):
        """Same convex subtraction as _covered_union, with batched fragments.

        Disjoint entry boxes cannot alter a cell. Coverage of all fragments is
        still required; failed or excessive splits remain unknown. This avoids
        repeating a complete camera-group scan for each individual fragment.
        """
        covered=self._covered(cells,groups)
        # An exterior vertex with no free-view support is an exact witness of
        # unknown space. Subtracting entry bodies cannot remove that vertex.
        # This only short-circuits unknown; it never certifies free space.
        vertices=cells.reshape(-1,3);owned=np.zeros(len(vertices),bool)
        for pose,bounds,margin in entry_boxes:
            owned|=_inside(vertices,pose,bounds,margin+1e-8)
        observed=np.zeros(len(vertices),bool)
        for group in groups:
            indices=np.flatnonzero(~observed & ~owned)
            for chunk in np.array_split(indices,max(1,math.ceil(len(indices)/128))):
                if not len(chunk):continue
                optical=(vertices[chunk]-group['origin'])@group['R'];z=optical[:,2]
                projected=optical@group['K'].T
                uv=projected[:,:2]/np.maximum(z[:,None],1e-12)
                bounds=group['bounds'];eps=1e-8
                hit=((z[:,None]>0)&(z[:,None]>=group['near']-eps)&
                    (z[:,None]<=group['far']+eps)&
                    (uv[:,0,None]>=bounds[:,0]-eps)&(uv[:,0,None]<=bounds[:,2]+eps)&
                    (uv[:,1,None]>=bounds[:,1]-eps)&(uv[:,1,None]<=bounds[:,3]+eps)&
                    (vertices[chunk,2,None]>=group['support_plane']-eps))
                observed[chunk]|=hit.any(axis=1)
        unknown_witness=(~owned & ~observed).reshape(cells.shape[:2]).any(axis=1)
        buckets={}
        for index in np.flatnonzero(~covered):
            if unknown_witness[index]:continue
            cell=cells[index];relevant=[]
            for pose,bounds,margin in entry_boxes:
                local=(cell-pose[:3,3])@pose[:3,:3]
                if not (np.any(local.max(0)<bounds[0]-margin-1e-10) or
                        np.any(local.min(0)>bounds[1]+margin+1e-10)):
                    relevant.append((pose,bounds,margin))
            if not relevant:continue
            pieces=[cell]
            for pose,bounds,margin in relevant:
                next_pieces=[]
                for piece in pieces:
                    outside=_subtract_box(piece,pose,bounds,margin)
                    if outside is None:
                        next_pieces=None;break
                    next_pieces.extend(outside)
                    if len(next_pieces)>128:
                        next_pieces=None;break
                pieces=next_pieces
                if pieces is None or not pieces:break
            if pieces is None:continue
            covered[index]=True
            for piece in pieces:buckets.setdefault(len(piece),[]).append((index,piece))
        for bucket in buckets.values():
            answers=self._covered(np.stack([p for _,p in bucket]),groups)
            for (index,_),answer in zip(bucket,answers):covered[index]&=answer
        return covered

    def _s11_joint_path_report(self, states, control_step):
        """Advisory sensor geometry with explicit low surfaces and local evidence.

        A near voxel whose lower grid face is below the pinned chassis underside
        is a low surface for chassis/wheels only. Its height bound is the robot
        collision envelope, not a fitted scene plane or an observed success label.
        The half-voxel correction handles grid quantization; far coarse voxels
        are never exempted. Arm/head checks retain all observed points.
        """
        records, old = self._records(control_step)
        points, radii, _ = occupied_arrays(records)
        groups = self._free_groups(records)
        boxes = [self._boxes(q, b) for q, b in states]
        inverse = np.linalg.inv(states[0][1])
        entry = [(p,b,0.) for p,b in boxes[0].values()]
        underside = float(self.geometry['base_link']['bounds'][0,2])
        local = points@inverse[:3,:3].T+inverse[:3,3] if len(points) else points
        half_voxel = radii/math.sqrt(3.)
        low = ((local[:,2]-half_voxel <= underside) &
               (half_voxel <= self.config.voxel_m/2+1e-12)) if len(points) else np.zeros(0,bool)
        # The union of swept calibrated link AABBs is a conservative route bound.
        swept=[]
        for body in boxes:
            for pose,bounds in body.values():
                vertices=_corners(bounds)@pose[:3,:3].T+pose[:3,3]
                swept.append((vertices.min(0)-self.config.margin_m,vertices.max(0)+self.config.margin_m))
        def relevant(record):
            if record.get('kind')=='voxel_cloud':
                p=np.asarray(record['centers_odom_m'],float);rad=np.asarray(record['radii_m'],float)
                return any(np.any(np.all(p+rad[:,None]>=lo,axis=1)&np.all(p-rad[:,None]<=hi,axis=1)) for lo,hi in swept)
            if record.get('kind')=='frustum':
                u0,v0,u1,v1=record['pixel_bounds'];K=np.asarray(record['intrinsics']);R=np.asarray(record['rotation_odom_from_optical']);o=np.asarray(record['origin_odom_m'])
                rays=np.array([[u,v,1.] for u in (u0,u1) for v in (v0,v1)])@np.linalg.inv(K).T
                p=np.concatenate([rays*z for z in (record['near_m'],record['far_m'])])@R.T+o
                lo,hi=p.min(0),p.max(0)
                return any(np.all(hi>=a)&np.all(lo<=b) for a,b in swept)
            return True  # Unrecognized old geometry cannot silently certify freshness.
        stale=any(relevant(record) for record in old)
        rows=[]
        from scipy.spatial import cKDTree
        for name in boxes[0]:
            chassis=name=='base_link' or 'wheel' in name
            ids=np.flatnonzero(~low if chassis else np.ones(len(points),bool))
            pnts,rs=points[ids],radii[ids]
            tree=cKDTree(pnts) if len(ids) else None
            max_radius=float(rs.max()) if len(ids) else 0.
            best=None;count=unknown=0;previous=None;cached=None
            for i,body in enumerate(boxes):
                pose,bounds=body[name]
                padding=0. if not i else float(np.linalg.norm(_corners(bounds)@pose[:3,:3].T+pose[:3,3]-_corners(bounds)@boxes[i-1][name][0][:3,:3].T-boxes[i-1][name][0][:3,3],axis=1).max())
                key=(pose.tobytes(),np.asarray(bounds).tobytes(),padding)
                if key!=previous:
                    j=distance=None
                    if tree is not None:
                        center=np.mean(bounds,axis=0)@pose[:3,:3].T+pose[:3,3]
                        seed=int(tree.query(center)[1]);upper=float(self._body_gaps(pnts[seed:seed+1],rs[seed:seed+1],pose,bounds,name,padding)[0])
                        cutoff=max(upper,padding) if name in self.convex_planes else upper
                        reach=max(0.,float(np.linalg.norm((bounds[1]-bounds[0])/2))+max_radius+self.config.margin_m+cutoff)+1e-10
                        near=np.asarray(sorted(tree.query_ball_point(center,reach)),dtype=int)
                        if not len(near):near=np.array([seed])
                        gaps=self._body_gaps(pnts[near],rs[near],pose,bounds,name,padding)
                        winner=int(np.argmin(gaps));j=int(near[winner]);distance=float(gaps[winner])
                    cells=self._cells(pose,bounds,self.config.margin_m+padding,None)
                    own=np.zeros(len(cells),bool)
                    for ep,eb,_ in entry:own|=_inside(cells,ep,eb,0.).all(axis=1)
                    cells=cells[~own]
                    covered=self._s11_covered_union(cells,groups,entry) if len(cells) else np.ones(0,bool)
                    cached=(j,distance,len(cells),int((~covered).sum()));previous=key
                j,distance,ncells,nunknown=cached
                if j is not None and (best is None or distance<best[0]):best=(distance,max(0,i-1),pnts[j])
                count+=ncells;unknown+=nunknown
            fraction=unknown/count if count else 0.
            nearest=None if best is None else (inverse[:3,:3]@best[2]+inverse[:3,3]).tolist()
            rows.append(dict(link=name,min_clearance_m=None if best is None else best[0],segment_index=None if best is None else best[1],nearest_point_base_m=nearest,unobserved_fraction=float(fraction),status='unknown' if stale or fraction>0 or best is None else 'observed'))
        return dict(accepted=True,readonly=True,reason='clearance_report',steps_used=0,
            path_clearance=dict(frame='command_entry_base_link',links=rows,path_samples=len(states),
                tracking_error='unknown',stale_evidence=stale,
                coverage_basis='swept_cells_outside_measured_entry_robot_volume',
                low_surface=dict(count=int(low.sum()),height_limit_base_m=underside,
                    source='pinned_base_link_collision_bounds_min_z',source_sha256=self.geometry['base_link']['sha256'],
                    quantization_rule='near_voxel_lower_grid_face_at_or_below_chassis_underside',
                    excluded_links=['base_link','r_wheel_link','l_wheel_link'],not_a_traversability_certificate=True)))

    def check_observed_arm_path(self, qpath, odometry, control_step):
        """Diagnostic stop evidence only; unknown never becomes free or a veto.

        No attachment/contact exemptions: observed furniture is not filtered.
        Prediction and braking bounds belong to the caller, not this geometry test.
        """
        report=self.check_joint_path(qpath,odometry,control_step,scope='observed_rest_arm_sweep',
                                     observed_arm_only=True)
        report.update(clearance_stop_required=report['clearance_status']=='observed_conflict',
                      clearance_complete=False)
        return report

    def check_lookahead(self, qpos, qvel, odometry, control_step, dt=.15):
        if not math.isfinite(dt) or not 0 < dt <= .25:
            raise ValueError('short finite clearance lookahead required')
        q, velocity = vector(qpos, 15), vector(qvel, 15)
        future = q + dt * velocity
        future[:3] = q[:3]
        return self.check_joint_path([q, future], odometry, control_step, scope='measured_joint_lookahead')
