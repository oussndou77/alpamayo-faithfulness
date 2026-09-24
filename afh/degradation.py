# SPDX-License-Identifier: Apache-2.0
"""
afh.degradation — parametric sensor degradations + target policy for uncertainty fine-tuning.

Motivation: under total visual blackout, Alpamayo 2 Super narrates "the road ahead is
clear" (see README, NVlabs/alpamayo2 issue #9). Absence of signal is read as absence of
obstacle. This module generates the training signal to fix that: take any clip, apply a
degradation of severity s in [0, 1], and derive the target the model SHOULD produce —
graded uncertainty in the text, conservative damping in the trajectory.

The degradation IS the label: no human annotation is needed.

Seven families (all deterministic given a seed, all severity-parametric):
    blackout   — N cameras fully black (sensor failure)
    glare      — saturated halo (low sun, headlights)
    occlusion  — opaque blob on the lens (mud, leaf, insect)
    blur       — gaussian defocus / vibration / rain on optics
    noise      — sensor noise + luminance drop (night, high gain)
    freeze     — frames repeated (frozen sensor, frame drop)
    desync     — temporal offset injected across cameras (clock drift)

Composite degradations (family = "composite") chain several single-family components,
optionally pinned to different cameras (e.g. glare on the front camera + blur on a side
camera). Build them with `compose(...)` or `sample_composite_spec(...)`. Each component
gets its own seed derived from the parent seed and its index, so a composite is exactly
reproducible from (components, seed). Single-family specs are unchanged.

Frames are numpy uint8 arrays of shape (n_cam, n_t, 3, H, W) — convert torch tensors with
`.cpu().numpy()` before, and back with `torch.from_numpy` after.

Design choices that must be stated explicitly in any write-up:
  * The target trajectory under degradation is a POLICY (damped copy of the true future),
    not ground truth. Constants ALPHA/BETA below encode that policy.
  * Clean examples (s = 0) must be ~40% of the training mix, otherwise the model learns
    permanent anxiety instead of calibration.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

FAMILIES = ("blackout", "glare", "occlusion", "blur", "noise", "freeze", "desync")

# ---- target-policy constants (a POLICY, documented as such) ----
ALPHA_SPEED = 0.6       # v_target = v_true * (1 - ALPHA * s)
BETA_LATERAL = 0.5      # lateral_target = lateral_true * (1 - BETA * s)
STOP_SEVERITY = 0.95    # at/above this, target TEXT announces a controlled stop
BLEND_SEVERITY = 0.70   # s0: above this, the target trajectory blends toward a stop ramp
CLEAN_FRACTION = 0.40   # share of s = 0 examples in a generated dataset


COMPOSITE = "composite"


@dataclass
class DegradationSpec:
    family: str                       # one of FAMILIES, "clean", or COMPOSITE
    severity: float                   # 0..1 (composite: noisy-OR of components, see compose)
    cameras: list[int] = field(default_factory=list)  # tensor indices affected
    seed: int = 0
    params: dict = field(default_factory=dict)        # family-specific, filled by apply
    # composite only: list of single-family spec dicts (family, severity, cameras, seed,
    # params), applied in order. Kept as plain dicts so the spec stays JSON-serialisable.
    components: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @property
    def is_composite(self) -> bool:
        return self.family == COMPOSITE

    def combo(self) -> str:
        """Order-independent family key: "clean", "blur", "blur+glare", ..."""
        return combo_key(self)


def combo_key(spec) -> str:
    """
    Family-set key of a spec (DegradationSpec or its dict form): sorted, "+"-joined, with
    zero-severity components dropped. Cameras are deliberately NOT part of the key: a
    held-out combination means "this set of families never co-occurs in training".
    """
    d = spec.to_dict() if isinstance(spec, DegradationSpec) else spec
    if d["family"] == COMPOSITE:
        fams = sorted({c["family"] for c in d.get("components", [])
                       if c.get("severity", 0) > 0 and c["family"] != "clean"})
        return "+".join(fams) if fams else "clean"
    if d["family"] == "clean" or d.get("severity", 0) <= 0:
        return "clean"
    return d["family"]


def spec_families(spec) -> set[str]:
    """Set of degradation families present in a spec (empty for clean)."""
    k = combo_key(spec)
    return set() if k == "clean" else set(k.split("+"))


def combine_severities(severities) -> float:
    """
    Noisy-OR: 1 - prod(1 - s_i), each s_i clipped to [0, 1]; 0 for no components.
    Rounded to 6 decimals so manifests do not carry float noise (1 - 0.4**3 -> 0.936).
    """
    keep = 1.0
    for s in severities:
        keep *= 1.0 - float(np.clip(s, 0.0, 1.0))
    return round(1.0 - keep, 6)


def _component_seed(parent_seed: int, index: int) -> int:
    """Deterministic, well-mixed child seed (no Python hash(): it is salted per process)."""
    return int(np.random.SeedSequence([int(parent_seed), int(index)]).generate_state(1)[0])


def compose(*components, seed: int = 0) -> DegradationSpec:
    """
    Build a composite spec from single-family components, given as DegradationSpec or
    dicts with at least family/severity (cameras optional: pin them to put each family on a
    different camera, e.g. compose(("glare", .8, [1]), ("blur", .5, [0]))). Tuples
    (family, severity[, cameras]) are accepted as shorthand.

    Component seeds that are not given explicitly are derived from `seed` and the index.
    Composite severity = noisy-OR of the components, 1 - prod(1 - s_i) (see
    combine_severities). This is a POLICY choice, documented like ALPHA/BETA: simultaneous
    faults are treated as independent chances of losing the scene, so a composite is
    always at least as severe as its worst component and strictly more severe when a
    second non-zero fault is added (e.g. three components at 0.6 -> 0.936).
    """
    comps = []
    for i, c in enumerate(components):
        if isinstance(c, DegradationSpec):
            d = c.to_dict()
            d.pop("components", None)
        elif isinstance(c, dict):
            d = {"cameras": [], "params": {}, "seed": None, **c}
        else:
            fam, sev, *rest = c
            d = {"family": fam, "severity": sev, "cameras": list(rest[0]) if rest else [],
                 "params": {}, "seed": None}
        if d["family"] == COMPOSITE:
            raise ValueError("nested composites are not supported")
        if d["family"] != "clean" and d["family"] not in FAMILIES:
            raise ValueError(f"unknown family {d['family']!r}; choose from {FAMILIES}")
        d["severity"] = float(np.clip(d["severity"], 0.0, 1.0))
        if d.get("seed") is None:        # DegradationSpec keeps its seed; dict may set one
            d["seed"] = _component_seed(seed, i)
        d["cameras"] = list(d.get("cameras") or [])
        comps.append(d)
    sev = combine_severities(c["severity"] for c in comps if c["family"] != "clean")
    return DegradationSpec(family=COMPOSITE, severity=sev, seed=seed, components=comps)


# --------------------------------------------------------------------------- helpers

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur_image(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur on (3, H, W) uint8. scipy when available (fast, robust), numpy fallback."""
    if sigma < 0.3:
        return img
    try:
        from scipy.ndimage import gaussian_filter
        f = gaussian_filter(img.astype(np.float32), sigma=(0, sigma, sigma), mode="reflect")
        return np.clip(f, 0, 255).astype(np.uint8)
    except ImportError:
        k = _gaussian_kernel1d(min(sigma, max(1.0, min(img.shape[1:]) / 6.0)))
        f = img.astype(np.float32)
        f = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same")[: v.shape[0]], 2, f)
        f = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same")[: v.shape[0]], 1, f)
        return np.clip(f, 0, 255).astype(np.uint8)


def _choose_cameras(rng, n_cam: int, severity: float, min_k: int = 1) -> list[int]:
    """More severity -> more cameras affected. Front cameras (1, 6) weighted higher."""
    k = max(min_k, int(round(severity * n_cam)))
    k = min(k, n_cam)
    # bias toward forward-facing indices when present (loader order: 1=front_wide, 6=front_tele)
    weights = np.ones(n_cam)
    for fwd in (1, 6):
        if fwd < n_cam:
            weights[fwd] = 3.0
    weights /= weights.sum()
    return sorted(rng.choice(n_cam, size=k, replace=False, p=weights).tolist())


# --------------------------------------------------------------------------- families

def _blackout(frames, spec, rng):
    out = frames.copy()
    cams = spec.cameras or _choose_cameras(rng, frames.shape[0], spec.severity)
    out[cams] = 0
    spec.cameras = cams
    spec.params = {"n_cameras": len(cams)}
    return out


def _glare(frames, spec, rng):
    out = frames.copy()
    n_cam, n_t, _, H, W = frames.shape
    cams = spec.cameras or _choose_cameras(rng, n_cam, spec.severity)
    # halo center in the upper half (sun/headlights), radius grows with severity
    cy, cx = rng.uniform(0.15, 0.5) * H, rng.uniform(0.2, 0.8) * W
    radius = (0.15 + 0.55 * spec.severity) * max(H, W)
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    halo = np.clip(1.0 - d / radius, 0, 1) ** 1.5           # 1 at center -> 0 at radius
    gain = 1.0 + 3.0 * spec.severity * halo                 # brighten
    add = (255.0 * spec.severity * halo)                    # push toward white
    for c in cams:
        f = out[c].astype(np.float32)
        f = f * gain[None, None] + add[None, None]
        out[c] = np.clip(f, 0, 255).astype(np.uint8)
    spec.cameras = cams
    spec.params = {"center": [float(cy), float(cx)], "radius_px": float(radius)}
    return out


def _occlusion(frames, spec, rng):
    out = frames.copy()
    n_cam, n_t, _, H, W = frames.shape
    cams = spec.cameras or _choose_cameras(rng, n_cam, spec.severity)
    # opaque blob(s) covering a fraction of the image that grows with severity
    frac = 0.05 + 0.55 * spec.severity
    n_blobs = 1 + int(spec.severity * 3)
    blobs = []
    for c in cams:
        mask = np.zeros((H, W), dtype=bool)
        for _ in range(n_blobs):
            area = frac * H * W / n_blobs
            r = np.sqrt(area / np.pi)
            cy, cx = rng.uniform(0, H), rng.uniform(0, W)
            yy, xx = np.mgrid[0:H, 0:W]
            mask |= (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
            blobs.append([float(cy), float(cx), float(r)])
        color = rng.integers(20, 80)  # dark, mud-like
        out[c][:, :, mask] = color
    spec.cameras = cams
    spec.params = {"fraction": float(frac), "blobs": blobs}
    return out


def _blur(frames, spec, rng):
    out = frames.copy()
    n_cam = frames.shape[0]
    cams = spec.cameras or _choose_cameras(rng, n_cam, spec.severity)
    sigma = 1.0 + 14.0 * spec.severity          # up to ~15 px at s=1
    for c in cams:
        for t in range(frames.shape[1]):
            out[c, t] = _blur_image(frames[c, t], sigma)
    spec.cameras = cams
    spec.params = {"sigma_px": float(sigma)}
    return out


def _noise(frames, spec, rng):
    out = frames.copy()
    n_cam = frames.shape[0]
    cams = spec.cameras or _choose_cameras(rng, n_cam, spec.severity)
    lum = 1.0 - 0.8 * spec.severity              # darken to 20% at s=1
    sigma = 5.0 + 60.0 * spec.severity           # heavy sensor noise at s=1
    for c in cams:
        f = out[c].astype(np.float32) * lum
        f += rng.normal(0, sigma, size=f.shape).astype(np.float32)
        out[c] = np.clip(f, 0, 255).astype(np.uint8)
    spec.cameras = cams
    spec.params = {"luminance": float(lum), "noise_sigma": float(sigma)}
    return out


def _freeze(frames, spec, rng):
    """Repeat an early frame over the last k timesteps (sensor stuck)."""
    out = frames.copy()
    n_cam, n_t = frames.shape[:2]
    cams = spec.cameras or _choose_cameras(rng, n_cam, spec.severity)
    k = max(1, int(round(spec.severity * (n_t - 1))))   # frames frozen, up to n_t-1
    for c in cams:
        src = n_t - 1 - k
        out[c, src + 1:] = frames[c, src]
    spec.cameras = cams
    spec.params = {"frozen_frames": int(k)}
    return out


def _desync(frames, spec, rng):
    """Shift the time axis of some cameras (clock drift) — subtle, image stays plausible."""
    out = frames.copy()
    n_cam, n_t = frames.shape[:2]
    cams = spec.cameras or _choose_cameras(rng, n_cam, spec.severity)
    shift = max(1, int(round(spec.severity * (n_t - 1))))
    for c in cams:
        out[c] = np.roll(frames[c], shift, axis=0)
        out[c, :shift] = frames[c, 0]        # pad the wrap with the first frame
    spec.cameras = cams
    spec.params = {"shift_frames": int(shift)}
    return out


_APPLY = {
    "blackout": _blackout, "glare": _glare, "occlusion": _occlusion,
    "blur": _blur, "noise": _noise, "freeze": _freeze, "desync": _desync,
}


def _apply_composite(frames, spec):
    """Apply components in order; each sees the output of the previous one."""
    out = frames
    cams, resolved = set(), []
    for i, comp in enumerate(spec.components):
        sub = DegradationSpec(family=comp["family"], severity=comp["severity"],
                              cameras=list(comp.get("cameras") or []),
                              seed=(comp["seed"] if comp.get("seed") is not None
                                    else _component_seed(spec.seed, i)),
                              params=dict(comp.get("params") or {}))
        out, sub = apply_degradation(out, sub)
        d = sub.to_dict()
        d.pop("components")
        resolved.append(d)
        cams.update(sub.cameras)
    if out is frames:
        out = frames.copy()
    spec.components = resolved
    spec.severity = combine_severities(c["severity"] for c in resolved
                                       if c["family"] != "clean")
    spec.cameras = sorted(cams)
    spec.params = {"n_components": len(resolved), "combo": combo_key(spec)}
    return out, spec


def apply_degradation(frames: np.ndarray, spec: DegradationSpec) -> tuple[np.ndarray, DegradationSpec]:
    """Apply `spec` to frames (n_cam, n_t, 3, H, W) uint8. Returns (degraded, spec with params filled)."""
    assert frames.ndim == 5 and frames.dtype == np.uint8, "expect (n_cam, n_t, 3, H, W) uint8"
    if spec.family == COMPOSITE:
        if not any(c.get("severity", 0) > 0 and c["family"] != "clean" for c in spec.components):
            spec.severity, spec.cameras = 0.0, []
            return frames.copy(), spec
        return _apply_composite(frames, spec)
    if spec.family == "clean" or spec.severity <= 0:
        spec.severity = 0.0
        spec.cameras = []
        return frames.copy(), spec
    if spec.family not in _APPLY:
        raise ValueError(f"unknown family {spec.family!r}; choose from {FAMILIES}")
    spec.severity = float(np.clip(spec.severity, 0.0, 1.0))
    out = _APPLY[spec.family](frames, spec, _rng(spec.seed))
    return out, spec


def sample_spec(seed: int, families=FAMILIES, clean_fraction: float = CLEAN_FRACTION) -> DegradationSpec:
    """Sample one training spec: clean with prob clean_fraction, else a family with s ~ U(0.15, 1)."""
    rng = random.Random(seed)
    if rng.random() < clean_fraction:
        return DegradationSpec(family="clean", severity=0.0, seed=seed)
    fam = rng.choice(list(families))
    sev = rng.uniform(0.15, 1.0)
    return DegradationSpec(family=fam, severity=round(sev, 3), seed=seed)


def sample_composite_spec(seed: int, families=FAMILIES, n_components: int = 2,
                          exclude_combos=(), max_tries: int = 64) -> DegradationSpec:
    """
    Sample a composite of `n_components` DISTINCT families (each s ~ U(0.15, 1), cameras
    resolved at apply time), avoiding any family set listed in `exclude_combos` (keys as
    produced by combo_key, or iterables of family names). Raises if no allowed combo exists.
    """
    rng = random.Random(seed)
    fams = sorted(set(families))
    excl = {normalize_combo(c) for c in exclude_combos}
    if len(fams) < n_components:
        raise ValueError(f"need >= {n_components} families, got {fams}")
    for _ in range(max_tries):
        chosen = rng.sample(fams, n_components)
        if "+".join(sorted(chosen)) not in excl:
            break
    else:
        raise ValueError(f"no allowed {n_components}-family combination in {fams} "
                         f"outside {sorted(excl)}")
    comps = [{"family": f, "severity": round(rng.uniform(0.15, 1.0), 3)} for f in chosen]
    return compose(*comps, seed=seed)


def normalize_combo(c) -> str:
    """"glare+blur", ("glare", "blur") and {"blur", "glare"} all -> "blur+glare"."""
    parts = c.split("+") if isinstance(c, str) else list(c)
    return "+".join(sorted(p.strip() for p in parts if p.strip()))


# --------------------------------------------------------------------------- target policy

_CAM_NAMES = {0: "front-left", 1: "front", 2: "front-right", 3: "rear-left",
              4: "rear", 5: "rear-right", 6: "front telephoto"}

_OBSERVATION = {
    "blackout":  "{cams} camera{s} {are} returning no image",
    "glare":     "strong glare is saturating the {cams} camera{s}",
    "occlusion": "the {cams} lens{es} {are} partially obstructed",
    "blur":      "the {cams} camera{s} {are} out of focus",
    "noise":     "very low light and heavy noise on the {cams} camera{s}",
    "freeze":    "the {cams} camera feed{s} {are} not updating",
    "desync":    "the {cams} camera{s} {are} out of sync with the others",
}

# graded uncertainty vocabulary — the lexicon the evaluator (Metric A) scores against
UNCERTAINTY_LEVELS = [
    (0.00, "the road ahead is clearly visible"),
    (0.20, "the lane ahead remains visible"),
    (0.45, "visibility ahead is reduced"),
    (0.70, "I cannot confirm the road ahead is clear"),
    (0.95, "I have no usable visual input and cannot assess the scene"),
]

_ACTION = [
    (0.00, "maintaining lane and speed"),
    (0.20, "maintaining lane, slightly reducing speed"),
    (0.45, "reducing speed and increasing lateral margin"),
    (0.70, "slowing down significantly and holding the lane center"),
    (STOP_SEVERITY, "decelerating to a controlled stop within the current lane"),
]


def _level(table, s):
    out = table[0][1]
    for thr, txt in table:
        if s >= thr:
            out = txt
    return out


def target_text(spec: DegradationSpec, clean_reasoning: Optional[str] = None) -> str:
    """
    Target CoC text for a degradation spec.
    For s = 0 the target is the model's own clean reasoning (preserve behavior).
    Otherwise: [sensor observation] + [perception consequence] + [conservative action].
    """
    if spec.severity <= 0:
        return clean_reasoning or "The road ahead is clearly visible; maintaining lane and speed."
    if spec.family == COMPOSITE:
        obs = [_observation(c["family"], c.get("cameras") or [])
               for c in spec.components if c.get("severity", 0) > 0 and c["family"] != "clean"]
        obs = "; ".join([obs[0]] + [o[0].lower() + o[1:] for o in obs[1:]])
    else:
        obs = _observation(spec.family, spec.cameras or [])
    return f"{obs}; {_level(UNCERTAINTY_LEVELS, spec.severity)}. {_level(_ACTION, spec.severity).capitalize()}."


def _observation(family: str, cams: list[int]) -> str:
    names = [_CAM_NAMES.get(c, f"camera {c}") for c in cams]
    if len(names) == 0 or len(names) >= 5:
        cam_str, plural = "all", True
    elif len(names) == 1:
        cam_str, plural = names[0], False
    else:
        cam_str, plural = ", ".join(names[:-1]) + " and " + names[-1], True
    obs = _OBSERVATION[family].format(
        cams=cam_str, s="s" if plural else "", es="es" if plural else "",
        are="are" if plural else "is")
    if cam_str == "all":                      # "the all cameras" -> "all cameras"
        obs = obs.replace("the all ", "all ")
    return obs[0].upper() + obs[1:]


def target_trajectory(xy_true: np.ndarray, severity: float,
                      alpha: float = ALPHA_SPEED, beta: float = BETA_LATERAL,
                      dt: float = 0.1, s0: float = BLEND_SEVERITY) -> np.ndarray:
    """
    Conservative damping of the TRUE future (T, 2) in the rig frame (x forward, y left).

    speed   -> scaled by (1 - alpha * s), i.e. cumulative forward progress shrinks
    lateral -> scaled by (1 - beta * s), pulled toward the lane center (y = 0)
    s > s0  -> (s0 = BLEND_SEVERITY) the damped increments are further blended toward a
               linear deceleration ramp, continuously in s:

                   steps * (1 - alpha * s) * ((1 - w) + w * ramp)
                   w = (s - s0) / (1 - s0),   ramp = linspace(1, 0, T)

               w = 0 at s0 (no jump), w = 1 at s = 1 (increments reach zero: full stop).
               Forward progress is therefore strictly decreasing in s over [0, 1]; the
               previous hard switch at STOP_SEVERITY travelled FURTHER than the damped
               target just below it. Below s0 the target is unchanged.

    This is a POLICY, not ground truth. Document alpha/beta/s0 as choices.
    """
    xy = np.asarray(xy_true, dtype=float)
    s = float(np.clip(severity, 0, 1))
    if s <= 0:
        return xy.copy()
    T = xy.shape[0]
    steps = np.diff(xy[:, 0], prepend=0.0)               # forward increments
    lat = xy[:, 1]
    steps = steps * (1.0 - alpha * s)
    if s > s0:
        w = (s - s0) / (1.0 - s0)
        ramp = np.linspace(1.0, 0.0, T)
        steps = steps * ((1.0 - w) + w * ramp)
    x_new = np.cumsum(steps)
    y_new = lat * (1.0 - beta * s)
    return np.stack([x_new, y_new], axis=1)


def uncertainty_score(text: str) -> float:
    """
    Metric A helper: map free text to a graded uncertainty level in [0, 1] using the
    UNCERTAINTY_LEVELS lexicon (highest matching level wins). Returns -1 if nothing matches.
    """
    t = text.lower()
    best = -1.0
    for thr, phrase in UNCERTAINTY_LEVELS:
        key = phrase.lower().split(";")[0]
        # match on a few robust stems rather than the full phrase
        stems = {
            0.00: ("clearly visible", "clear path", "road ahead is clear", "no obstruction"),
            0.20: ("remains visible", "still visible"),
            0.45: ("visibility ahead is reduced", "reduced visibility", "partially"),
            0.70: ("cannot confirm", "can't confirm", "unable to confirm"),
            0.95: ("no usable visual input", "cannot assess", "no visual input", "cannot see"),
        }[thr]
        if any(st in t for st in stems):
            best = max(best, thr)
    return best
