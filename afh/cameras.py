# SPDX-License-Identifier: Apache-2.0
"""
afh.cameras — camera identity for the PhysicalAI-AV loader, torch-free.

The loader returns frames as a tensor (n_cam, n_t, 3, H, W) whose camera order is given
by data["camera_indices"]: a position in the tensor is NOT a camera. Depending on the
clip and the model the tensor holds 4 cameras ([0, 1, 2, 6]), 6 (Alpamayo 2 Super:
[0, 1, 2, 3, 5, 6]) or all 7. Reasoning in tensor positions silently mislabels cameras:
in the 4-camera tensor, position 3 is front_tele, not rear_left.

CAM_INDEX_TO_ID is the loader's camera index -> camera_id map. It must stay identical to
the one runners/run_counterfactual_a2.py uses in occlude_frames (that runner imports torch,
so afh cannot import it; tests/test_uncertainty_cold.py checks the two never diverge).
"""

from __future__ import annotations

from typing import Iterable, Optional

CAM_INDEX_TO_ID = {
    0: "camera_cross_left_120fov", 1: "camera_front_wide_120fov",
    2: "camera_cross_right_120fov", 3: "camera_rear_left_70fov",
    4: "camera_rear_tele_30fov", 5: "camera_rear_right_70fov",
    6: "camera_front_tele_30fov",
}

FRONT_CAMERA_IDS = ("camera_front_wide_120fov", "camera_front_tele_30fov")

# wording used in target texts (unchanged from the former index-keyed table)
CAM_ID_TO_NAME = {
    "camera_cross_left_120fov": "front-left", "camera_front_wide_120fov": "front",
    "camera_cross_right_120fov": "front-right", "camera_rear_left_70fov": "rear-left",
    "camera_rear_tele_30fov": "rear", "camera_rear_right_70fov": "rear-right",
    "camera_front_tele_30fov": "front telephoto",
}


def check_camera_indices(camera_indices: Optional[Iterable[int]],
                         n_cam: Optional[int] = None) -> Optional[list[int]]:
    """
    Validate a loader camera_indices list (tensor position -> loader camera index).
    None means "not given": callers fall back to position == loader index, the 7-camera
    convention the harness used before (documented fallback, kept for old manifests).
    """
    if camera_indices is None:
        return None
    ci = [int(i) for i in camera_indices]
    unknown = [i for i in ci if i not in CAM_INDEX_TO_ID]
    if unknown:
        raise ValueError(f"unknown loader camera index {unknown}; known: {sorted(CAM_INDEX_TO_ID)}")
    if len(set(ci)) != len(ci):
        raise ValueError(f"duplicate camera index in {ci}")
    if n_cam is not None and len(ci) != n_cam:
        raise ValueError(f"camera_indices has {len(ci)} entries but the tensor has {n_cam} cameras")
    return ci


def camera_id(position: int, camera_indices: Optional[list[int]] = None) -> Optional[str]:
    """camera_id of a tensor position; without camera_indices the position is the index."""
    idx = camera_indices[position] if camera_indices is not None else position
    return CAM_INDEX_TO_ID.get(int(idx))


def front_positions(n_cam: Optional[int], camera_indices: Optional[list[int]] = None) -> list[int]:
    """Tensor positions holding a forward camera (front_wide / front_tele)."""
    if camera_indices is not None:
        return [p for p, i in enumerate(camera_indices) if CAM_INDEX_TO_ID[i] in FRONT_CAMERA_IDS]
    fronts = [i for i, cid in CAM_INDEX_TO_ID.items() if cid in FRONT_CAMERA_IDS]
    return [i for i in fronts if n_cam is None or i < n_cam]


def camera_name(position: int, camera_indices: Optional[list[int]] = None) -> str:
    cid = camera_id(position, camera_indices)
    return CAM_ID_TO_NAME.get(cid, f"camera {position}")
