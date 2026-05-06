"""SMPL fitting via staged optimization: global → pose → shape.

Requires SMPL model files (gender-neutral .pkl) from https://smpl.is.tue.mpg.de/
Place at: models/smpl/SMPL_NEUTRAL.pkl
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import smplx
import torch
import torch.nn as nn

from fitmirror.body.pose import PoseResult

logger = logging.getLogger(__name__)

# Default SMPL model path — override via env SMPL_MODEL_PATH
DEFAULT_SMPL_PATH = Path("models/smpl")

# Iteration budget per stage
STAGE_ITERS = {1: 300, 2: 500, 3: 500}
LR = {1: 5e-3, 2: 3e-3, 3: 2e-3}


@dataclass
class SMPLFitResult:
    """Output of SMPL fitting."""
    betas: np.ndarray          # (10,)  shape parameters
    body_pose: np.ndarray      # (69,)  axis-angle pose (23 joints × 3)
    global_orient: np.ndarray  # (3,)   global orientation
    transl: np.ndarray         # (3,)   translation
    vertices: np.ndarray       # (6890, 3) mesh vertices in camera frame
    joints: np.ndarray         # (45, 3)  3D joints
    faces: np.ndarray          # (13776, 3) mesh faces (static)
    scale: float               # pixel-to-metre scale factor


def _perspective_project(
    points3d: torch.Tensor,
    focal: float,
    center: tuple[float, float],
) -> torch.Tensor:
    """Simple pinhole projection. points3d: (N, 3) → (N, 2) pixel coords."""
    x = points3d[:, 0] / (points3d[:, 2] + 1e-6) * focal + center[0]
    y = points3d[:, 1] / (points3d[:, 2] + 1e-6) * focal + center[1]
    return torch.stack([x, y], dim=-1)


class SMPLFitter:
    """Fits SMPL parameters to MediaPipe 2D keypoints + monocular depth."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_SMPL_PATH,
        gender: str = "neutral",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"SMPL model not found at {model_path}. "
                "Download from https://smpl.is.tue.mpg.de/ and place "
                "SMPL_NEUTRAL.pkl at models/smpl/."
            )
        self._smpl = smplx.create(
            str(model_path),
            model_type="smpl",
            gender=gender,
            num_betas=10,
            batch_size=1,
        ).to(self.device)
        self._faces: np.ndarray = self._smpl.faces.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        pose_result: PoseResult,
        depth_map: np.ndarray,
        mask: np.ndarray,
    ) -> SMPLFitResult:
        """Run 3-stage fitting.

        Stage 1: global orient + translation (camera placement)
        Stage 2: + body pose
        Stage 3: + betas (shape)
        """
        kpts_px, weights, smpl_ids = pose_result.smpl_keypoints
        H, W = pose_result.image_hw
        focal = max(H, W)  # approximate focal length in pixels
        center = (W / 2.0, H / 2.0)

        # Estimate scale from shoulder span (pixel) → real shoulder width ~38 cm
        scale = self._init_scale(pose_result, focal)

        kpts_t = torch.tensor(kpts_px, dtype=torch.float32, device=self.device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)
        smpl_ids_t = smpl_ids  # numpy, used for indexing

        # ---- learnable parameters ----
        betas = nn.Parameter(torch.zeros(1, 10, device=self.device))
        body_pose = nn.Parameter(torch.zeros(1, 69, device=self.device))
        global_orient = nn.Parameter(torch.zeros(1, 3, device=self.device))
        transl = nn.Parameter(torch.tensor([[0.0, 0.0, 5.0]], device=self.device))

        # Stage configs: (params_to_optimize, n_iters, lr)
        stages = [
            ([global_orient, transl], STAGE_ITERS[1], LR[1]),
            ([global_orient, transl, body_pose], STAGE_ITERS[2], LR[2]),
            ([global_orient, transl, body_pose, betas], STAGE_ITERS[3], LR[3]),
        ]

        for stage_idx, (params, n_iters, lr) in enumerate(stages, 1):
            optimizer = torch.optim.Adam(params, lr=lr)
            for it in range(n_iters):
                optimizer.zero_grad()
                output = self._smpl(
                    betas=betas,
                    body_pose=body_pose,
                    global_orient=global_orient,
                    transl=transl,
                    return_verts=True,
                )
                joints3d = output.joints[0]  # (45, 3)

                # Project selected joints to 2D
                sel_joints = joints3d[smpl_ids_t]  # (N, 3)
                proj = _perspective_project(sel_joints, focal, center)  # (N, 2)

                # 2D reprojection loss (weighted)
                loss_2d = (weights_t * ((proj - kpts_t) ** 2).sum(-1)).mean()

                # Depth consistency loss on joint depth
                depth_vals = self._sample_depth(kpts_px, depth_map)
                depth_t = torch.tensor(depth_vals, dtype=torch.float32, device=self.device)
                pred_depth_norm = (sel_joints[:, 2] - sel_joints[:, 2].min()) / (
                    sel_joints[:, 2].max() - sel_joints[:, 2].min() + 1e-6
                )
                loss_depth = ((pred_depth_norm - depth_t) ** 2 * weights_t).mean()

                # Regularisation
                loss_pose_reg = (body_pose**2).mean() * 0.01
                loss_shape_reg = (betas**2).mean() * 0.001

                loss = loss_2d + 0.5 * loss_depth + loss_pose_reg + loss_shape_reg
                loss.backward()
                optimizer.step()

                if it % 100 == 0:
                    logger.debug(
                        f"Stage {stage_idx} iter {it}: loss={loss.item():.4f} "
                        f"2d={loss_2d.item():.4f} depth={loss_depth.item():.4f}"
                    )

        # Final forward pass
        with torch.no_grad():
            final = self._smpl(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                return_verts=True,
            )

        return SMPLFitResult(
            betas=betas.detach().cpu().numpy()[0],
            body_pose=body_pose.detach().cpu().numpy()[0],
            global_orient=global_orient.detach().cpu().numpy()[0],
            transl=transl.detach().cpu().numpy()[0],
            vertices=final.vertices.detach().cpu().numpy()[0],
            joints=final.joints.detach().cpu().numpy()[0],
            faces=self._faces,
            scale=scale,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_scale(self, pose_result: PoseResult, focal: float) -> float:
        """Estimate pixel-to-metre via shoulder span heuristic (~38 cm shoulder width)."""
        lm = pose_result.landmarks_px
        vis = pose_result.visibility
        SHOULDER_WIDTH_M = 0.38
        if vis[11] > 0.5 and vis[12] > 0.5:
            span_px = abs(lm[11, 0] - lm[12, 0])
            if span_px > 1:
                return SHOULDER_WIDTH_M / span_px * focal
        return 5.0  # fallback: 5 metres camera distance

    @staticmethod
    def _sample_depth(kpts_px: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """Sample depth map at keypoint locations (bilinear)."""
        H, W = depth_map.shape
        vals = []
        for x, y in kpts_px:
            xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
            vals.append(depth_map[yi, xi])
        arr = np.array(vals, dtype=np.float32)
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / (rng + 1e-6)
