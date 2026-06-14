"""WBC asset management - download G1 WBC MuJoCo model from HuggingFace.

The WBC policy requires a specific G1 XML with motor (torque) actuators,
different from the standard Menagerie G1 (position actuators). This module
handles downloading the WBC-specific model and meshes from the
``nvidia/GR00T-WholeBodyControl`` HuggingFace repository.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Files needed from the HF repo for the G1 WBC model
_WBC_MODEL_FILES = [
    "decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.xml",
    "decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml",
]

# Mesh directory in the HF repo
_WBC_MESH_PREFIX = "decoupled_wbc/sim2mujoco/resources/robots/g1/meshes/"

_CACHE_DIR_NAME = "g1_wbc"


def get_wbc_asset_dir() -> Path:
    """Get the local cache directory for WBC assets."""
    base = Path(os.environ.get("STRANDS_ASSETS_DIR", Path.home() / ".strands_robots" / "assets"))
    return base / _CACHE_DIR_NAME


def get_wbc_xml_path(checkpoint: str = "nvidia/GR00T-WholeBodyControl") -> Path:
    """Get path to the G1 WBC XML, downloading if needed.

    Args:
        checkpoint: HuggingFace repo ID.

    Returns:
        Path to the local g1_gear_wbc.xml file.
    """
    asset_dir = get_wbc_asset_dir()
    xml_path = asset_dir / "g1_gear_wbc.xml"

    if xml_path.exists():
        return xml_path

    logger.info("Downloading G1 WBC assets from %s...", checkpoint)
    _download_wbc_assets(checkpoint, asset_dir)

    if not xml_path.exists():
        raise FileNotFoundError(
            f"G1 WBC XML not found at {xml_path} after download. "
            f"Check that the checkpoint '{checkpoint}' contains the expected files."
        )

    return xml_path


def _download_wbc_assets(checkpoint: str, target_dir: Path) -> None:
    """Download WBC model files and meshes from HuggingFace.

    Strategy:
    1. Try snapshot_download with allow_patterns (efficient, single call).
    2. Fall back to individual hf_hub_download calls.
    3. Copy from local cache if already present (e.g. from a previous run).
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_tree
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for WBC asset download. Install with: pip install huggingface_hub"
        ) from e

    target_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir = target_dir / "meshes"
    meshes_dir.mkdir(exist_ok=True)

    # Download main model files
    for rel_path in _WBC_MODEL_FILES:
        filename = Path(rel_path).name
        local_path = target_dir / filename
        if local_path.exists():
            continue
        try:
            downloaded = hf_hub_download(
                repo_id=checkpoint,
                filename=rel_path,
            )
            shutil.copy2(downloaded, local_path)
            logger.debug("Downloaded %s -> %s", rel_path, local_path)
        except Exception as e:
            logger.warning("Failed to download %s: %s", rel_path, e)

    # Download meshes - enumerate repo tree for the meshes directory
    try:
        tree = list_repo_tree(checkpoint, path_in_repo="decoupled_wbc/sim2mujoco/resources/robots/g1/meshes")
        mesh_files = [item.rfilename for item in tree if hasattr(item, "rfilename") and item.rfilename.endswith(".STL")]
    except Exception:
        # Fall back to known mesh list from the XML
        mesh_files = _get_known_mesh_files()

    for mesh_rel in mesh_files:
        mesh_name = Path(mesh_rel).name
        local_mesh = meshes_dir / mesh_name
        if local_mesh.exists():
            continue
        try:
            if "/" in mesh_rel:
                # Full path from tree listing
                downloaded = hf_hub_download(repo_id=checkpoint, filename=mesh_rel)
            else:
                # Just filename - prepend path
                downloaded = hf_hub_download(
                    repo_id=checkpoint,
                    filename=f"{_WBC_MESH_PREFIX}{mesh_name}",
                )
            shutil.copy2(downloaded, local_mesh)
        except Exception as e:
            logger.debug("Mesh download failed for %s: %s", mesh_name, e)

    logger.info("WBC assets downloaded to %s (%d meshes)", target_dir, len(list(meshes_dir.glob("*.STL"))))


def _get_known_mesh_files() -> list[str]:
    """Return known mesh filenames for G1 WBC model (fallback when tree listing fails)."""
    return [
        "pelvis.STL",
        "pelvis_contour_link.STL",
        "left_hip_pitch_link.STL",
        "left_hip_roll_link.STL",
        "left_hip_yaw_link.STL",
        "left_knee_link.STL",
        "left_ankle_pitch_link.STL",
        "left_ankle_roll_link.STL",
        "right_hip_pitch_link.STL",
        "right_hip_roll_link.STL",
        "right_hip_yaw_link.STL",
        "right_knee_link.STL",
        "right_ankle_pitch_link.STL",
        "right_ankle_roll_link.STL",
        "waist_yaw_link_rev_1_0.STL",
        "waist_roll_link_rev_1_0.STL",
        "torso_link_rev_1_0.STL",
        "logo_link.STL",
        "head_link.STL",
        "left_shoulder_pitch_link.STL",
        "left_shoulder_roll_link.STL",
        "left_shoulder_yaw_link.STL",
        "left_elbow_link.STL",
        "left_wrist_roll_link.STL",
        "left_wrist_pitch_link.STL",
        "left_wrist_yaw_link.STL",
        "left_hand_palm_link.STL",
        "left_hand_thumb_0_link.STL",
        "left_hand_thumb_1_link.STL",
        "left_hand_thumb_2_link.STL",
        "left_hand_middle_0_link.STL",
        "left_hand_middle_1_link.STL",
        "left_hand_index_0_link.STL",
        "left_hand_index_1_link.STL",
        "right_shoulder_pitch_link.STL",
        "right_shoulder_roll_link.STL",
        "right_shoulder_yaw_link.STL",
        "right_elbow_link.STL",
        "right_wrist_roll_link.STL",
        "right_wrist_pitch_link.STL",
        "right_wrist_yaw_link.STL",
        "right_hand_palm_link.STL",
        "right_hand_thumb_0_link.STL",
        "right_hand_thumb_1_link.STL",
        "right_hand_thumb_2_link.STL",
        "right_hand_middle_0_link.STL",
        "right_hand_middle_1_link.STL",
        "right_hand_index_0_link.STL",
        "right_hand_index_1_link.STL",
    ]
