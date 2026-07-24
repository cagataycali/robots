# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for the GS backdrop alignment transforms (zero ML deps).

The ``GsplatBackground`` skybox/backdrop modes stand a captured 3DGS scene
upright and place it in the MuJoCo world via a ``world_from_gs`` transform
fit from the gaussian positions. These fits are pure-numpy geometry
(``_upright_view_transform``, ``_fit_backdrop_transform``, ``_auto_up_sign``,
``_fit_skybox_transform``) and are exposed for tuning, so their output
contract -- an orthonormal upright rotation, metric vs. radius scaling, and
floor/centroid placement -- is pinned here without needing gsplat or CUDA.
"""

from typing import Any

import numpy as np

from strands_robots.rendering.backgrounds import (
    _auto_up_sign,
    _fit_backdrop_transform,
    _fit_skybox_transform,
    _upright_view_transform,
)


def _room_cloud(n: int = 4000, seed: int = 0) -> np.ndarray:
    """A room-like slab: wide in X, deep in Y, thin in Z (so +Z reads as up)."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.uniform(-4.0, 4.0, n),
            rng.uniform(-2.0, 2.0, n),
            rng.uniform(-0.1, 0.1, n),
        ]
    ).astype(np.float64)


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine ``world_from_gs`` to ``(N, 3)`` points."""
    return pts @ T[:3, :3].T + T[:3, 3]


# --------------------------------------------------------------------------- #
# _upright_view_transform
# --------------------------------------------------------------------------- #


def test_upright_view_transform_is_a_proper_rotation_about_the_centroid() -> None:
    pts = _room_cloud()
    R, viewpoint = _upright_view_transform(pts)
    # A proper rotation: orthonormal with det +1 (no reflection).
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
    # The bake viewpoint sits at the scene centroid.
    assert np.allclose(viewpoint, pts.mean(axis=0))


def test_upright_view_transform_maps_thin_axis_to_world_up() -> None:
    # The thin (smallest-variance) axis of the slab is world +Z, so the fit
    # must stand it upright: the gs thin axis maps onto the world Z axis.
    pts = _room_cloud()
    R, _ = _upright_view_transform(pts)
    mapped = R @ np.array([0.0, 0.0, 1.0])
    assert abs(abs(mapped[2]) - 1.0) < 1e-2  # aligned with world Z
    assert np.linalg.norm(mapped[:2]) < 1e-2  # nothing left in-plane


# --------------------------------------------------------------------------- #
# _fit_backdrop_transform
# --------------------------------------------------------------------------- #


def test_fit_backdrop_transform_centres_scene_on_requested_point() -> None:
    pts = _room_cloud()
    center = np.array([0.05, 0.05, 0.25])
    T = _fit_backdrop_transform(pts, center=center, radius=3.0)
    # The centroid maps exactly onto the requested backdrop center.
    assert np.allclose(_apply(T, pts.mean(axis=0)), center, atol=1e-6)


def test_fit_backdrop_transform_scales_horizontal_extent_to_radius() -> None:
    pts = _room_cloud()
    radius = 3.0
    T = _fit_backdrop_transform(pts, center=np.zeros(3), radius=radius)
    world = _apply(T, pts)
    horiz = np.linalg.norm(world[:, :2] - world[:, :2].mean(axis=0), axis=1)
    # By construction the 95th-percentile in-plane radius is fit to ``radius``.
    assert np.isclose(np.percentile(horiz, 95), radius, rtol=0.02)
    # Rotation part is a uniform scale times an orthonormal basis.
    M = T[:3, :3]
    s = np.linalg.norm(M[0])
    assert np.allclose((M / s) @ (M / s).T, np.eye(3), atol=1e-6)


# --------------------------------------------------------------------------- #
# _auto_up_sign
# --------------------------------------------------------------------------- #


def test_auto_up_sign_returns_a_deterministic_unit_sign() -> None:
    pts = _room_cloud()
    sign = _auto_up_sign(pts)
    assert sign in (-1.0, 1.0)
    # Deterministic for a fixed cloud (callers cache the fit across frames).
    assert _auto_up_sign(pts) == sign


# --------------------------------------------------------------------------- #
# _fit_skybox_transform
# --------------------------------------------------------------------------- #


def test_fit_skybox_transform_places_floor_percentile_at_floor_z() -> None:
    pts = _room_cloud()
    floor_z = -0.3
    T = _fit_skybox_transform(
        pts,
        up_sign=1.0,
        floor_z=floor_z,
        floor_pct=2.0,
        metric=True,
        up_axis=(0.0, 0.0, 1.0),
        major_axis=(1.0, 0.0, 0.0),
    )
    world = _apply(T, pts)
    # The floor_pct percentile of world-z is seated exactly at floor_z (this
    # is how the arm gets placed on the photoreal surface).
    assert np.isclose(np.percentile(world[:, 2], 2.0), floor_z, atol=1e-4)


def test_fit_skybox_transform_centres_horizontal_centroid() -> None:
    pts = _room_cloud()
    center = (0.05, 0.05)
    T = _fit_skybox_transform(
        pts,
        center=center,
        metric=True,
        up_axis=(0.0, 0.0, 1.0),
        major_axis=(1.0, 0.0, 0.0),
    )
    world_centroid = _apply(T, pts.mean(axis=0))
    assert np.allclose(world_centroid[:2], center, atol=1e-4)


def test_fit_skybox_transform_metric_keeps_scale_nonmetric_fits_radius() -> None:
    pts = _room_cloud()
    # metric=True keeps the asset's own scale: with an identity basis the
    # rotation block is orthonormal (unit scale).
    T_metric = _fit_skybox_transform(pts, metric=True, up_axis=(0.0, 0.0, 1.0), major_axis=(1.0, 0.0, 0.0))
    assert np.allclose(T_metric[:3, :3], np.eye(3), atol=1e-9)
    # metric=False rescales the horizontal 95th-percentile extent to ``radius``.
    radius = 2.5
    T_scaled = _fit_skybox_transform(
        pts,
        radius=radius,
        center=(0.0, 0.0),
        metric=False,
        up_axis=(0.0, 0.0, 1.0),
        major_axis=(1.0, 0.0, 0.0),
    )
    world = _apply(T_scaled, pts)
    horiz = np.linalg.norm(world[:, :2], axis=1)
    assert np.isclose(np.percentile(horiz, 95), radius, rtol=0.02)


def test_fit_skybox_transform_up_sign_flips_world_up_direction() -> None:
    pts = _room_cloud()
    kw: dict[str, Any] = dict(metric=True, up_axis=(0.0, 0.0, 1.0), major_axis=(1.0, 0.0, 0.0))
    T_pos = _fit_skybox_transform(pts, up_sign=1.0, **kw)
    T_neg = _fit_skybox_transform(pts, up_sign=-1.0, **kw)
    # The world-Z row of the basis inverts when the up sign flips.
    assert np.allclose(T_pos[2, :3], -T_neg[2, :3], atol=1e-9)


def test_fit_skybox_transform_yaw_rotates_in_plane_orientation() -> None:
    pts = _room_cloud()
    kw: dict[str, Any] = dict(metric=True, up_axis=(0.0, 0.0, 1.0), major_axis=(1.0, 0.0, 0.0))
    T0 = _fit_skybox_transform(pts, yaw_deg=0.0, **kw)
    T90 = _fit_skybox_transform(pts, yaw_deg=90.0, **kw)
    # A 90-degree yaw about world +Z sends the in-plane major axis (world +X)
    # onto world -Y, while leaving the up axis (row 2) untouched.
    assert np.allclose(T0[0, :3], [1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(T90[0, :3], [0.0, -1.0, 0.0], atol=1e-9)
    assert np.allclose(T0[2, :3], T90[2, :3], atol=1e-9)
