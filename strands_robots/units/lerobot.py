"""LeRobot's motor normalisation, as a unit scheme.

A LeRobot checkpoint un-normalises to the units its dataset stored. For an SO-10x class arm that is
the Feetech bus range: each body joint's calibrated travel maps onto ``range_m100_100`` (-100..100),
the gripper onto ``range_0_100`` (0..100). The per-joint mode is registry data
(``hardware.lerobot_norm``: a ``"*"`` default plus per-key overrides).

The convention, once: the feetech driver maps a joint's calibrated ``range_min..range_max`` counts
linearly onto the model's joint limits ``lo..hi``, and this scheme maps the same ``lo..hi`` onto
-100..100 (or 0..100). Because both use the same ``lo``/``hi``, counts -> rad -> lerobot equals
LeRobot's own normalisation ``(counts - range_min) / (range_max - range_min) * 200 - 100``.

LeRobot's ``degrees`` mode centres on the calibrated range's midpoint and scales by ticks (4096 per
turn); an exact map needs the ``Calibration``, which a ``Description`` does not carry, so this scheme
refuses ``degrees`` rather than approximate it.
"""

from __future__ import annotations

import numpy as np

from ..description import Description
from .scheme import Units, register

_RANGE = {"range_m100_100": (-100.0, 100.0), "range_0_100": (0.0, 100.0)}


@register("lerobot")
def lerobot(description: Description) -> Units:
    norm = description.hardware.lerobot_norm
    if norm is None:
        raise ValueError(
            f"{description.name} declares no hardware.lerobot_norm in the registry; the lerobot units need one"
        )
    keys = description.action_keys
    missing = [k for k in keys if k not in description.limits]
    if missing:
        raise ValueError(f"'lerobot' needs joint limits for every action key; {description.name} lacks {missing}")
    lo = np.array([description.limits[k][0] for k in keys])
    hi = np.array([description.limits[k][1] for k in keys])
    lo_u, hi_u = [], []
    for k in keys:
        mode = norm.get(k, norm.get("*"))
        if mode == "degrees":
            raise ValueError(
                f"{description.name}: joint {k!r} uses LeRobot's 'degrees' mode, which is centred on the "
                "calibrated midpoint and needs the Calibration to map exactly; the lerobot units cannot derive it"
            )
        if mode not in _RANGE:
            raise ValueError(f"{description.name}: joint {k!r} has unknown lerobot_norm mode {mode!r}")
        a, b = _RANGE[mode]
        lo_u.append(a)
        hi_u.append(b)
    return Units("lerobot", keys, lo, hi, np.array(lo_u), np.array(hi_u))
