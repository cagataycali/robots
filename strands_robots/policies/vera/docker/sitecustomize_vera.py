"""VERA torchmetrics compat shim (auto-loaded via PYTHONPATH/sitecustomize).

VERA's DFoT metrics import three symbols from ``torchmetrics.image.lpip`` that
were public in the pre-1.x layout but were renamed/removed in modern
torchmetrics (the version the NGC base ships):

    NoTrainLpips      -> renamed to the private ``_NoTrainLpips``
    _valid_img        -> removed from the public module
    (FrechetInceptionDistance at ``torchmetrics.image`` needs torch-fidelity)

Rather than pin torchmetrics back to 0.11.x (which collides with lightning 2.6),
we keep the modern torchmetrics and re-expose the names VERA expects. These are
**eval-only metrics** not used on the inference rollout path, so the shim only
needs to satisfy the import - exactness of the metric is irrelevant for serving.
"""

from __future__ import annotations


def _apply() -> None:
    try:
        import torchmetrics.image.lpip as _lpip
    except Exception:
        return

    # NoTrainLpips: modern torchmetrics underscores it.
    if not hasattr(_lpip, "NoTrainLpips"):
        priv = getattr(_lpip, "_NoTrainLpips", None)
        if priv is not None:
            _lpip.NoTrainLpips = priv

    # _valid_img: removed from the public module in modern torchmetrics - mirror
    # the surviving implementation in torchmetrics.functional.image.lpips, the one
    # the metric itself calls. Its two branches are deliberately asymmetric:
    # normalize=True bounds both ends of [0, 1], while normalize=False bounds only
    # the LOWER end of [-1, 1]. Bounding the upper end here too would refuse a
    # tensor the real check accepts, and _lpips_update turns a False into a
    # ValueError - so a decoded frame that overshoots 1.0 by a float step would
    # fail an eval that the unshimmed torchmetrics runs.
    if not hasattr(_lpip, "_valid_img"):

        def _valid_img(img, normalize: bool):  # noqa: ANN001
            value_check = img.min() >= 0.0 and img.max() <= 1.0 if normalize else img.min() >= -1.0
            return img.ndim == 4 and img.shape[1] == 3 and bool(value_check)

        _lpip._valid_img = _valid_img


_apply()
