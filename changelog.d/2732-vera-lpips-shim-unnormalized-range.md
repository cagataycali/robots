### Fixed

- **policies/vera/docker**: the LPIPS compat shim no longer refuses a frame the
  check it stands in for accepts. VERA's DFoT metrics import `_valid_img` from
  `torchmetrics.image.lpip`, where modern torchmetrics no longer keeps it, and
  the image installs `sitecustomize_vera` as an auto-imported `sitecustomize` to
  supply the name - so the shim's verdict, not the real one, is the verdict the
  metric acts on. The two branches of the real check in
  `torchmetrics.functional.image.lpips` are deliberately asymmetric: both ends of
  `[0, 1]` are bounded when `normalize=True`, and only the LOWER end of
  `[-1, 1]` when `normalize=False`, because a frame already in the network's own
  range is allowed to overshoot `1.0` - which a decoder emitting a tanh-shaped
  frame routinely does by a float step or two. The shim bounded the upper end on
  both branches, and `_lpips_update` turns a `False` from that check into a
  `ValueError`, so the added bound did not make the metric stricter, it made an
  eval that the unshimmed torchmetrics runs raise inside the image. Measured
  against torchmetrics 1.9.0, 3 of 14 `(frame, normalize)` cells disagreed with
  the real check - all three `normalize=False` frames whose maximum exceeds
  `1.0`, the smallest of them by one float32 step - and 14 of 14 agree now. The
  `normalize=True` branch, the lower bound, and the rank and channel-count rules
  are unchanged, and the comment claiming the reimplementation was faithful now
  records the asymmetry and why it matters.
