### Fixed: microduck refuses an `inf` orientation without a NumPy `RuntimeWarning` first

`quat_rotate_inverse` lets a non-finite `base_quat` component propagate so the
assembled-vector pass refuses it by name, but for `inf` the normalising divide
was `inf / inf` and NumPy reported it as `RuntimeWarning: invalid value
encountered in divide` before the `ValueError` arrived - under `-W error` the
caller saw the warning instead of the refusal. The one division is now scoped
under `np.errstate(invalid="ignore")`; the refusal is unchanged (#4188).
