### Fixed: an `.spz` scene whose header count does not match its payload is refused

`_load_spz_splats` graded the header's magic and version and then trusted
`num_points`, the signed int32 that sizes every block of the body. Nothing else
in the file restates those lengths, so a count that disagreed with the payload
could not be caught later:

* a count **short** of the payload was silent - each block read its declared
  prefix, so every later block started at the wrong offset and the scene
  decoded to garbage that came back as a successfully loaded background. A real
  213,944-splat asset with its count halved rendered as a flat wash of colour
  with no error;
* a count of **zero** returned an empty splat set, i.e. a backdrop that
  rasterizes no gaussians, also reported as loaded;
* a **negative** count is not refused by `np.frombuffer`, which reads any
  negative `count` as "the whole remaining buffer" while `off += N * width`
  walks the offset backwards through the header;
* a count **past** the payload surfaced as a bare
  `buffer is smaller than requested size`, naming neither the file nor the
  field that sized the read - the interrupted-download case, where the cached
  file is reused on every later load.

The count is now graded for sign beside the magic and the version, and the
declared counts must account for the payload's exact byte length - the same
sizes-must-reconcile test the `.msh` and binary-STL readers already apply, and
which the shipped `tabletop` scene satisfies exactly. The refusal names the
file, the count, and both byte totals. The SH block's own shortfall check is
subsumed by the reconciliation and its wording folded into it.
