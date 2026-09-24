### Fixed: a Git LFS pointer is not the mesh a model declares

"Are this model's meshes on disk?" has a single owner, and it graded each
reference with `os.path.exists`. A `git clone` run where git-lfs is not
installed (or is configured to skip smudging) writes a ~130-byte
`version https://git-lfs.github.com/spec/v1` pointer in place of every file the
upstream repository keeps in LFS, and exits 0 - so the stub is on disk and every
reader called the tree complete:

```python
download_robots(names=["reachy_mini"], force=True)   # 1 downloaded, 0 failed
Robot("reachy_mini")                                 # MuJoCo: stl_decoder: number of
                                                     # faces should be between 1 and
                                                     # 200000 ... perhaps this is an
                                                     # ASCII file?
```

`reachy_mini` is the shipped entry that reaches it: its upstream stores all 49
of the meshes its MJCF declares in LFS. The fetch reported the robot delivered,
`_needs_download` then reported the assets present (so a retry was a no-op), and
MuJoCo's decoder was the first surface to object - against a path that exists.

The scan now classifies each reference (`absent` or `Git LFS pointer`) instead
of only counting, so all three readers agree and the two reports name the
remedy the case actually takes:

```
download_robots(...)  -> failed: mjcf/reachy_mini.xml declares 49 mesh file(s)
                         MuJoCo cannot load (Git LFS pointer): ... - install
                         git-lfs (`git lfs install`) and fetch again
Robot("reachy_mini")  -> Robot 'reachy_mini' has no loadable meshes after
                         re-fetching: <same detail>
```

`_ensure_meshes` also reads its verdict from the tree after the auto-download
rather than from the download call returning, so any fetch that does not deliver
the meshes is reported instead of handed to the loader.
