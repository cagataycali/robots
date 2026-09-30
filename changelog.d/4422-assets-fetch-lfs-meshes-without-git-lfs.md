### Fixed: a robot whose meshes live in Git LFS downloads real meshes without git-lfs

`download_assets` fetched a GitHub-sourced robot with `git clone`, which on a machine
without git-lfs writes every LFS-tracked file as a text pointer and exits 0.
`reachy_mini`'s upstream keeps all its meshes in LFS, so the download reported
success and `add_robot("reachy_mini")` then failed on "49 mesh file(s) MuJoCo cannot
load (Git LFS pointer)", with installing git-lfs as the only way out. The fetcher now
replaces each pointer with the object from GitHub's LFS media endpoint at the cloned
commit, keeping it only when its SHA-256 and size match the pointer (1 GiB cap per
robot). `reachy_mini` now downloads and loads on MuJoCo and on Isaac Sim 6.1 with no
git-lfs installed.
