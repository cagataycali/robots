### Bug Fixes

Every copy of a downloaded robot tree now refuses to follow a symlink inside it.
`strands_robots.assets.download` reaches the same upstream description
repositories by three routes. The two clone routes each passed
`reject_symlinks=True` and carried a comment naming the escape: `shutil.copytree`
defaults to `symlinks=False`, which *follows* a nested symlink, so a description
carrying `robot_dir/assets -> $HOME/.ssh` copies host files into the asset cache.
The `robot_descriptions` route reaches those same clones through the installed
package and called a bare `shutil.copytree`, so a package holding that link
copied the private key into `$STRANDS_ASSETS_DIR` and reported `downloaded`. That
copy is not an edge path: it is taken whenever the cache cannot hold a symlink -
a FAT/exFAT card on an edge robot, or Windows without the privilege - which is
exactly the host the guard was missing on.

The two copy helpers are now one owner, `_copy_external_tree`, and the symlink
skip is unconditional: no tree reaching it is trusted, so a caller can no longer
ask for the following behaviour. Each skipped entry is logged by name, so a model
left incomplete by an intra-tree link is diagnosable rather than merely wrong.
The `robot_descriptions` route passes `drop_docs=False`, keeping the docs and
preview images its preferred path exposes by symlinking the package directory
whole - a cache that cannot hold a symlink must not silently hold fewer files
than one that can. A package-wide test pins the rule instead of the two sites:
the only `shutil.copytree` call in `strands_robots` is the one inside that owner.

`docs/security.md` gains a **Robot asset cache (`STRANDS_ASSETS_DIR`)** section
stating the posture an operator can rely on: what is treated as externally
sourced, that symlinks are never followed into the cache, that `safe_join`
closes the `../` and symlinked-root routes, and that the cache itself is
integrity-sensitive because it holds the MJCF a `mode="real"` robot is built
from.
