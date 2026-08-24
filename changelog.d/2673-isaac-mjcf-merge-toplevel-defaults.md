### Fixed: a model's several top-level `<default>` elements merge into one root class

`<default>` is a top-level MJCF element, and MuJoCo merges every one a model carries into the single root class - the same model-global treatment it gives `<compiler>`, `<asset>` and `<worldbody>`. The merge is per attribute and in document order, so a later element overriding `size` does not discard a `type` an earlier one declared.

The Isaac loaders' default-class resolver read each top-level element independently, restarting from nothing, so the last one **replaced** the others. The dropped attribute then failed the familiar silent way: a geom resolving against the root class found no `type`, so `load_mjcf` reported the `box (0.05, 0.05, 0.05)` fallback for a capsule under a successful load. `group` is read through the same resolver, so collision/visual filtering was wrong for the same geoms.

The shape does not require a file that writes two `<default>` elements itself, which is what makes it more than a curiosity: `<include>` is a textual splice, so a scene including a robot contributes one element each. Measured on Menagerie, 7 models carry two top-level `<default>` elements once spliced, and `pal_tiago` and `pal_tiago_dual` lost their whole root class to the replace - 15 of 35 and 11 of 43 geoms respectively had no resolvable `group`.

The root class is now accumulated across the elements rather than restarted at each. A nested class is still flattened where it appears, which is what MuJoCo does: it inherits the root as accumulated up to its own position, so a `type` declared by a top-level element *after* the one enclosing it does not reach it. That boundary is pinned, so widening the merge later is a decision rather than an accident.
