---
description: Inject a triangle-mesh asset into a live scene, and give a surface a material or a texture.
---

# Meshes and materials

## Mesh objects

Beyond primitives, `add_object` can inject a triangle-mesh asset (STL/OBJ) into
the live scene at runtime. Pass `shape="mesh"` with a `mesh_path` to the asset
file; the extent is defined by the mesh's own units, so `size` is ignored on
this backend - a read the Isaac backend's mesh `add_object` shares. The Newton
backend consumes it instead, as a per-axis scale on the
loaded geometry, so a mesh add carrying a `size` does not mean the same thing
there - which meaning is right is tracked in
[#2300](https://github.com/strands-labs/robots/issues/2300).

```python
sim.add_object(name="bracket", shape="mesh", mesh_path="/abs/path/bracket.stl",
               position=[0.3, 0.0, 0.1])
# 'bracket' added: mesh at [0.3, 0.0, 0.1], extent=[0.12, 0.08, 0.03]m from the
# asset (collision uses its convex hull), 0.1kg
```

As for every shape, the success text reports the extent read back off the
compiled geom rather than echoing the request - and for a mesh the request
carries no extent at all, so there is nothing else it could report. The asset
can be any size.

### A mesh geom collides as its convex hull

MuJoCo collides a mesh geom as its **convex hull**, not as the triangles that
render. For a convex asset (a bracket, a mug body, a crate) the two coincide and
there is nothing to think about. For a concave one - a scanned or generated room
shell, a tray, a shelf, a bowl - the hull fills every cavity, so:

* an object placed "inside" the cavity starts inside solid geometry and is pushed
  out, and one dropped in rests on the filled hull instead of on the interior
  floor;
* a camera still shows the open interior, because rendering uses the triangles.
  Nothing looks wrong.

To get load-bearing concave geometry, decompose the asset into convex parts and
add one mesh object per part:

```python
for i, part in enumerate(convex_parts):          # e.g. a V-HACD decomposition
    sim.add_object(name=f"room_{i}", shape="mesh", mesh_path=part, is_static=True)
```

A single-mesh room is still useful as a visual backdrop; it just is not a floor.

`mesh_path` is required for `shape="mesh"` - a mesh without a path is rejected
with an actionable error rather than an opaque recompile failure. If the mesh
file cannot be loaded the add is rejected and the scene is rolled back to its
previous compilable state (including the mesh asset), so the object name stays
reusable and one bad add never bricks later scene edits.

## Materials and textures

By default an object renders with a flat `color` (rgba) - a glossy, obviously
synthetic primitive. Pass `material=` to `add_object` to attach a real MuJoCo
material so the surface can be matte or carry a texture. This narrows the
sim-to-real visual gap for VLM/VLA policies trained on real footage. The
`color` (rgba) still applies and tints a textured or solid material.

```python
# Matte (non-plastic) surface: kill specular highlight + shininess.
sim.add_object("apple", shape="sphere", size=[0.04, 0, 0], color=[0.8, 0.1, 0.1, 1],
               material={"specular": 0, "shininess": 0, "reflectance": 0})

# Image texture from disk (absolute path), tiled 2x2 across the surface.
sim.add_object("table", shape="box", size=[0.5, 0.5, 0.02], is_static=True,
               material={"texture": "/abs/path/wood.png", "texrepeat": [2, 2],
                         "specular": 0, "shininess": 0})

# Procedural builtin texture (no image file needed).
sim.add_object("floor_tile", shape="box", size=[0.3, 0.3, 0.01], is_static=True,
               material={"builtin": "checker", "rgb1": [0.2, 0.3, 0.4],
                         "rgb2": [0.1, 0.2, 0.3], "texdim": 512})
```

`material` is a dict; all keys are optional:

| Key | Type | Meaning |
|-----|------|---------|
| `reflectance` / `specular` / `shininess` | float 0..1 | Surface response. `specular=0, shininess=0` = matte; the defaults read as glossy plastic. |
| `texrepeat` | `[u, v]` | Texture tiling across the surface. |
| `texture` | str | Absolute path to an image file (PNG/etc.) used as the RGB texture. |
| `builtin` | `"checker" \| "gradient" \| "flat"` | Procedural texture, coloured by `rgb1` / `rgb2` and sized `texdim` (default 512) per side. |

Specify **either** `texture` **or** `builtin`, not both. An invalid texture
path, an unknown `builtin` name, or specifying both fails loudly with a
`ValueError` (returned as a `status=error` dict through the agent tool) - there
is no silent fallback to the flat-plastic default.

Only the keys in the table above are accepted. A key outside it (a typo such as
`rgb_1`, or a field borrowed from another renderer such as `roughness`), an
empty `material={}`, or `rgb1`/`rgb2`/`texdim` without `builtin` is rejected the
same way - the alternative is an object that compiles with MuJoCo's glossy
defaults while `add_object` reports success:

```python
sim.add_object("cube", material={"builtin": "checker", "rgb_1": [1, 0, 0]})
# status=error: unknown material key(s): 'rgb_1' (did you mean 'rgb1'?).
#               Accepted keys: builtin, reflectance, rgb1, rgb2, shininess, ...
```

For natural surfaces prefer
an **image texture**; the `checker` builtin reads as a literal checkerboard.
Materials are currently supported by the MuJoCo backend; the Newton backend
rejects a non-`None` `material` rather than silently ignoring it.

## See also

- [Objects](objects.md)
- [World building](world-building.md)
