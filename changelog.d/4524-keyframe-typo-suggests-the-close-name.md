### Fixed: an unknown keyframe name suggests the keyframe it was a typo of

`Robot("panda", keyframe="hom")` answered `Keyframe 'hom' not found in
'scene.xml'. Available: 'home'.` and left the caller to spot the typo. The
MuJoCo, mjlab and Isaac keyframe refusals now append the package's
`Did you mean: 'hom' -> 'home'?` clause before the list of keyframes the model
declares. A name that is not close to any keyframe still gets the plain list.
