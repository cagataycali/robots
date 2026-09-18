### Fixed: the humanoid head-camera fence mounts on a body the page's sim has

`docs/robots/humanoids.md` told you to mount a head camera on `g1/torso_link`
after building `Robot("unitree_g1")`. Bodies are namespaced by the name you pass
to `Robot(...)`, so that sim reports `unitree_g1/torso_link` and `add_camera`
refused the fence with `status=error`. The fence now builds its own
`Robot("unitree_g1")` and mounts on `unitree_g1/torso_link`; the wrist-link
mention and a note on the namespace rule follow. A docs grader now checks that
every literal `parent_body="<ns>/<body>"` in a page's fences names a robot that
page builds.
