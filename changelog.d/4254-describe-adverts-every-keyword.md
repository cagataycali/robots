### Fixed: describe() advertises add_robot's keyframe and MuJoCo render's output_path

An agent reading `describe()` could not discover `add_robot(keyframe=...)` or MuJoCo's `render(output_path=...)`: both adverts stopped short of the real signature. The shared advert names `keyframe`; the MuJoCo `describe()` amends its `render` advert; a grader now requires every real keyword to be advertised unless the advert is abridged with `...` on purpose.
