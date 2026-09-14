### Fixed: a `cameras=` no native driver will open is refused instead of dropped

`Robot(mode="real")` forwarded `cameras=` verbatim into the native driver it
built. No driver shipped here reads it - these robots address their cameras
through their own SDK - so nine discarded it on the first line of `__init__` and
three stored it under a private attribute nothing ever read. The keyword was
accepted, the build reported success, and the caller was handed a robot with no
cameras at all. Nothing said so, and the omission is invisible at the call: it
surfaces later as a recording with no image columns, or a visual policy asked to
act on a frame nobody captured. Eight shipped robots declare `driver="strands"`
in the registry, so a plain `Robot("unitree_go2", mode="real", cameras={...})`
reached that path without the caller ever writing `driver=`.

Both sibling doors already answered correctly, which makes this a convergence
rather than a new rule: `mode="sim"` refuses this same keyword and names the door
that does attach cameras, and the spawn arguments the real branch cannot honour
are at least reported. A dropped camera is the one a caller cannot see, so it is
the one that has to be refused - naming `driver="lerobot"`, which really does
attach it.

The refusal is read off the driver class rather than a list kept in the factory,
because drivers grow: one that opens the cameras it is given declares
`reads_cameras = True` and receives the dict verbatim, with nothing in the factory
to update. An empty `cameras` declares no camera and is still honoured by doing
nothing. The three drivers that stored the config under `self._cameras` now
discard it like the other nine, so the attribute no longer suggests a driver kept
something it never used.
