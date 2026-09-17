### Docs: the nav is eight sections and the landing page shows a robot moving

The site's sidebar had grown to 24 top-level entries - three of them ROS
variants, several of them single pages sitting beside `Robots` - so the index of
58 pages was longer than most of the pages it indexed. They are now eight
sections (Start, Robots, Simulate, Real Hardware, Policies, Agents & Dashboard,
Data & Training, Reference), each one level deep. No page moved, so every
published URL resolves exactly as before and no redirect map was needed.

`docs/index.md` is the 30-second version of the product: the install, the
`Robot("so101", mode="sim")` call with the agent line that drives it, three
cards (simulate / real hardware / hand it to an agent), four clips of robots
actually moving, and the driver-coverage lookup. A new guard runs the page's own
example - the arguments are read out of the fence - so a landing page that
cannot build a robot fails the suite instead of the reader, and keeps the nav
from regrowing: at most eight top-level entries, one level of nesting, and every
page under `docs/` reachable from it.
