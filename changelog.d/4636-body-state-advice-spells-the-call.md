### Fixed: advice that points at `get_body_state` spells the call, keyword included

A `move_to` that pushed an object and a close that touched nothing both told
the caller to use `get_body_state` without saying how. Having just written
`add_object(name=...)`, a caller guessed `get_body_state(name=...)` and got a
plain `TypeError`. The replies now spell `get_body_state(body_name='red_cube')`
(or `body_name=<the name add_object gave it>` when no object is near).
