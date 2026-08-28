### Fixed: a Microduck scale or threshold that silently changes what the policy commands is refused

The Microduck provider validates its *structural* arguments carefully.
`MicroduckPolicyBundle` refuses an empty mapping, refuses a value that is not a
`MicroduckPolicy` (by name and by type), and refuses an `active` skill that is
not one of its keys; `MicroduckPolicy.set_robot_state_keys` routes its list
through the shared `name_list_error` domain. The two caller-supplied *numbers* in
that surface reached their consumer through a bare `float()`.

Neither number fails when it is unusable - each one silently changes what the
policy commands. Measured through the public surfaces on a 14-DOF stub-backed
policy, with the decode `motor_target = default_pose + raw_action * action_scale`:

| `action_scale` | accepted | non-finite targets | targets == `default_pose` |
| --- | --- | --- | --- |
| `0.25` (control) | yes | 0/14 | no |
| `0` | yes | 0/14 | **yes** |
| `False` | yes | 0/14 | **yes** |
| `nan` | yes | **14/14** | no |
| `inf` | yes | **14/14** | no |

A scale of `0` (or `False`, an `int` subclass) makes every target exactly
`default_pose`: the network's decision is discarded and the biped holds its
nominal stance while the rollout reports success. A non-finite scale makes all
fourteen targets `nan`. Nothing downstream checked it - `decode_action` runs per
tick inside `get_actions`, so a non-real value surfaced as a bare `TypeError`
from its own `float()` only after the session had loaded and the rollout had
started. The scale reaches the decode two ways, the constructor kwarg and the
ONNX `action_scale` metadata, and both were bare, so guarding one route only
would have let the same value in through the other. A declared value that is not
a number now names the policy and the field instead of raising a bare
`could not convert string to float: 'fast'`.

The bundle's velocity gate is `|twist| >= switch_on_velocity`, and a magnitude is
never negative, so the threshold has the same shape of failure in both
directions:

| `switch_on_velocity` | accepted | `|twist|=0.5` selects | `|twist|=0.0` selects |
| --- | --- | --- | --- |
| `0.1` (control) | yes | `walk` | `stand` |
| `nan` | yes | **`stand`** | `stand` |
| `inf` | yes | **`stand`** | `stand` |
| `-1.0` | yes | `walk` | **`walk`** |
| `-inf` | yes | `walk` | **`walk`** |
| `True` | yes (as `1.0`) | `stand` | `stand` |

A non-finite threshold can never select the move skill, because no magnitude
compares `>=` to it - a biped told to walk stands still. A threshold of `0` or
below can never select the idle skill - a biped told to stop keeps walking. Both
are reported as a successful tick.

All three routes now consult `positive_finite_number_error`, which is not a new
judgement: `WBCConfig` - the other ONNX locomotion provider, decoding with the
same `default_angles + action_scale * raw_action` formula - already holds its
identically-named `action_scale` to that domain, and `Policy.set_control_frequency`
in the base class states the same reason for its own rate ("`nan` and `inf` both
survive a bare `hz <= 0` test"). Usable values are unchanged: a float, an `int`,
a numpy float and a threshold as small as `1e-9` all still build and still gate
both ways, and omitting `switch_on_velocity` still leaves the gate off.

The two `action_scale` routes share one domain owner rather than two copies, and
the accompanying test derives the inventory of caller-supplied scalars from the
package's own annotations, so a third number arriving in this surface is graded
when it lands instead of inheriting an exemption by being absent from a list.
