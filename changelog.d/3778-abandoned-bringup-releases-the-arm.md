### Fixed: a real-robot bring-up that is abandoned no longer leaves the arm locked

`execute`/`start` connect the arm before the policy is built, and connecting
turns torque on. When the bring-up then ends without ever commanding the arm -
the policy cannot be built or initialized, or a stop/shutdown is latched at one
of the two stage gates - the arm this task connected is disconnected again
(torque released) and the reply says so. A connection made before the task is
left alone, and a rollout that fails while running still holds its pose.

Each of those exits is a terminal state, so it also settles the elapsed time:
an abandoned bring-up reports the seconds it really spent instead of `0.0s`.
