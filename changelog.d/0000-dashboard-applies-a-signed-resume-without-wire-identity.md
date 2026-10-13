### Fixed: the dashboard applies a signed resume that reaches it over AWS IoT Core

The dashboard now applies a fleet resume that arrives with no wire identity
(AWS IoT Core carries no Zenoh `source_info`) when its signed assertion passes
the same check the robots run: the operator's key, the dashboard's rail peer id
among the targets, the `estop_id` of the lockout the dashboard recorded, and
the freshness window. Before, every resume over IoT Core was left unattributed,
so the fleet badge stayed LOCKED after every robot had cleared. A resume that
fails the check still leaves the lockout standing, and the trail records why
under `assertion_refused`.
