### Fixed: `send_action` refuses a batch with an unknown key whole, before anything moves

On MuJoCo, Newton and Isaac, `send_action({"1": 0.5, "elbow": 0.3})` wrote the
key that resolved, stepped physics, and only then answered `status="error"`
naming `elbow` - so the error described a world that had already moved, and a
caller retrying on it stroked joint `1` twice. Every key is now resolved first:
one unknown key writes nothing, the world does not advance, and the refusal's
`json` block carries `unresolved_keys` with an empty `applied`. Policy rollouts
keep driving a policy that emits a key the robot lacks: the runner resends the
keys that resolve and reports the rest, as before.
