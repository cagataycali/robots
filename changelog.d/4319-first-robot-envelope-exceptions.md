### Docs: First robot names the two calls that do not return the envelope

`docs/start/first-robot.md` said every method in its table returns the
`status`/`content` envelope. `get_observation()` returns the flat observation
dict and `cleanup()` returns `None`, so `robot.cleanup()["content"]` raised
`TypeError`. The sentence now names both, and a test runs every call in the
table on a sim robot and requires the sentence to name exactly the ones that
return no envelope.
