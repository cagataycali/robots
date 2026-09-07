### Bug Fixes

- **tools/robot_mesh**: four sites decoded the caller-supplied `command` string
  behind `except json.JSONDecodeError`, which covers malformed syntax and
  nothing else. `json.loads` also raises a plain `ValueError` on a number wider
  than `sys.get_int_max_str_digits()` - well-formed JSON, since RFC 8259 bounds
  no range, that this build cannot construct an `int` for - and a
  `RecursionError` on a document nested past the interpreter stack, which is a
  `RuntimeError` outside `ValueError` entirely. Both escaped every site: on the
  two pre-pass sites inside the tool as an exception raised past its own
  `{"status": "error"}` envelope with no audit row for a refused `broadcast` or
  `send`, though every refusal beside them writes one; on the two Device Connect
  sites absorbed by the dispatcher's outermost handler and reported as a
  transport error it never dialled. One `_decoded_command_body` owner now
  decodes every body and raises `ValueError` naming the cause, mirroring
  `mesh._acl_config._parse_json5`, which converts the same `RecursionError` for
  the same reason. Each cause reads differently, because a body nested too
  deeply or carrying a number too wide is valid JSON this build cannot read and
  reusing the syntax wording would send an operator hunting a syntax error that
  is not there. The two spellings the narrow handler always caught keep
  byte-identical wording.
