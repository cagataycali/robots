### Fixed: two tools published `Parameter <name>` where their docstring gives a description

`g1_decode_error_code` and `g1_get_state` each began a docstring line with an RST role
(`:func:` / `:data:`), which `docstring_parser` reads as a REST field. Its style
auto-detection then scored REST equal to Google on those docstrings and broke the tie in
REST's favour, so the Google `Args:` block was not read at all and `strands.tool` fell
back to the literal `"Parameter code"` / `"Parameter driver"` in the published schema: a
model choosing a value for either parameter was told nothing about it, while the sentence
that describes it sat in the source. Both docstrings are rewrapped so no line begins with
a role - the same words, whitespace only - and the schema comparison above now fails for
any tool that loses its descriptions this way.
