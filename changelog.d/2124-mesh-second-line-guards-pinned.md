### Quality: pin the two second-line guards on the mesh wire-authorisation path

`validate_command`'s control-character post-check on `policy_host` and
`_load_acl_file`'s `O_NOFOLLOW` refusal both exist for a stated reason: the
first check in front of each one can be widened or raced, and the second is
what keeps the boundary closed when it is. Both were unreached by the suite,
so either could have been removed with every test green.

Three assertions could not tell the two layers apart and are now exact:
`test_rejects_crlf_in_policy_host` accepted either layer's message;
`test_acl_load_refuses_symlink` matched a lowercase `symlink` that pytest's
own `tmp_path` supplies, so it passed for any `ValueError` naming the path;
and `test_validate_command_finite_numerics`'s module docstring credited
`_coerce_int`'s `int(...)` wrap for refusals its explicit guards make. Tests
only -- no library behaviour changes.
