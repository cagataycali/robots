### Fixed: an empty `urdf_path`/`data_config` names that parameter instead of diagnosing the robot name

`MuJoCoSimEngine.add_robot` read its model source by truthiness, so a source the caller
supplied but left empty was indistinguishable from one they omitted. `urdf_path=""` took
the absent branch, ran the deprecated name-as-registry-key fallback for a call that had
asked for a file, and the refusal then diagnosed the name: it offered close-match
suggestions for a name the caller never asked to resolve, and advised them to "pass
data_config=<registered model> or urdf_path=<file>" - the kwarg they had just passed. The
message was byte-identical to what supplying no source at all returns, so the report could
not tell the two apart, while one whitespace character away (`urdf_path=" "`) the diagnosis
was already correct ("File not found").

A supplied-but-empty model source is now refused naming that parameter, the way
`register_urdf` already refuses an empty `urdf_path` and the way the Newton
(`urdf_path is not None`) and Isaac (`usd_path is None`) backends already read their own
model source - MuJoCo's `add_robot` was the one site in the family reading it by
truthiness. The guard sits after the existing name-taken check, so a caller who both
reuses a label and empties the path is still told about the label, and whitespace stays a
path rather than an empty value.
