### Fixed: the state-key mismatch diagnostic recommends an embodiment that binds the observation

Both `lerobot_local` state-key diagnostics ended in one fixed sentence for every
caller, offering `embodiment='so101'` as the example. On a real SO arm that
advice is a loop: lerobot's `SOFollower` reports `'<motor>.pos'` keys, while the
`so101` configuration declares the MuJoCo asset's numeric joints `'1'..'6'`, so
following it lands back on the same all-missing guard that printed it - and its
`state_units='degrees'` would convert units the hardware reports natively. The
configuration that does bind that observation, `so_real`, was never named.

The remedy is now derived from the observation. New public
`matching_embodiments(observation_keys)` returns every shipped embodiment whose
entire `state_keys` set the observation carries; `state_key_remedy` names the one
match, lists all of them when the observation cannot distinguish them (the real
SO, Koch and OMX arms report identical `.pos` keys), and offers no embodiment at
all when none matches. `set_robot_state_keys([...])` is always offered, quoting
the observed keys verbatim when short enough to paste. Both the all-missing and
partial-missing guards use the one helper, so their advice cannot drift.
