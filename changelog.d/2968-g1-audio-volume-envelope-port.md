### Added

- `strands_robots.tools.g1.g1_audio_volume_envelope` ports the read-only
  envelope half of neon-the-g1's `g1_audio` volume-side into two agent-facing
  lookups: `g1_list_audio_volume_envelope` (name the observed `0-100` clamp
  the SDK's `AudioClient.SetVolume` accepts today) and `g1_volume_admits`
  (decide one query, refusing below-floor, above-ceiling, bool-masquerading-
  as-int, and non-int-non-bool values with the `7404` gate-refusal code the
  driver's own `_check_motion_gates` quotes). Read-only, no driver instance,
  no DDS, no SDK: `import strands_robots.tools.g1.g1_audio_volume_envelope`
  pulls no `unitree_sdk2py` submodule. Refs #358.
