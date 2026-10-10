### Tests: eight refusal suites read each refusal once per surface

The native-driver refusal, the recording pre-flight, the MuJoCo entity-name
lookup, the creation-time entity and reserved camera names, the object size,
the camera pixel count and the conversion-escape suites gave every probe value,
and in two of them every facet of one refusal message, its own cell. Each
surface now builds its refusal once, loops the probe table, and names the
failing value in the assertion. The eight files go from 1,160 cells to 419
with the same package lines executed.
