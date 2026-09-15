### The inert-normalization diagnostic now names the stats the checkpoint ships

A base checkpoint whose normalizer stats are keyed by the training dataset
(`so100.buffer.action`) leaves `action`/`observation.state` un-normalized, and
the load-time warning prescribed supplying stats without saying where to get
them -- so a caller had to open the checkpoint's own
`*_normalizer_processor.safetensors` to discover the remedy was already on disk.
The warning now lists the prefixed spellings found per missing key
(`lerobot/smolvla_base`: `{'observation.state': [], 'action':
['so100-blue.buffer.action', 'so100-red.buffer.action', 'so100.buffer.action']}`),
including an empty list when a feature genuinely has no candidate. They are
listed rather than adopted because several prefixes with different distributions
can be present, so picking one would be a silent guess.
