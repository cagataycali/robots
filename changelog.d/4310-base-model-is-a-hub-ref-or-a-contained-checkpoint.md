### Fixed: `base_model` is a Hub reference or a checkpoint inside a known home

Every trainer's preflight now refuses a `base_model` that is neither a Hub
reference (`name` or `org/name`, optionally `@revision`) nor a path inside the
training output home or the Hub cache. Before, only a leading dash was refused,
so an agent could hand a trainer `/etc` or a `..` path and read the disk
through the backend's loader error. The refusal names the homes and never
whether the target exists. (f027)
