### Fixed: `train_policy` refuses an unknown action before anything else

Calling `train_policy` with an action it does not know (`"lsit"`, `"fine_tune"`) used to be answered with "a data source and output_dir are required", and, once you supplied those, with whatever the trainer build reported. The action name is now graded first, so a mistyped action gets the refusal that names the five actions the tool answers.
