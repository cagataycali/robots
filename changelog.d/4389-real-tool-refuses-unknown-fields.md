### Fixed: the real arm tool refuses an input field outside its schema

`Robot.stream` refuses a key its action does not take, by name and with the valid list, before the operator gate; it used to drop the key silently, so a misspelt `policy_config` ran the default configuration on a real arm. The sim tool and the real tool now build that sentence with one helper. (#4167)
