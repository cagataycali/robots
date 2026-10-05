### Fixed: `send_action`'s unknown-key warning no longer says the batch went through

Since #4486 a MuJoCo `send_action` batch with one unknown key is refused whole,
and the returned error says "Nothing was applied". The WARNING logged for the
same call still said "The value was dropped.", which reads as if only the bad
key was skipped and the rest landed. It now says "The whole batch was refused
and nothing was written." The action-controller fallback, which really does
write the other keys and drop just the bad one, keeps "The value was dropped."
