### Fixed: a provider typo with a slash or colon gets "Did you mean", not a `lerobot_local` refusal

`create_policy` treated any string containing `/` or `:` as an address or a
checkpoint, so `"wbc/"`, `"protomotions:"`, `":"` or a bare `host:port` were
forwarded to `lerobot_local` as a HuggingFace id: the caller got a
trust-remote-code refusal naming a provider they never typed, and
`provider_can_be_created` reported them buildable. Only a `scheme://` URL, a
scheme-less address a provider declares in `url_patterns`, an `org/repo` id or a
filesystem path with a name in it is read that way now; anything else is looked
up as a provider name and refused with the nearest spellings.
