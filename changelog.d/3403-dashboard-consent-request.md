### Added: the dashboard turns a refusal into a consent request by its code, not its wording

`strands_robots.dashboard.consent.classify_refusal` takes a refusal the SDK raised (or its wire shape, or the agent-motion verdict) and returns a `ConsentRequest` naming what the operator can grant: the environment variable comes from `refusal_codes.REFUSAL_GRANTS`, the subject from the refusal itself, and the message is shown, never parsed. A refusal without a code is not continuable and gets no card. Teleop slew refusals carry no code yet, so they are not offered.
