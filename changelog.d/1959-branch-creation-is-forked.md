### Docs: PR Workflow step 1 names the rule that forces a fork

`AGENTS.md` opened its PR Workflow with "Create feature branch from `main`",
which this repository does not permit. The `default` ruleset's conditions are
`ref_name.include: ["~ALL"]` rather than the default branch alone, and its rules
include `creation` with `bypass_actors: []`, so `git push <base>
HEAD:refs/heads/<new>` is refused for every account. The remaining steps already
assumed a fork -- step 5 read "Open PR from your fork" -- so the file described a
first step the rest of it did not use.

What made it worth naming rather than leaving to be rediscovered is that the
refusal is unattributed: GitHub answers with `push declined due to repository
rule violations` and does not say which rule, which makes it indistinguishable
from the two refusals the file does describe -- a token missing a permission, and
the `.github/workflows/**` write refusal that makes an installation token read
`BLOCKED`. Both of those are answered by presenting a wider token, so that is the
natural next move and it cannot work here: a ruleset bypass is granted per
ruleset, so no role carries one, and there is no classic branch protection for an
account to be exempt from.

Step 1 now says to branch on the fork, names the `creation` rule, the `~ALL`
scope and the empty bypass list, shows the `GET .../rulesets/{id}` read that
settles it, and points at the cross-repo remedy (`createPullRequest` takes the
base repository as `repositoryId` and the fork as a separate
`headRepositoryId`). `tests/test_ref_creation_ruleset_scope.py` pins both halves:
it implements the refusal derivation over the published ruleset payload -- so a
bypass actor added later drops the refusal and fails the pin rather than leaving
the guidance reading plausibly -- and asserts the rule, its scope, the empty
bypass list and the remedy stay inside step 1 rather than drifting out of it.
