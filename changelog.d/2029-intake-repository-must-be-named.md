### CI: the duplicate-claim intake check refuses an inferred repository

`scripts/check_duplicate_claim.py --issue N` resolved the repository from
`$GITHUB_REPOSITORY`, which names where the command is *running*. AGENTS.md step 1 asks
that question at intake -- before any pull request exists -- so it is a local invocation
by whoever is about to do the work, and nothing ties their working directory to the
repository the issue belongs to. Run from elsewhere, the check read a different
repository's open pull requests, found none of them claiming the number, and reported
`unique-claim` with exit `0`. Measured with `huggingface/lerobot` as the ambient
repository, `--issue 2029` compared **405** unrelated open pull requests; naming the
repository compared 4. Both said no duplicate, so the wrong answer was indistinguishable
from the right one, and it only misled in the reassuring direction -- a spurious
collision would have been investigated and found nonexistent, a missed one is invisible.

Intake mode now requires `--repo`, refusing at argparse time with a message that names
what the environment inferred, before any lookup. Nothing can detect the substitution
after the fact: an issue number alone does not name a repository, so there is no second
source to compare against, and numbers are dense enough that an unrelated repository
very often has one at the same number -- which is also why confirming the issue *exists*
would not be a reliable substitute, on top of reversing this script's deliberate
decision not to read it. The `--pr` mode keeps the environment default, since a workflow
reviewing a pull request runs where that pull request lives. Step 1 prints the explicit
command, and a test runs the printed command through the script's own preconditions, so
shortening one without the other fails.
