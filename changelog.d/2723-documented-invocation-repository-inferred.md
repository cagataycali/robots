### Docs: every documented check invocation names the repository it reads, not only the fenced ones

`AGENTS.md` spelled two check invocations inline -- step 1's merge-base overlap
sweep and the last-push-approval split -- in a form that leaves the repository to
`$GITHUB_REPOSITORY`. Run from a scheduled agent, whose checkout need not be this
repository, step 1's spelling reported a clean open set for a repository holding
none of the pull requests in question and exited 0: `cagataycali/strands-gtc-nvidia`,
0 open pull requests, where the named repository had 2.

`TestNoDocumentedInvocationLeavesTheRepositoryInferred` exists to prevent exactly
that, but selected only mentions carrying a `python3` prefix. The prefix is
typography rather than a property of the command -- a reader copies what is inside
the backticks -- and requiring it hid 2 of the 13 documented invocations, both of
which were the defect. The selector now grades a mention that names any option
regardless of prefix, and skips one that names none, since that is a
cross-reference rather than a command.
