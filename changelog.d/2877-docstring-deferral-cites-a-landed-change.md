### Fixed: a docstring that defers pending work no longer cites a change this repository already landed

The regression guard added for the caller-reachable half of this contract scoped
itself to strings, on the grounds that a developer-facing docstring legitimately
cites historical work and a maintainer reading one has the git history to hand.
That reason describes a *backward citation*. It does not describe a *forward
deferral*, which the same module identifies as the opposite speech act: a
deferral tells the next contributor a capability is still outstanding and where
to watch for it, and git history cannot answer "when will this land".

Seven docstring sentences in `strands_robots/drivers/g1.py` deferred to `#358`
and `#361`. Both resolve, in this repository, to merged pull requests about
unrelated subsystems, so a reader following one found a landed change and could
not tell whether the capability was missing or the note was stale. One sentence
was worse than misdirecting: `send_action` said "the loop lands in the follow-up
PR that closes issue #361 in full" while `_ControlLoop` ships in the same
module, describing shipped code as future work.

The two conditions now also grade docstring sentences. The scope is the sentence
rather than the docstring, because one paragraph routinely carries a deferral and
a backward citation and a paragraph-wide read blames the deferral for the
credit's reference. The deferral vocabulary gains "lands in" and "future work",
which - measured across the package - adds no caller-reachable offender and takes
the docstring rule from two of the seven sentences to all seven. Each fixed
sentence names the missing capability instead of a tracker number, matching the
remedy the string half already applies. `#` comments stay out of scope: a comment
is neither read by a caller nor part of the rendered API surface.
