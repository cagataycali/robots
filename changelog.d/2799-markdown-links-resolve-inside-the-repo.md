### Fixed: a Markdown link that climbs out of the repository

`strands_robots/policies/moveit2/server/README.md` pointed at the MoveIt2 integration test with a
five-segment `../../../../../` prefix. The page sits four directories deep, so the target resolved
one level above the checkout and named nothing: a 404 for every reader who clicked it, on GitHub and
in an editor alike. The prefix is now four segments and resolves to the file it names.

Nothing in the repository graded that. `mkdocs build --strict` is the only link checker here, and it
is narrower than it looks in two ways. It resolves links only for files under `docs_dir`, so a
Markdown file anywhere else - `README.md`, `AGENTS.md`, a `changelog.d` fragment, a package
reference page such as this one - is read by no checker at all. And the `build` job that runs it is
not among the branch ruleset's required status checks, so even a broken link *under* `docs/` cannot
block a merge; it surfaces afterwards on `main`, where the same job's failure holds back the Pages
deploy that depends on it.

So `tests/test_markdown_links_resolve.py` puts the question inside the suite the required check
runs, and grades every Markdown file in the tree rather than the `docs/` subset: 896 files, 568 link
targets of which 404 are relative, one offender before this change and none after. Its rule is that a relative target
must resolve to a path that exists *and* that lies inside the repository. The second half carries
its own weight - a target that climbs out may happen to land on something real beside one clone and
on nothing beside another, so existence alone is a verdict about the machine rather than about the
link.

Fragment resolution stays with MkDocs, which owns the slugifier that decides whether a heading
anchor exists; re-deriving those rules here would be a second implementation of somebody else's
contract. Site-absolute targets are left alone too: MkDocs reads them as site-root-relative and
GitHub as repository-root-relative, so they need a policy rather than a resolution, and the tree
ships none.
