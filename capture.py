"""Capture the sweep's live finding plus the per-branch signals that miss it.

Prints its own tree first so every number is attributable.
"""
from __future__ import annotations
import itertools, json, os, pathlib, sys, time, urllib.error, urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import check_merge_base_overlap as mbo  # noqa: E402

print("TREE:", pathlib.Path(mbo.__file__).resolve().parents[1])
REPO, TOKEN = "strands-labs/robots", os.environ["PAT_TOKEN"]
OUT = pathlib.Path(sys.argv[1])
facts: dict = {"tree": str(pathlib.Path(mbo.__file__).resolve().parents[1]), "repo": REPO}

def save() -> None:
    OUT.write_text(json.dumps(facts, indent=2), encoding="utf-8")

def gql(query: str, variables: dict) -> dict:
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as fh:
        return json.load(fh)

# --- the sweep, run exactly as documented -----------------------------------
rows, unevaluated = mbo.collect_open_pull_requests(REPO, "main", TOKEN)
facts["open_non_draft"] = len(rows)
facts["pairs"] = len(rows) * (len(rows) - 1) // 2
facts["unevaluated"] = [{"number": n, "reason": r} for n, r in unevaluated]
facts["prs"] = [
    {"number": r.number, "merge_base": r.merge_base[:8],
     "behind_by": r.behind_by, "head_paths": len(r.edits),
     "base_paths": None if r.landed_since is None else len(r.landed_since)}
    for r in rows
]
pairs = mbo.pair_overlaps(rows)
stale = mbo.stale_base_overlaps(rows)
facts["pair_findings"] = [
    {"left": a, "right": b, "blocking": sorted(bl), "prose": sorted(pr)} for a, b, bl, pr in pairs
]
facts["stale_findings"] = [
    {"number": n, "blocking": sorted(bl), "prose": sorted(pr), "behind_by": bb} for n, bb, bl, pr in stale
]
save()

# --- the full pair matrix (for the heat panel) -------------------------------
by_num = {r.number: r for r in rows}
matrix = []
for left, right in itertools.combinations(sorted(by_num), 2):
    shared = mbo.overlapping_paths(by_num[left].edits, by_num[right].edits)
    blocking, prose = mbo.partition_overlap(shared)
    matrix.append({"left": left, "right": right, "n_blocking": len(blocking), "n_prose": len(prose)})
facts["matrix"] = matrix
save()

# --- the per-branch signals, for the pair the sweep reports ------------------
blocking_pairs = [p for p in facts["pair_findings"] if p["blocking"]]
assert blocking_pairs, "no blocking pair found; the artifact has nothing to show"
focus = sorted({blocking_pairs[0]["left"], blocking_pairs[0]["right"]})
facts["focus"] = focus
Q = """
query($owner:String!,$name:String!,$number:Int!){
  repository(owner:$owner,name:$name){ pullRequest(number:$number){
    number title mergeable mergeStateStatus reviewDecision
    baseRefName headRefOid isDraft } } }
"""
owner, name = REPO.split("/")
per_branch = []
for number in focus:
    pr = gql(Q, {"owner": owner, "name": name, "number": number})["data"]["repository"]["pullRequest"]
    for _ in range(6):  # UNKNOWN means GitHub is still computing; the read above triggers it
        if pr["mergeStateStatus"] != "UNKNOWN":
            break
        time.sleep(6)
        pr = gql(Q, {"owner": owner, "name": name, "number": number})["data"]["repository"]["pullRequest"]
    row = by_num[number]
    # what the SINGLE-BRANCH check reads: paths main gained since M, intersected with the branch's own edits
    base_side = row.landed_since if row.landed_since is not None else set()
    single = mbo.overlapping_paths(row.edits, base_side)
    per_branch.append({
        "number": number, "title": pr["title"][:52],
        "mergeable": pr["mergeable"], "mergeStateStatus": pr["mergeStateStatus"],
        "reviewDecision": pr["reviewDecision"] or "awaiting-first-review",
        "merge_base": row.merge_base[:8], "behind_by": row.behind_by,
        "base_side_paths": None if row.landed_since is None else len(row.landed_since),
        "single_branch_overlap": sorted(single),
        "single_branch_verdict": "overlap" if single else "No overlap",
    })
facts["per_branch"] = per_branch
save()

# --- the shared path, and what the sweep says about it ----------------------
facts["shared_paths"] = blocking_pairs[0]["blocking"]
save()

# --- assertions: the artifact must not claim more than was measured ---------
assert facts["open_non_draft"] >= 2
shared = set(facts["shared_paths"])
assert shared, "the pair must share a behaviour-bearing path"
for entry in per_branch:
    named = shared & set(entry["single_branch_overlap"])
    entry["names_a_shared_path"] = sorted(named)
    assert not named, (
        f"#{entry['number']}'s own single-branch run already names {sorted(named)}, "
        "so the pairwise finding is not invisible to it after all"
    )
# The pull request that produced the pairwise finding is one the stale-base mode had to
# exclude for a capped base-side set: dropping it wholesale would have lost this finding.
capped = {row["number"] for row in facts["unevaluated"]}
facts["capped_but_still_paired"] = sorted(capped & set(focus))
save()
print(json.dumps({k: facts[k] for k in ("open_non_draft", "pairs", "focus", "shared_paths")}, indent=2))
print("per-branch:", json.dumps(per_branch, indent=2))
print("stale:", json.dumps(facts["stale_findings"], indent=2))
print("unevaluated:", facts["unevaluated"])
