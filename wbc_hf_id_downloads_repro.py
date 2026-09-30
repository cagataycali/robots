#!/usr/bin/env python3
"""
Repro: docs/robots/unitree_g1.md:33 claims a HuggingFace id is *refused* for
`wbc` checkpoints ("A HuggingFace id is refused (#4161): pass the local
directory"). In fact, the WBCPolicy constructor eagerly runs
`_maybe_download_checkpoint`, which calls
`huggingface_hub.snapshot_download(repo_id=<id>, allow_patterns=["*.onnx","*.json"])`
BEFORE inspecting the checkpoint. For `nvidia/GEAR-SONIC` -- the public NVIDIA
repo that shows up top of any HuggingFace search for "GEAR-SONIC WBC" -- this
downloads ~2.5 GB of ONNX weights (planner_sonic.onnx is 774 MB alone) and
only THEN raises "checkpoint looks like the SONIC VLA inference stack".

That is: the docs' "refused" claim covers a code path that DOES touch the
network, DOES burn 2.5 GB of bandwidth (per invocation on a cache-cold host,
per CI job on ephemeral runners), and only THEN reports the mismatch. The
error message never tells the user a large download just happened.

Impact:
- CI on ephemeral runners: 2.5 GB / job on every mistake.
- Metered / mobile / satellite bandwidth: the docs said "refused", the code
  said "sure, let me grab that".
- Time-to-error: on a slow link, tens of minutes before the user learns the
  checkpoint was wrong -- and the docs said they'd learn instantly.

Both of the marker files needed to refuse this pre-download are already
tabulated in the same module:
  `_SONIC_INFERENCE_STACK_FILES = frozenset({"model_encoder.onnx",
    "model_decoder.onnx", "planner_sonic.onnx"})`
So the fix is: for HF-shaped ids, list the repo's file tree first
(`huggingface_hub.HfApi().list_repo_files`, a metadata-only call, no
weights), and if the tree contains any SONIC marker file, raise the same
"this is the SONIC VLA inference stack, not decoupled-WBC" error the
post-download path already emits -- BEFORE `snapshot_download` runs.

Cite: strands_robots/policies/wbc/policy.py:1103-1160 (_maybe_download_checkpoint
downloads first), :958-1101 (_reject_sonic_stack_marker fires only after
download completes), :135-139 (marker files known to the module),
docs/robots/unitree_g1.md:33 ("A HuggingFace id is refused (#4161)").

Reproduces on a cache-cold host, requires network + huggingface_hub. Run:
    python /tmp/wbc_hf_id_downloads_2gb_repro.py

Prints the pre-call and post-failure HF cache footprint. The delta shows how
much bandwidth the "refused" call actually consumed.
"""
import os
import sys
import shutil
import time

def hf_cache_size():
    """Return HF hub cache footprint (bytes), resolving symlinks."""
    root = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--GEAR-SONIC")
    if not os.path.isdir(root):
        return 0
    seen = set()
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            p = os.path.join(dirpath, f)
            real = os.path.realpath(p)
            if real in seen or not os.path.exists(real):
                continue
            seen.add(real)
            total += os.path.getsize(real)
    return total

def main():
    from strands_robots import Robot

    pre = hf_cache_size()
    print(f"[pre] HF cache for nvidia/GEAR-SONIC: {pre/1e6:.1f} MB")

    g1 = Robot("g1")
    t0 = time.time()
    try:
        g1.run_policy(
            policy_provider="wbc",
            policy_config={"checkpoint": "nvidia/GEAR-SONIC"},
            duration=0.05,
            control_frequency=50,
        )
        print("[unexpected] call succeeded")
        return 1
    except RuntimeError as e:
        elapsed = time.time() - t0
        post = hf_cache_size()
        delta = post - pre
        msg = str(e)
        print(f"[post] HF cache for nvidia/GEAR-SONIC: {post/1e6:.1f} MB "
              f"(delta {delta/1e6:+.1f} MB)")
        print(f"[time] {elapsed:.1f}s until error was raised")
        print(f"[err ] {msg[:200]}")
        # Assert the paper-cut (pre-fix behaviour): a "refused" call materialised
        # weights on disk. After the fix, delta stays at 0 because the preflight
        # metadata call (huggingface_hub.HfApi().list_repo_files) sees the SONIC
        # markers before snapshot_download runs; the same exception is raised
        # with (a) an added "Refusing before download to save bandwidth" clause
        # and (b) no ~2 GB of ONNX weights in the HF cache.
        if delta > 100 * 1024 * 1024:  # >100 MB downloaded is the smoking gun
            print(
                "\nDEFECT CONFIRMED (pre-fix): docs/robots/unitree_g1.md:33 says "
                "'A HuggingFace id is refused'; this call refused only AFTER "
                f"downloading {delta/1e6:.0f} MB in {elapsed:.0f} s."
            )
            return 0
        else:
            if "before download" in msg.lower() or "refusing before" in msg.lower():
                print(
                    "\nFIX VERIFIED: refusal happens pre-download "
                    f"(delta {delta/1e6:.1f} MB, {elapsed:.1f}s), and the error "
                    "message names the reason without a wasted fetch."
                )
                return 0
            print("\nno large download observed (cache warm? fix applied?)")
            return 2

if __name__ == "__main__":
    sys.exit(main())
