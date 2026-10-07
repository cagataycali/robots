"""Repro: README L94 "Storage Buckets" is HF-only.

README L94 reads:

> | **Teleoperate and record** LeRobotDataset episodes from leader arms,
> | gamepads or WASD; stream to HF datasets or **Storage Buckets** | ...

The capitalized plural "Storage Buckets" after the comma-separated
alternative "HF datasets" reads to a new user as two destination FAMILIES:

  1. "HF datasets" (= HuggingFace Datasets Hub, mentioned by name), AND
  2. "Storage Buckets" (= generic cloud blob storage: AWS S3, GCS, R2, Azure)

Reality: both routes are *HF-only*. The only sync entry point is
``strands_robots.dataset_transfer.sync_dataset_to_bucket``, which refuses any
URL-shaped bucket argument because its allowlist is HF-repo-id-shaped
(``"name"`` or ``"org/name"``, ``[A-Za-z0-9._-]`` only). The implementation
shells out to the ``hf`` CLI (``huggingface_hub >= 1.5``) and uses an
``hf://buckets/...`` URI. No ``boto3`` is imported anywhere in
``strands_robots/dataset_transfer.py`` or ``strands_robots/dataset_recorder.py``;
no GCS / Azure client either.

The docs page the row links to (``docs/learn/data/stream-and-sync.md``) is
unambiguous: it writes "HF Storage Bucket" (singular, prefixed) in every
occurrence. The README is the only surface that writes "Storage Buckets" bare,
plural, after "HF datasets".

Run with:

    MUJOCO_GL=egl python bugbash_repros/readme_l94_storage_buckets_hf_only_repro.py

No sim, no network, no auth required - the refusal fires in the input-
validation allowlist, before the ``hf`` CLI is even located.
"""

from __future__ import annotations

import inspect
import sys


def main() -> int:
    from strands_robots.dataset_transfer import sync_dataset_to_bucket

    # Signature trace: `bucket: str`, no scheme/protocol parameter.
    sig = inspect.signature(sync_dataset_to_bucket)
    assert "bucket" in sig.parameters, sig
    print(f"[1] sync_dataset_to_bucket{sig}")

    # Candidate values a README-literal reader would try. The first two are
    # the "obvious" interpretations of "Storage Buckets" when read as the
    # second item of "HF datasets or Storage Buckets".
    candidates = [
        ("s3://my-bucket/datasets", "AWS S3"),
        ("gs://my-bucket/datasets", "Google Cloud Storage"),
        ("https://my-bucket.s3.amazonaws.com/", "signed S3 URL"),
        ("azure://my-container/datasets", "Azure Blob"),
    ]

    rejected = 0
    for bucket, label in candidates:
        result = sync_dataset_to_bucket(
            "/tmp/nonexistent_bugbash_root_do_not_create",
            bucket,
            run_id="x",
        )
        status = result.get("status") if isinstance(result, dict) else None
        message = result.get("message", "") if isinstance(result, dict) else ""
        assert status == "error", (
            f"{label} bucket {bucket!r}: expected status=error, got {result!r}"
        )
        # The refusal MUST cite the allowlist shape so the mismatch is readable.
        # Current message:
        #   invalid bucket 's3://my-bucket/datasets': must match 'name' or
        #   'org/name' using [A-Za-z0-9._-] (no path traversal or shell metacharacters).
        assert "'name' or 'org/name'" in message, message
        print(f"[2.{rejected + 1}] {label:28s} -> {message[:90]}...")
        rejected += 1

    assert rejected == len(candidates), rejected
    print(f"[3] All {rejected} non-HF bucket URIs are refused by the allowlist")

    # The docstring itself names the scope precisely - the README is the only
    # surface that doesn't inherit this scoping word.
    doc = inspect.getdoc(sync_dataset_to_bucket) or ""
    assert "HF Storage Bucket" in doc, doc[:400]
    print("[4] Function docstring scopes the API: 'HF Storage Bucket' (singular, prefixed)")

    # And the import graph confirms the absence of alternatives.
    import strands_robots.dataset_transfer as mod

    src = inspect.getsource(mod)
    for forbidden in ("boto3", "google.cloud.storage", "azure.storage"):
        assert forbidden not in src, f"unexpected import in dataset_transfer: {forbidden}"
    print("[5] dataset_transfer imports no boto3 / google-cloud-storage / azure-storage")

    # Final: show what a README-literal reader has as their mental model vs.
    # what the project actually supports.
    print()
    print("README L94 reads:  'stream to HF datasets or Storage Buckets'")
    print("User expects:      HF datasets OR S3 / GCS / Azure / R2")
    print("Project supports:  HF datasets only; 'Storage Buckets' == 'HF Storage Buckets'")
    print()
    print("Fix: scope the noun on L94 the same way the docs page already does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
