"""Export must forward base_model - the spec revalidates on export, so a
provider whose training REQUIRED a base model (smolvla post-tune, GR00T)
refused to export its own finished run when the dashboard dropped the field.
Found live: deploy button answered 'base_model is required' for a job whose
record carried base_model the whole time.

Run with --no-cov.
"""

from unittest import mock

from strands_robots.dashboard import training


def test_export_forwards_base_model():
    with mock.patch("strands_robots.tools.train_policy.train_policy") as tp:
        tp.return_value = {"content": [{"text": "ok"}], "status": "success"}
        training.export("mock", "/out", "/data", None, "lerobot/smolvla_base")
    kwargs = tp.call_args.kwargs
    assert kwargs["base_model"] == "lerobot/smolvla_base"
    assert kwargs["action"] == "export"


def test_export_without_base_model_sends_empty_string():
    # ACT-from-scratch trains with base_model="" - export must mirror that
    with mock.patch("strands_robots.tools.train_policy.train_policy") as tp:
        tp.return_value = {"content": [{"text": "ok"}], "status": "success"}
        training.export("mock", "/out", "/data")
    assert tp.call_args.kwargs["base_model"] == ""
