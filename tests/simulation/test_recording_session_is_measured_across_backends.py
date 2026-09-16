"""Every backend measures the recording SESSION, not the dataset it resumed.

``stop_recording`` is shared: :class:`DatasetRecordingMixin` implements the whole
lifecycle once and the MuJoCo, Newton and Isaac backends mix it in unchanged.
Only ``start_recording`` is per-backend, and each of the three resolves
create-vs-resume itself and installs the recorder it built.

That split is why a session-scoped count cannot be a per-backend detail.
``DatasetRecorder.resume`` seeds ``frame_count`` / ``episode_count`` with the
dataset's totals, so the shared empty-capture guard reading them alone passes a
resumed session that captured nothing on the PREVIOUS sessions' frames and
reports an episode that was never written as saved. The counts a session starts
from fix that - but a backend that installs the recorder itself never stashes
them, and gets the wrong answer back out of code it does not contain.

Structural, because the divergence is: MuJoCo's ``start_recording`` runs under
``mujoco`` alone, while Newton's and Isaac's need ``warp`` / ``isaacsim``, so a
behavioural sweep would cover exactly the one backend that is easiest to get
right. What every backend can be held to without its engine installed is that it
arms the recorder through the one seam that stashes those counts, and tells the
caller when it resumed. Both populations are discovered from the source rather
than listed, so a fourth backend is held to the same contract the day it lands.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from types import SimpleNamespace

from strands_robots.simulation import recording as _recording
from strands_robots.simulation.recording import DatasetRecordingMixin

# The two methods that own the recorder handle: one arms it (and stashes the
# counts the session starts from), one releases it (and drops them).
_SEAM = {"_arm_dataset_recorder", "_release_dataset_recorder"}


def _package_root() -> pathlib.Path:
    return pathlib.Path(inspect.getfile(_recording)).parent.parent


def _functions_assigning_the_recorder() -> dict[str, int]:
    """``{"module.py:function": lineno}`` for each ``...["dataset_recorder"] = ...``."""
    found: dict[str, int] = {}
    root = _package_root()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Assign):
                    continue
                for target in inner.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "dataset_recorder"
                    ):
                        found.setdefault(f"{path.relative_to(root)}:{node.name}", inner.lineno)
    return found


def _functions_resuming_a_dataset() -> dict[str, str]:
    """``{"module.py:function": source}`` for each function that resumes a dataset."""
    found: dict[str, str] = {}
    root = _package_root()
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                body = ast.get_source_segment(source, node) or ""
                if "_DatasetRecorder.resume(" in body:
                    found[f"{path.relative_to(root)}:{node.name}"] = body
    return found


def _announces_a_resume(body: str) -> bool:
    """Whether ``body`` tells the caller the dataset is being appended to.

    The notice comes back from ``_arm_dataset_recorder`` when the session is
    armed with ``resumed=``, which is how the three backends keep one wording;
    a backend that resolved a resume and then armed without saying so is what
    this reads for. An inline sentence counts too - what is pinned is that the
    caller is told, not how it is spelled.
    """
    return "resumed=resume_existing" in body or "Resuming the existing dataset" in body


class TestOneSeamOwnsTheRecorderHandle:
    def test_no_backend_installs_the_recorder_itself(self) -> None:
        """Installing the handle directly is how a backend skips the stash.

        Newton and Isaac each did, in ``start_recording``, so the guard the
        shared ``stop_recording`` applies read a count nothing had written and
        both kept reporting an unwritten episode as a saved one.
        """
        found = _functions_assigning_the_recorder()
        assert found, "the scan found no assignment at all - it is not reading the recorder handle"
        adrift = {where: at for where, at in found.items() if where.split(":")[1] not in _SEAM}
        assert adrift == {}, f"these install the recorder outside the seam, skipping the stash: {adrift}"


class TestEveryBackendThatResumesSaysSo:
    def test_a_resuming_start_recording_tells_the_caller(self) -> None:
        """``start_recording`` read identically fresh or resumed on every backend."""
        resuming = _functions_resuming_a_dataset()
        assert len(resuming) == 3, f"expected one resuming start_recording per backend, got {sorted(resuming)}"
        silent = sorted(where for where, body in resuming.items() if not _announces_a_resume(body))
        assert silent == [], f"these resume a dataset without saying so: {silent}"


class TestTheSeamCarriesTheSessionCounts:
    def test_arming_stashes_what_the_dataset_already_held(self) -> None:
        recorder = SimpleNamespace(frame_count=19, episode_count=1)
        state: dict[str, object] = {}
        DatasetRecordingMixin._arm_dataset_recorder(state, recorder, resumed=True)
        assert state == {"dataset_recorder": recorder, "frames_at_start": 19, "episodes_at_start": 1}

    def test_a_resume_is_announced_with_the_counts_it_will_be_measured_against(self) -> None:
        line = DatasetRecordingMixin._arm_dataset_recorder(
            {}, SimpleNamespace(frame_count=19, episode_count=1), resumed=True
        )
        assert line.startswith("Resuming the existing dataset (1 episode(s), 19 frames);")
        assert "overwrite=True" in line

    def test_a_fresh_dataset_is_not_announced(self) -> None:
        assert DatasetRecordingMixin._arm_dataset_recorder({}, SimpleNamespace(frame_count=0, episode_count=0)) == ""

    def test_a_recorder_that_counts_nothing_still_opens_a_session(self) -> None:
        """The counters are read defensively, so a stand-in recorder still arms.

        Reading ``recorder.frame_count`` directly instead turns a recorder
        without the attribute into ``"Dataset init failed"`` from
        ``start_recording``'s own exception handler - a resume that dead-ends on
        the reply rather than on the data.
        """
        state: dict[str, object] = {}
        line = DatasetRecordingMixin._arm_dataset_recorder(state, object(), resumed=True)
        assert (state["frames_at_start"], state["episodes_at_start"]) == (0, 0)
        assert line.startswith("Resuming the existing dataset (0 episode(s), 0 frames);")

    def test_releasing_drops_everything_the_session_owned(self) -> None:
        state: dict[str, object] = {"trajectory": [{"t": 0}]}
        DatasetRecordingMixin._arm_dataset_recorder(state, SimpleNamespace(frame_count=19, episode_count=1))
        DatasetRecordingMixin._release_dataset_recorder(state)
        assert state == {"dataset_recorder": None, "trajectory": []}
