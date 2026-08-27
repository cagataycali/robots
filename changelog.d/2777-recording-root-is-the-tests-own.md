### Tests: a recording test's dataset root must be one it owns, not just one it names

`tests/test_recording_root_is_not_the_shared_cache.py` requires every recording
call that names a `repo_id` to name a `root`, so a unit test cannot resolve its
dataset directory out of `$HF_LEROBOT_HOME` and inherit whatever the host's cache
already holds. The rule read the keyword and not the value it carried, and those
are different claims: `root="/tmp/ds"` is not the shared cache, and it is shared
all the same - with every other test that names it, and with every process on the
machine that can write `/tmp`.

That was live rather than theoretical.
`tests/test_dataset_recorder.py::test_create_passes_vcodec_directly_when_supported`
named `/tmp/ds`, and `DatasetRecorder.create` resolves and inspects that
directory through `_prepare_dataset_target` before the injected fake dataset
class is reached. On a host where `/tmp/ds/meta` existed the test failed with
`FileExistsError: A LeRobotDataset already exists at /tmp/ds` - a filesystem
verdict on a test whose subject is codec normalisation and which writes nothing,
and which passes with no other change once the same call points at a fresh
directory. The immediately following test in the same file already used
`root=str(tmp_path / "dataset")`, so the correct shape was one line away.

The rule now reads the value as well as the keyword, and refuses exactly one
shape: a `root` given as a string literal. That boundary is deliberate. A literal
cannot be per-test unique, which is decidable from the syntax alone with no false
positives, whereas `root=root` with `root` bound from `tmp_path` earlier in the
function is the idiomatic form 101 of the 194 recording calls use - telling that
apart from a shared local would need dataflow analysis a hygiene sweep has no
business doing. `root=None` is left alone for a different reason: it asks for the
shared home, which is a question about the home rather than about the call site,
and the modules that pass it are the ones whose subject is that resolution.

Three call sites named a literal and now take a `tmp_path` root. Two of them are
refused before the root is resolved and so were latent; the convention this
module implements already requires a root of those too, on the grounds that
exempting them means modelling which guard fires first, which is a fact about the
implementation rather than about the test.
