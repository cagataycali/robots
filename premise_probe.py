from pathlib import Path
import strands_robots.training.lerobot as L
print("TREE:", Path(L.__file__).parents[2])

# 1. Exception MROs: is the Hub-fetch failure class an OSError?
import huggingface_hub.errors as HE
for name in ("LocalEntryNotFoundError", "EntryNotFoundError", "HfHubHTTPError",
             "RepositoryNotFoundError", "OfflineModeIsEnabled", "GatedRepoError"):
    cls = getattr(HE, name, None)
    if cls is None:
        print(f"  {name}: ABSENT"); continue
    print(f"  {name}: OSError={issubclass(cls, OSError)} ValueError={issubclass(cls, ValueError)} "
          f"mro={[c.__name__ for c in cls.__mro__[:5]]}")

# 2. Does transformers surface a plain OSError?
print("\n2. transformers cached_file raise class -> measured live below")

# 3. Local-cache probe with no network
from huggingface_hub import try_to_load_from_cache
for repo, fn in [("Qwen/Qwen3-VL-4B-Instruct", "config.json"),
                 ("Qwen/Qwen3-VL-4B-Instruct", "tokenizer_config.json"),
                 ("definitely/not-a-real-repo-xyz", "config.json")]:
    r = try_to_load_from_cache(repo, fn)
    print(f"  try_to_load_from_cache({repo!r}, {fn!r}) -> {type(r).__name__}: {str(r)[:70]}")

# 4. robometer default base_model_id
import sys
sys.path.insert(0, "/tmp/lerobot-src/src")
from lerobot.rewards.robometer.configuration_robometer import RobometerConfig
import dataclasses
for f in dataclasses.fields(RobometerConfig):
    if f.name in ("base_model_id", "vlm_config"):
        d = f.default if f.default is not dataclasses.MISSING else f.default_factory
        print(f"  field {f.name}: type={f.type} default={d}")
