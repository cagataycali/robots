"""Measure what the model receives for every bound agent tool, in this tree."""
import importlib, json, pathlib, pkgutil, sys, ast
import docstring_parser
import strands_robots.tools as T

tree = str(pathlib.Path(T.__file__).parents[2])
print("TREE:", tree)

TARGETS = {
    "gr00t_inference": ["hf_repo", "hf_subfolder", "hf_local_dir", "hf_token",
                        "lifecycle", "remove_volumes", "force"],
    "lerobot_teleoperate": ["policy_path", "dagger_input_device", "dagger_num_episodes"],
    "train_policy": ["lora_r", "lora_alpha", "lora_target_modules"],
}

out = {"tree": tree, "params": {}, "totals": {}, "description": {}}
ph_total = phantom_total = n_tools = 0
for info in sorted(m.name for m in pkgutil.iter_modules(T.__path__)):
    if info.startswith("_"):
        continue
    mod = importlib.import_module(f"strands_robots.tools.{info}")
    for name, obj in vars(mod).items():
        spec = getattr(obj, "tool_spec", None)
        if not isinstance(spec, dict) or spec.get("name") != name:
            continue
        n_tools += 1
        props = spec["inputSchema"]["json"]["properties"]
        ph_total += sum(1 for k, v in props.items()
                        if v.get("description", "").strip() == f"Parameter {k}")
        src = ast.parse(pathlib.Path(mod.__file__).read_text())
        fn = next(n for n in ast.walk(src)
                  if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == name)
        a = fn.args
        real = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
        phantom_total += sum(1 for p in docstring_parser.parse(ast.get_docstring(fn) or "").params
                             if p.arg_name not in real)
        if name in TARGETS:
            for p in TARGETS[name]:
                d = " ".join(props[p].get("description", "").split())
                out["params"][f"{name}.{p}"] = {
                    "text": d,
                    "placeholder": d == f"Parameter {p}",
                }
        if name == "gr00t_inference":
            d = spec["description"]
            out["description"] = {
                "host RCE": "host RCE" in d,
                "STRANDS_GR00T_REPO_URL_ALLOW": "STRANDS_GR00T_REPO_URL_ALLOW" in d,
                "chars": len(d),
            }
out["totals"] = {"tools": n_tools, "placeholder": ph_total, "phantom": phantom_total}
dest = pathlib.Path(sys.argv[1])
dest.write_text(json.dumps(out, indent=2))
print(json.dumps(out["totals"]))
