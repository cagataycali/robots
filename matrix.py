import ast, json, pathlib, sys, collections
COV=json.load(open(f"/tmp/cov-{sys.argv[1]}.json"))["files"]
OPT=("isaac/","newton/","rendering/backgrounds","rtps/idl","groot/server_wrapper","cosmos3/policy_diffusers")
def optdep(p): return any(k in p for k in OPT)

# VIEW 3: guard-refusal asymmetry. For each `if err := GUARD(...)` find the
# refusal statement within the next few lines and mark covered/missing.
print("### VIEW 3 - guard-refusal asymmetry (same function, some refusals covered, some not)")
per_fn=collections.defaultdict(list)
for path,d in COV.items():
    if optdep(path): continue
    src=pathlib.Path(path).read_text().splitlines()
    miss=set(d["missing_lines"]); ex=set(d["executed_lines"])
    tree=ast.parse("\n".join(src))
    for fn in ast.walk(tree):
        if not isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)): continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.If): continue
            seg=ast.get_source_segment("\n".join(src), node.test) or ""
            if ":=" not in seg and "_error(" not in seg: continue
            # refusal = first raise/return in the If body
            for st in node.body:
                t=(src[st.lineno-1].strip() if st.lineno<=len(src) else "")
                if t.startswith(("raise ","return ")):
                    state = "MISS" if st.lineno in miss else ("exec" if st.lineno in ex else "?")
                    per_fn[(path,fn.name)].append((st.lineno,state,seg[:52]))
                    break
for (p,f),rows in sorted(per_fn.items()):
    states={s for _,s,_ in rows}
    if "MISS" in states and "exec" in states:
        print(f"  {p}::{f}")
        for ln,st,seg in rows: print(f"      {st:4}  L{ln}  {seg}")

# SHARED-HELPER view: a guard called from N functions where only some refusals ran.
print("\n### SHARED-HELPER refusal matrix (guard used by >=2 functions, mixed coverage)")
by_guard=collections.defaultdict(list)
for path,d in COV.items():
    if optdep(path): continue
    src=pathlib.Path(path).read_text().splitlines()
    miss=set(d["missing_lines"]); ex=set(d["executed_lines"])
    tree=ast.parse("\n".join(src))
    for fn in ast.walk(tree):
        if not isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)): continue
        for node in ast.walk(fn):
            if not isinstance(node,ast.If): continue
            seg=ast.get_source_segment("\n".join(src),node.test) or ""
            names=[n.func.id for n in ast.walk(node.test)
                   if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
            names+= [n.func.attr for n in ast.walk(node.test)
                   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
            g=[n for n in names if n.endswith("_error") or n.startswith("_validate")]
            if not g: continue
            for st in node.body:
                t=(src[st.lineno-1].strip() if st.lineno<=len(src) else "")
                if t.startswith(("raise ","return ")):
                    st2 = "MISS" if st.lineno in miss else ("exec" if st.lineno in ex else "?")
                    by_guard[g[0]].append((path,fn.name,st.lineno,st2))
                    break
for guard,rows in sorted(by_guard.items()):
    fns={(p,f) for p,f,_,_ in rows}
    if len(fns)<2: continue
    if not any(s=="MISS" for *_,s in rows): continue
    print(f"  {guard}:  {sum(1 for *_,s in rows if s=='exec')} covered / {sum(1 for *_,s in rows if s=='MISS')} MISSING")
    for p,f,ln,s in sorted(rows, key=lambda r:(r[3]!='MISS',r[0])):
        print(f"      {s:4}  {p}::{f}  L{ln}")
