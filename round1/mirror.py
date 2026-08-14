"""Mirror of CodeQL py/unused-local-variable, validated against published alerts 903/904."""
import ast, pathlib, sys

def unused_locals(src: str):
    """Report assigned-but-never-read locals, treating a tuple target as used
    when ANY of its elements is read (the hypothesis alerts 903/904 imply)."""
    tree = ast.parse(src)
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        loads = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Tuple):
                    names = [e for e in tgt.elts if isinstance(e, ast.Name)]
                    if any(e.id in loads for e in names):
                        continue  # tuple partially used -> CodeQL stays silent
                    for e in names:
                        if e.id not in loads:
                            out.append((e.lineno, e.col_offset + 1, e.end_col_offset + 1, e.id))
                elif isinstance(tgt, ast.Name) and tgt.id not in loads:
                    out.append((tgt.lineno, tgt.col_offset + 1, tgt.end_col_offset + 1, tgt.id))
    return sorted(out)

if __name__ == "__main__":
    p = pathlib.Path(sys.argv[1])
    print(f"TREE: {pathlib.Path(__file__).parents[1]}")
    for line, c0, c1, name in unused_locals(p.read_text()):
        print(f"  {p}:{line} cols {c0}-{c1}  {name}")
