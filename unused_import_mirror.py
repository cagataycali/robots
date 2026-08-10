"""Mirror of py/unused-import: an imported name never referenced outside a string."""
import ast, pathlib, sys

def unused_imports(src: str) -> list[tuple[int, int, int, str]]:
    tree = ast.parse(src)
    bound: dict[str, tuple[int, int, int]] = {}
    for node in tree.body:  # top level only
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname is None and "." in a.name:
                    continue  # `import a.b` binds `a`
                bound[a.asname or a.name] = (node.lineno, node.col_offset + 1, node.end_col_offset + 1)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue  # a compiler directive, never referenced by name
            for a in node.names:
                if a.name == "*":
                    continue
                bound[a.asname or a.name] = (node.lineno, node.col_offset + 1, node.end_col_offset + 1)
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            n = node
            while isinstance(n, ast.Attribute):
                n = n.value
            if isinstance(n, ast.Name):
                used.add(n.id)
    return sorted((ln, c0, c1, name) for name, (ln, c0, c1) in bound.items() if name not in used)

if __name__ == "__main__":
    root = pathlib.Path(sys.argv[1])
    files = [root] if root.is_file() else sorted(root.rglob("*.py"))
    total = 0
    for f in files:
        for ln, c0, c1, name in unused_imports(f.read_text()):
            print(f"{f}:{ln}:{c0}-{c1}\t{name}")
            total += 1
    print(f"--- {total} finding(s) over {len(files)} file(s) ---")
