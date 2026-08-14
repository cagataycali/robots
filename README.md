# Artifact: review-history markers removed from shipped docstrings

`capture.py` measures, per touched module: the marker count on main, the marker
count on the branch, and the **docstring-stripped AST digest** of both (an
`ast.NodeTransformer` that drops each leading string `Expr`, then
`sha256(ast.dump(tree))`). An identical digest is mechanical proof that no
executable line moved. It also runs the new guard against main's source and
against the branch, and censuses the whole package.

`compose.py` draws the figure and re-asserts every number it prints, plus a
clean 8px border.

    python3 capture.py <base-sha> /tmp/facts.json
    python3 compose.py /tmp/facts.json docstring-sweep.png
