#!/bin/bash
for V in 1.4.1 1.5.0; do
  D=/tmp/shadow$V
  "$D/bin/pip" install -q typer >/dev/null 2>&1
  echo "=============== huggingface_hub $V ==============="
  echo "installed: $("$D/bin/python" -c 'import huggingface_hub as h; print(h.__version__)')"
  echo "--- \$ hf buckets create demo-bucket ---"
  out=$("$D/bin/hf" buckets create demo-bucket 2>&1); rc=$?
  echo "$out" | grep -v '^$' | tail -5
  echo "  rc=$rc"
  echo "--- \$ hf sync --help ---"
  out=$("$D/bin/hf" sync --help 2>&1); rc=$?
  echo "$out" | grep -v '^$' | tail -5
  echo "  rc=$rc"
  echo
done
