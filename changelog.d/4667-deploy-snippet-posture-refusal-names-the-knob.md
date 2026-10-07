### Security: a rejected mesh auth posture refuses a real arm with a fixed sentence, not the resolver's exception

When `STRANDS_MESH_AUTH_MODE` holds a value the mesh resolver rejects (a
misspelling, or `none` without its second factor), the real-mode deploy
snippet and the spawn route used to answer with the resolver's `ValueError`
text, which quotes the raw environment value, and the snippet route sent that
sentence to the browser as a 422 (CodeQL py/stack-trace-exposure). The
refusal is now one fixed sentence that names `STRANDS_MESH_AUTH_MODE` and the
two values it accepts; the resolver's reason goes to the dashboard log at
WARNING, where the operator reads it. Same gate, same outcome: nothing starts.
