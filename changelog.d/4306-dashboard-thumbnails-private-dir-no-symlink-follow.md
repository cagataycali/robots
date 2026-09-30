### Fixed: dashboard episode thumbnails live in a private directory and the thumb route never follows a link

`RecordController` kept thumbnails in `<tmpdir>/strands-record-thumbs`, a
guessable name in the shared temp directory that nothing created until the first
recorded frame, so another account on the host could create it first and fill it
with symlinks; `GET /api/record/thumb/{episode}/{camera}` composed the file name
from the URL, tested it with `Path.is_file` and served it with `FileResponse`,
both of which follow links, so the credential store or the bootstrap token came
back labelled `image/jpeg` (f006, CWE-59 / CWE-377). The default root is now a
per-process `mkdtemp` directory (`0700`, removed at shutdown); a configured root
is created `0700` and refused unless it is a real directory the service user owns
that nobody else can write; the route opens with `O_NOFOLLOW`, requires a regular
file inside the root by `fstat` on the descriptor it reads, and the writer
refuses to write through a link.
