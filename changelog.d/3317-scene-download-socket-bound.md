### Fixed: a preset 3DGS scene download bounds every socket operation

`download_gsplat_scene` fetched its asset with `urllib.request.urlretrieve`,
whose signature is `url, filename, reporthook, data` -- there is no timeout
parameter to pass. The fetch therefore ran on the process-global default socket
timeout, which is `None` unless some unrelated import mutated it, so a peer that
completed the connection, sent its headers and then stopped sending held the
calling thread indefinitely.

That is not merely slow, it defeats a documented fail-loud contract. The
`isaac_gs` demo's `resolve_background` raises `RuntimeError` when the photoreal
path cannot initialize -- a download failure included -- precisely so a user who
asked for the captured scene never silently receives the procedural gradient,
and `allow_fallback=True` restores that demotion "for every failure mode". A
stalled host produced neither outcome. Measured against a loopback peer that
answers with `Content-Length` and then goes quiet, `resolve_background(
allow_fallback=True)` was still blocked after 90 seconds of observation: nothing
raised, so nothing was demoted, and the flag whose whole purpose is to keep the
demo rendering could not fire either. The `mujoco_gs` Gradio handler wedges the
same way, after yielding its "Downloading ..." progress line.

The body now streams through `urlopen(url, timeout=...)`, whose `timeout`
reaches the connection object itself -- urllib's `AbstractHTTPHandler.do_open`
forwards it as `http_class(host, timeout=req.timeout)` -- so it becomes the
socket timeout and bounds each read of the body, not merely the connect and TLS
handshake. Nothing process-global is mutated. The same stalled peer now raises
`TimeoutError` at 60.1 seconds and the fallback path returns its
`PanoramaBackground`.

The bound is per socket operation rather than on the whole transfer, because
these assets run to hundreds of megabytes and a slow link is not a stalled one;
what no honest transfer needs is a minute of silence mid-body. It is a new
`timeout` keyword argument on `download_gsplat_scene`, graded by the same
`positive_finite_number_error` domain the rest of the tree uses for a span of
time, so a bound of `0`, a negative one, `inf` or `nan` is refused rather than
silently standing for no bound at all.

`urlretrieve` also raised `ContentTooShortError` when the body came up short of
the declared `Content-Length`; that check is kept, with the same
`(filename, headers)` content attached, because the fetch renames its `.part`
sidecar onto the cache path and a truncated scene cached as a complete one is
worse than a download that failed -- every later call would read the short file
straight out of the cache.
