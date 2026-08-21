/**
 * Q127: an <img src> the dashboard's own auth cannot reach.
 *
 * The episode thumbnails were rendered as `<img src="/api/record/thumb/0/top">`, straight from the
 * URL the server puts in the session payload. That has two faults at once and both only appear
 * where it matters least visibly:
 *
 *  1. A browser image request carries NO Authorization header, so with auth enabled — every remote
 *     session through the tunnel — each thumbnail 401s and renders as a broken image. The operator
 *     reads that as "the recording did not capture anything", which is the opposite of true, and it
 *     appears exactly when the episode they just recorded is the thing they want to check.
 *  2. A relative path resolves against the PAGE, not the configured backend, so a UI pointed at
 *     another server asks the wrong host for the picture and gets its 404.
 *
 * endpoints.ts already knew this ("an <img src> cannot carry an Authorization header") and
 * CameraGallery already did it right. This makes that rule reusable instead of remembered: the
 * bytes come through the authed fetch, the object URL is revoked on unmount and on every src
 * change, and a failure renders the alt text rather than a broken-image glyph.
 */
import { useEffect, useState } from 'react'
import { apiBlob } from '../lib/endpoints'

export default function AuthedImg(
  { path, alt, className }: { path: string; alt: string; className?: string },
) {
  const [url, setUrl] = useState('')
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let alive = true
    let made = ''
    setFailed(false)
    setUrl('')
    void apiBlob(path)
      .then(u => {
        if (!alive) { URL.revokeObjectURL(u); return }   // unmounted mid-flight: still not a leak
        made = u
        setUrl(u)
      })
      .catch(() => { if (alive) setFailed(true) })
    return () => {
      alive = false
      if (made) URL.revokeObjectURL(made)
    }
  }, [path])

  // No glyph for a picture that is merely late, and words instead of a broken icon when it fails.
  if (failed) return <span className="hint small" title={path}>{alt} — unavailable</span>
  if (!url) return <span className="thumb-loading" aria-label={`${alt} loading`} />
  return <img src={url} alt={alt} className={className} loading="lazy" />
}
