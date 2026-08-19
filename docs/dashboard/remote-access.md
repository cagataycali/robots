# Remote access: your home lab from a phone

The dashboard commands real motors. Putting it on the internet is therefore an
ordering problem, not a networking one: **prove the guard refuses a remote
client before you open the door.** This page does it in that order.

## What the guard actually does

Every `/api/*` and `/ws/*` route goes through one ASGI middleware (raw ASGI, not
`BaseHTTPMiddleware`, because a WebSocket scope never reaches an HTTP middleware -
and `/ws/mesh`, `/ws/chat`, `/ws/voice` are exactly the routes that drive motors
and spend money). It accepts, in order:

1. the static `security.auth_token` (env `DASHBOARD_AUTH_TOKEN`), or
2. a WebAuthn session JWT minted by a registered passkey.

With **neither** configured the posture is **local-only, never open**: loopback
passes, everything else is refused. And "loopback" is judged honestly - a
request carrying `cf-connecting-ip`, `x-forwarded-for` or `x-real-ip` is treated
as remote no matter what the socket peer says, because a tunnel connects *from*
localhost on behalf of the whole internet.

Verify it on your own machine before you trust it:

```bash
curl -s -o /dev/null -w '%{http_code}\n' localhost:8090/api/fleet
# 200  <- you are local

curl -s -H 'x-forwarded-for: 203.0.113.9' localhost:8090/api/fleet
# {"detail":"unauthorized"}   <- 401, exactly what a tunnelled client would get
```

The static page still returns 200 - the app shell is not secret. The data,
the commands and the sockets are.

Only `/api/auth/status`, `/api/auth/register/{begin,finish}` and
`/api/auth/login/{begin,finish}` are exempt, so a browser can ask "do I need to
log in?" and complete a ceremony. Registration gates *itself*: once a
credential exists, open enrollment is over.

## Path A: a token and a link (works today)

```bash
DASHBOARD_AUTH_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')" \
  python -m strands_robots dashboard --port 8090
```

Now the 401 above applies to *everyone*, including you - so hand the browser the
token once:

```
https://robots.example.com/?token=<the token>
```

The frontend stores it in `localStorage`, sends `Authorization: Bearer …` on
every API call, and appends `?token=` to WebSocket URLs (a browser cannot set
headers on a WS handshake, so the query string is the only channel; the server
accepts it for `/ws` only). The link is also what makes a **QR code that puts a
phone straight on one robot** - `?backend=` picks the host, `?token=` unlocks it.

A URL-borne secret lands in history and in logs. Treat the link as the password
it is: generate a long token, use it once per device, and rotate it by
restarting with a new value. For anything beyond your own household, put an
identity proxy (Cloudflare Access and friends) in front of the tunnel as well.

## Path B: passkeys (the module is in, the enrollment screen is not yet)

`strands_robots/dashboard/auth.py` implements the full WebAuthn rail - the
private key never leaves your Touch ID / Face ID enclave, the dashboard stores
only public keys in `~/.strands_dashboard/auth.json` (chmod 600, hot-reloaded on
mtime change), and a signed challenge mints a short-lived HS256 JWT.

```bash
STRANDS_DASH_AUTH_ENABLED=true \
STRANDS_DASH_AUTH_RP_ID=robots.example.com \
STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(24))')" \
  python -m strands_robots dashboard --port 8090

curl -s localhost:8090/api/auth/status
# {"enabled":false,"setup_required":true,"credentials":[],"bootstrap_required":true,
#  "rp_id":"localhost","secure_context":true,"rpid_usable":true,"authenticated":false}
```

`setup_required` means no credential exists yet; `bootstrap_required` means the
first enrollment must present that one-time token, so the open-enrollment window
cannot be walked through by whoever finds the URL first.

**Status, honestly:** the module, the routes and the guard are shipped and
tested. The browser-side enrollment screen is not in the frontend yet - nothing
in `frontend/src` calls `navigator.credentials`. Until it lands, use Path A for
remote access. When it lands, enroll **from the device you will actually use
remotely** (your phone), because a passkey lives in that device's enclave.

Two rules WebAuthn imposes that will otherwise waste your evening:

- **`rp_id` cannot be a raw IP.** `https://192.168.1.50` is refused by the
  browser before the ceremony starts; `/api/auth/status` says so in `warning`.
  Use a hostname, or force `STRANDS_DASH_AUTH_RP_ID`.
- **The origin must be secure**: HTTPS, or `http://localhost`. A LAN hostname
  over plain HTTP is not a secure context. The tunnel gives you HTTPS for free,
  which is why passkeys and tunnels arrive together.

## The tunnel

```bash
cloudflared tunnel create robots
# ~/.cloudflared/robots.yml
#   tunnel: <id>
#   credentials-file: /Users/you/.cloudflared/<id>.json
#   ingress:
#     - hostname: robots.example.com
#       service: http://localhost:8090
#     - service: http_status:404
cloudflared tunnel route dns robots robots.example.com
cloudflared tunnel run robots
```

Order of operations, and the reason for it:

1. Configure a credential (Path A token, or a passkey once you can enroll one).
2. **Restart the dashboard** - the guard is loaded at startup, so a running
   process that predates your change proves nothing. A 200 from an old process
   is stale code, not evidence.
3. Re-run the `x-forwarded-for` probe above and confirm the 401.
4. *Then* `cloudflared tunnel run`.
5. Load the site on the phone and check that a **wrong** token is refused.

## Once you are remote

- **E-stop from a phone is the feature**, not a curiosity - it is the reason
  this is worth exposing. Set the **same** `STRANDS_MESH_OVERRIDE_CODE` on every
  peer, or a broadcast stop locks each robot until its process restarts.
- Cameras are the bandwidth. Lower `STRANDS_MESH_CAMERA_HZ` on the publishing
  peer rather than fighting the uplink.
- The mesh is a separate trust domain from the web session. `--local-dev`
  disables mesh TLS for single-machine work; do not leave it on for a fleet
  that spans machines. See [Multi-robot Mesh](../mesh.md).
- A session lasts `STRANDS_DASH_AUTH_TOKEN_TTL` seconds (default 86400).
  `GET /api/auth/credentials` lists enrolled passkeys and
  `DELETE /api/auth/credentials/{id}` revokes a lost phone.
