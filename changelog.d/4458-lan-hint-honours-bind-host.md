### Fixed: the "open the local address" banner no longer offers an address the dashboard is not listening on

The dashboard binds `127.0.0.1` by default (a LAN bind is refused until a
passkey or token guards the API), yet the LAN hint read the machine's private
addresses and offered `http://<lan-ip>:<port>` to a viewer on the same network --
a link that failed to load. On a loopback bind the hint now offers no URL and
says how to get one (`--host 0.0.0.0` once the API is guarded).
