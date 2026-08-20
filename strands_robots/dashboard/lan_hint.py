"""Is this viewer on the same network as the dashboard — and should it stop leaving it?

MEASURED INCIDENT (BUGS.md Q52). A browser tab reached this dashboard through the
Cloudflare tunnel and streamed one camera at 0.45 MB/s: 20.7 GB out in 21 hours, 1.6
socket reopens a second, the stream dying and reconnecting forever. The viewer turned out
to be a device in the SAME HOUSE as the Mac serving it, so every JPEG went out of the
home upstream to Cloudflare and came straight back in — paying twice for the round trip
on the exact link that then could not sustain it.

The honest fix is not more compression: it is to tell that viewer the local address, which
costs nothing and needs no tunnel at all.

Two design decisions worth keeping:

1. THE COMPARISON NEEDS NO EXTERNAL LOOKUP. A SLAAC host holds its own GLOBAL IPv6
   address on the interface, so "same /64 as one of my own global addresses" is decidable
   from local facts alone. Asking an echo service for "my public IP" would add a network
   dependency to a diagnostic — and it is exactly the kind of call that fails on the
   network you are trying to diagnose.
2. IT MUST BE ALLOWED TO SAY IT DOES NOT KNOW. Behind IPv4 NAT the server sees only its
   private address, and a shared public address is not evidence of a shared LAN; so an
   IPv4 client returns ``None`` (unknown) rather than a guess. A wrong "you are local"
   sends the operator to an unreachable URL and makes the dashboard look broken.
"""

from __future__ import annotations

import ipaddress
from typing import Iterable, Sequence

#: A /64 is the unit a home network is delegated, and the unit SLAAC hosts share.
V6_NETWORK_BITS = 64


def _global_v6(addr: str) -> ipaddress.IPv6Address | None:
    """Parse a routable IPv6 address, or None for anything that proves nothing."""
    try:
        ip = ipaddress.ip_address(addr.split("%")[0])  # strip any zone id
    except ValueError:
        return None
    if not isinstance(ip, ipaddress.IPv6Address):
        return None
    if ip.is_link_local or ip.is_loopback or ip.is_private or ip.is_multicast:
        return None
    return ip


def same_network(client_ip: str | None, own_addrs: Iterable[str]) -> bool | None:
    """True/False when it can be decided from local facts, None when it cannot.

    Loopback is trivially local (the dashboard's own machine). An IPv6 client is local
    when it shares a /64 with one of our global addresses. Everything else - IPv4,
    unparseable, no client address at all - is unknown, deliberately.
    """
    if not client_ip:
        return None
    bare = client_ip.split("%")[0]
    try:
        parsed = ipaddress.ip_address(bare)
    except ValueError:
        return None
    if parsed.is_loopback:
        return True
    client = _global_v6(bare)
    if client is None:
        return None  # IPv4 (NAT hides the answer) or a non-routable address
    nets = {
        ipaddress.ip_network(f"{a}/{V6_NETWORK_BITS}", strict=False)
        for a in (_global_v6(x) for x in own_addrs)
        if a is not None
    }
    if not nets:
        return None  # we have no global address of our own to compare against
    return any(client in n for n in nets)


def lan_urls(private_addrs: Sequence[str], port: int) -> list[str]:
    """Direct URLs for a viewer that turns out to be local, best candidate first.

    Only PRIVATE IPv4 addresses are offered: a link-local v6 needs a zone id the browser
    will not accept, and handing out our global address would route the stream back out
    through the ISP - the very thing this exists to avoid.
    """
    out: list[str] = []
    for a in private_addrs:
        try:
            ip = ipaddress.ip_address(a.split("%")[0])
        except ValueError:
            continue
        if isinstance(ip, ipaddress.IPv4Address) and ip.is_private and not ip.is_loopback:
            url = f"http://{ip}:{port}"
            if url not in out:
                out.append(url)
    return out


def hint(client_ip: str | None, own_addrs: Sequence[str], port: int) -> dict:
    """The payload the UI renders. `same_network=None` must render NOTHING."""
    local = same_network(client_ip, own_addrs)
    urls = lan_urls(own_addrs, port) if local else []
    if local and not urls:
        # Local, but we cannot name a usable address - say so instead of implying one.
        return {
            "same_network": True,
            "client_ip": client_ip,
            "lan_urls": [],
            "why": "you appear to be on this network, but the dashboard could not find a "
                   "private address of its own to point you at",
        }
    if local:
        return {
            "same_network": True,
            "client_ip": client_ip,
            "lan_urls": urls,
            "why": "you are on the same network as this dashboard - the local address skips "
                   "the trip out to Cloudflare and back, which is where camera streams stall",
        }
    if local is False:
        return {"same_network": False, "client_ip": client_ip, "lan_urls": [],
                "why": "you are reaching this dashboard from another network"}
    return {"same_network": None, "client_ip": client_ip, "lan_urls": [],
            "why": "cannot tell from an IPv4 address behind NAT whether you are local"}
