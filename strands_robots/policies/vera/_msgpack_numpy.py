# Vendored msgpack-numpy codec — wire-compatible with vera's server protocol
# and openpi-client / DreamZero. Apache-2.0 / BSD-style.
#
# Provides NumPy ndarray + scalar support for msgpack so the VERA WebSocket
# policy client can speak the server's wire protocol from a numpy>=2
# environment (composes with lerobot, strands-robots, etc.) without any
# openpi-client / vera Python dependency.
"""Adds NumPy array support to msgpack.

The encoding matches vera's ``vera/server/protocol/_msgpack_numpy.py`` (which
is itself wire-compatible with DreamZero / openpi-client's
``msgpack_numpy``). ndarrays and numpy scalars are encoded as tagged maps so
dicts-of-arrays pack/unpack directly.
"""

from __future__ import annotations

import functools

import msgpack
import numpy as np


def _pack(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):  # void/object/complex are not wire-safe
            raise ValueError(f"cannot serialize ndarray of dtype {obj.dtype!r}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": list(obj.shape),
        }
    if isinstance(obj, np.generic):
        # Match cosmos3's tag ("__npgeneric__"); vera server uses "__npscalar__".
        # We support BOTH on the decode side so we can talk to either server.
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack(obj):
    if b"__ndarray__" in obj:
        # Use frombuffer().reshape().copy() so the result is writable (frombuffer
        # returns a read-only view over the transient bytes).
        return (
            np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"dtype"]))
            .reshape(tuple(obj[b"shape"]))
            .copy()
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    if b"__npscalar__" in obj:  # vera server's tag name
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack, use_bin_type=True)
packb = functools.partial(msgpack.packb, default=_pack, use_bin_type=True)

Unpacker = functools.partial(
    msgpack.Unpacker, object_hook=_unpack, raw=False, strict_map_key=False
)
unpackb = functools.partial(
    msgpack.unpackb, object_hook=_unpack, raw=False, strict_map_key=False
)
