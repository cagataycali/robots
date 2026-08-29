Ports the read-only half of the neon bundle's error-code lookup to the ``@tool``
surface: :mod:`strands_robots.tools.g1.g1_error_codes` snapshots the SDK
return-code catalogue that already lives on
:data:`~strands_robots.tools.g1._g1_common.ERR_CODES` and exposes two agent-
facing verbs (``g1_list_error_codes``, ``g1_decode_error_code``) so a caller
that receives a ``refusal_code`` from any other verb in this package can list
the catalogue - or decode one code by number - through the same tool surface
every other verb here answers on. No SDK submodule loads on import (the
snapshot lives in :mod:`._g1_common` which never touches ``unitree_sdk2py``
either); the two verbs quote every text field verbatim from the catalogue so
a re-word of one entry lands in the constant once. Refs strands-labs/robots#358.
