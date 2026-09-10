"""use_unitree - a universal ``@tool`` wrapper around every Unitree SDK2 client.

Ported from ``cagataycali/neon-the-g1/tools/use_unitree.py`` (refs
strands-labs/robots#2928). ONE tool covers the entire unitree_sdk2_python
surface without a hand-written ``@tool`` per method - adding a new method
upstream requires no change here, because discovery is dynamic
(``inspect`` first, AST fallback for machines without the SDK).

Services (auto-discovered from the SDK):
    loco            unitree_sdk2py.g1.loco.g1_loco_client.LocoClient
    arm             unitree_sdk2py.g1.arm.g1_arm_action_client.G1ArmActionClient
    audio           unitree_sdk2py.g1.audio.g1_audio_client.AudioClient
    motion_switcher unitree_sdk2py.comm.motion_switcher.motion_switcher_client.MotionSwitcherClient
    vui             unitree_sdk2py.go2.vui.vui_client.VuiClient
    robot_state     unitree_sdk2py.go2.robot_state.robot_state_client.RobotStateClient

SDK-load hygiene: ``import strands_robots.tools.g1.use_unitree`` pulls no
``unitree_sdk2py`` submodule. Every SDK touch is inside a function body -
the meta operations (``list_services`` / ``list_operations`` /
``describe_operation``) fall back to AST-walking the SDK source when the
SDK cannot import (CI, dev boxes without cyclonedds), so discovery works
robot-free.

Client singletons: each SDK client is cached after first ``Init()``
because a second ``Init()`` on the same client class can crash the
process (the same rule :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
serialises DDS factory construction for). ``Init()`` builds the client's
DDS request/response endpoints, so it runs under the *shared*
``_DDS_INIT_LOCK`` from :mod:`~strands_robots.tools.g1._g1_common` - the
same lock the driver and :mod:`~strands_robots.tools.g1._dds_engine` hold
while creating readers or writers, because concurrent endpoint
construction segfaults the CycloneDDS bindings. All RPC execution is
serialised on one further lock - the SDK clients are not thread-safe;
concurrent calls clobber each other's response futures and return
rc=3104.

Safety rails:
    * Mutative ops (Set*, Execute*, Move*, ...) are detected and flagged
      in the response.
    * ``HIGH_DANGER_OPS`` names the calls that can drop or walk the robot
      (ZeroTorque, SetFsmId, SetVelocity, Move, ReleaseMode, ...); those
      are flagged loudly in every response envelope - including the error
      envelope, where the flag is what says whether an unanswered command
      may still be executing.
    * Every mutative or high-danger op stops for operator approval BEFORE
      the SDK RPC is dispatched, through the same decision path the ROS
      transports use (:func:`~strands_robots.tools._command_gate.gate_motion`):
      ``STRANDS_UNITREE_COMMAND_ALLOW`` (comma-separated ``service.operation``
      entries, or ``*``) pre-approves; ``BYPASS_TOOL_CONSENT=true`` lifts the
      gate with a WARNING; otherwise the operator is prompted through the
      tool context, and with no context reachable the call is refused and
      nothing is sent. A log line is not an authorization control: before
      this gate an agent steered by untrusted content could walk the robot
      or drop its torque with only a warning in the log (F-001, CWE-862).
      Meta and read-only ops are never gated, and neither is
      ``loco.StopMove`` - stopping must never get harder.
    * Prefer the FSM-gated verbs (``g1_send_action``, ``g1_run_policy``,
      ``g1_set_stand_height``, ...) for routine motion - they route
      through :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`.
      ``use_unitree`` is the raw escape hatch.
"""

from __future__ import annotations

import inspect
import logging
import os
import threading
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from strands_robots.tools._command_gate import gate_motion
from strands_robots.tools.g1._g1_common import _DDS_INIT_LOCK, ensure_dds

logger = logging.getLogger(__name__)

# One lock for every SDK RPC - the clients are not thread-safe.
_CALL_LOCK = threading.Lock()

# service_name -> cached client instance
_CLIENTS: dict[str, Any] = {}
_CLIENTS_LOCK = threading.Lock()

# service_name -> (sdk_class_qualname, default_timeout_s)
SERVICES: dict[str, tuple[str, float]] = {
    "loco": ("unitree_sdk2py.g1.loco.g1_loco_client.LocoClient", 5.0),
    "arm": ("unitree_sdk2py.g1.arm.g1_arm_action_client.G1ArmActionClient", 10.0),
    "audio": ("unitree_sdk2py.g1.audio.g1_audio_client.AudioClient", 10.0),
    "motion_switcher": (
        "unitree_sdk2py.comm.motion_switcher.motion_switcher_client.MotionSwitcherClient",
        3.0,
    ),
    "vui": ("unitree_sdk2py.go2.vui.vui_client.VuiClient", 3.0),
    "robot_state": (
        "unitree_sdk2py.go2.robot_state.robot_state_client.RobotStateClient",
        3.0,
    ),
}

MUTATIVE_PREFIXES = (
    "Set",
    "Execute",
    "Move",
    "Start",
    "Stop",
    "Damp",
    "Sit",
    "HighStand",
    "LowStand",
    "WaveHand",
    "ShakeHand",
    "Squat2StandUp",
    "Lie2StandUp",
    "StandUp2Squat",
    "BalanceStand",
    "ZeroTorque",
    "LedControl",
    "TtsMaker",
    "PlayStream",
    "PlayStop",
    "SelectMode",
    "ReleaseMode",
    "ServiceSwitch",
    "SwitchTo",
)

READONLY_WHITELIST = {
    "CheckMode",
    "GetActionList",
    "GetVolume",
    "GetBrightness",
    "GetSwitch",
    "GetFsmId",
    "ServiceList",
    "Init",
}

HIGH_DANGER_OPS = {
    ("loco", "ZeroTorque"),  # robot collapses off-gantry
    ("loco", "SetFsmId"),  # fsm_id=0 -> collapse
    ("loco", "SetVelocity"),  # walking
    ("loco", "Move"),  # walking (continuous!)
    ("loco", "WaveHand"),  # leg motion in some FSMs
    ("loco", "ShakeHand"),
    ("motion_switcher", "ReleaseMode"),  # robot uncontrolled
}

# Pre-approve ``service.operation`` pairs (comma-separated, ``*`` for all) for
# headless runs. Read by the shared gate, which also honours BYPASS_TOOL_CONSENT.
COMMAND_ALLOW_ENV = "STRANDS_UNITREE_COMMAND_ALLOW"

# Mutative by prefix (``Stop``) but never gated: it is the tool's own emergency
# stop, and an approval prompt in front of a halt makes the robot less safe.
_UNGATED_STOP_OPS = frozenset({("loco", "StopMove")})


def _is_readonly(operation_name: str) -> bool:
    if operation_name in READONLY_WHITELIST:
        return True
    if operation_name.startswith("_"):
        return False
    return operation_name.startswith("Get") or operation_name.startswith("Check")


def _is_mutative(operation_name: str) -> bool:
    if _is_readonly(operation_name):
        return False
    return any(operation_name.startswith(p) for p in MUTATIVE_PREFIXES)


def _import_client_class(qualname: str) -> Any:
    mod_path, class_name = qualname.rsplit(".", 1)
    mod = __import__(mod_path, fromlist=[class_name])
    return getattr(mod, class_name)


def _get_client(service_name: str) -> Any:
    """Singleton SDK client - Init() exactly once per process.

    ``client.Init()`` is what creates the client's DDS request/response
    channel endpoints, so this is a *bus* construction and not merely a cache
    fill: it has to be serialised against every other endpoint construction
    in the process, not just against other callers of this function.
    :data:`~strands_robots.tools.g1._g1_common._DDS_INIT_LOCK` is the shared
    lock the driver and :mod:`~strands_robots.tools.g1._dds_engine` already
    hold while creating readers or writers, and these tools are the second
    consumer that lock was introduced for (issue #358). Holding only the
    module-private ``_CLIENTS_LOCK`` would serialise this path against itself
    while leaving it racing the engine's subscribers - and the loss is a
    native segfault in the CycloneDDS bindings, which is not an exception the
    "return an envelope, never raise" boundary can catch.

    Lock order is ``_CLIENTS_LOCK`` then ``_DDS_INIT_LOCK``, which cannot
    deadlock: ``_CLIENTS_LOCK`` is private to this module, so nothing can
    acquire it while holding ``_DDS_INIT_LOCK``, and :func:`ensure_dds` - the
    other ``_DDS_INIT_LOCK`` holder on this path - is called by
    :func:`_execute` before it reaches here and has released the lock by
    then. ``_DDS_INIT_LOCK`` is a plain, non-reentrant ``Lock``, so that
    ordering is load-bearing rather than incidental.

    Args:
        service_name: A key of :data:`SERVICES`.

    Returns:
        The cached SDK client for that service, constructed on first call.

    Raises:
        KeyError: ``service_name`` is not a known service.
        Exception: Whatever the SDK raises from import, construction or
            ``Init()``; :func:`_execute` converts it to an error envelope.
    """
    with _CLIENTS_LOCK:
        if service_name not in _CLIENTS:
            qualname, timeout = SERVICES[service_name]
            # The import is not an endpoint construction, so it stays outside
            # the shared lock and the critical section covers only the bus.
            cls = _import_client_class(qualname)
            with _DDS_INIT_LOCK:
                client = cls()
                client.SetTimeout(timeout)
                client.Init()
            _CLIENTS[service_name] = client
        return _CLIENTS[service_name]


def _ast_methods_for_class(qualname: str) -> dict[str, list[str]]:
    """AST-walk the SDK source for a class - works without importing the SDK."""
    import ast as _ast

    mod_path, class_name = qualname.rsplit(".", 1)
    rel_path = mod_path.replace(".", "/") + ".py"

    candidates = [
        os.environ.get("UNITREE_SDK_PATH", ""),
        "/tmp/unitree_sdk2_python",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "unitree_sdk2_python"),
    ]
    src_file = None
    for root in candidates:
        if not root:
            continue
        p_ = os.path.join(root, rel_path)
        if os.path.isfile(p_):
            src_file = p_
            break
    if src_file is None:
        return {}

    try:
        with open(src_file, encoding="utf-8") as fh:
            tree = _ast.parse(fh.read(), filename=src_file)
    except (OSError, SyntaxError, UnicodeDecodeError):
        # The on-disk source is unreadable or is not Python. ``{}`` here is
        # indistinguishable from "the class declares no methods", so keep the
        # set narrow: any other exception is a defect in this reader.
        return {}

    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef) and node.name == class_name:
            methods: dict[str, list[str]] = {}
            for item in node.body:
                if isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    if item.name.startswith("_"):
                        continue
                    methods[item.name] = [a.arg for a in item.args.args if a.arg != "self"]
            return methods
    return {}


def list_services() -> list[dict[str, Any]]:
    return [{"service_name": name, "sdk_class": qualname} for name, (qualname, _t) in SERVICES.items()]


def list_operations(service_name: str) -> list[str]:
    if service_name not in SERVICES:
        raise KeyError(f"unknown service: {service_name}")
    qualname, _t = SERVICES[service_name]

    try:
        cls = _import_client_class(qualname)
        ops = [name for name, _m in inspect.getmembers(cls, predicate=inspect.isfunction) if not name.startswith("_")]
        if ops:
            return sorted(ops)
    except (ImportError, AttributeError):
        # SDK not importable - fall through to AST discovery. That is the whole
        # condition this clause is for:
        # no ``unitree_sdk2py`` on this machine (ImportError), or an SDK that
        # renamed the client class (AttributeError). Any other exception is a
        # defect in this reader, and swallowing it would answer from a
        # possibly-stale on-disk source - or return an empty list that reads
        # as "this service has no operations" - with no signal either way.
        pass

    return sorted(_ast_methods_for_class(qualname).keys())


def describe_operation(service_name: str, operation_name: str) -> dict[str, Any]:
    if service_name not in SERVICES:
        return {"error": f"unknown service: {service_name}"}

    qualname, _t = SERVICES[service_name]

    try:
        cls = _import_client_class(qualname)
        fn = getattr(cls, operation_name, None)
        if fn is not None and callable(fn):
            try:
                signature = inspect.signature(fn)
            except (TypeError, ValueError):
                # Not introspectable (a C builtin, or a wrapper carrying no
                # __signature__). Reporting ``parameters: []`` here would read
                # as "this operation takes no arguments", so decline the
                # introspected answer and fall through to the AST reader,
                # which can still name the arguments.
                signature = None
            if signature is not None:
                params = []
                for p in signature.parameters.values():
                    if p.name == "self":
                        continue
                    entry: dict[str, Any] = {"name": p.name, "kind": str(p.kind)}
                    if p.annotation is not inspect.Parameter.empty:
                        entry["type"] = getattr(p.annotation, "__name__", str(p.annotation))
                    if p.default is not inspect.Parameter.empty:
                        entry["default"] = (
                            p.default if isinstance(p.default, (str, int, float, bool, type(None))) else str(p.default)
                        )
                    params.append(entry)
                return {
                    "service_name": service_name,
                    "operation_name": operation_name,
                    "signature": f"{operation_name}{signature}",
                    "docstring": inspect.getdoc(fn) or "",
                    "parameters": params,
                    "is_mutative": _is_mutative(operation_name),
                    "is_readonly": _is_readonly(operation_name),
                    "high_danger": (service_name, operation_name) in HIGH_DANGER_OPS,
                    "source": "inspect",
                }
    except (ImportError, AttributeError):
        # SDK not importable - fall through to AST discovery, as in
        # :func:`list_operations`. Anything else
        # is a defect here, and swallowing it would report ``source: "ast"``
        # for an answer whose provenance the caller cannot then trust.
        pass

    ast_methods = _ast_methods_for_class(qualname)
    if operation_name not in ast_methods:
        return {
            "error": f"unknown operation: {service_name}.{operation_name}",
            "available": sorted(ast_methods.keys()),
        }
    arg_names = ast_methods[operation_name]
    return {
        "service_name": service_name,
        "operation_name": operation_name,
        "signature": f"{operation_name}({', '.join(arg_names)})",
        "docstring": "",
        "parameters": [{"name": n, "kind": "POSITIONAL_OR_KEYWORD"} for n in arg_names],
        "is_mutative": _is_mutative(operation_name),
        "is_readonly": _is_readonly(operation_name),
        "high_danger": (service_name, operation_name) in HIGH_DANGER_OPS,
        "source": "ast",
    }


def _normalize_response(result: Any) -> Any:
    if result is None or isinstance(result, (bool, int, float, str)):
        return result
    if isinstance(result, (tuple, list)):
        return [_normalize_response(x) for x in result]
    if isinstance(result, dict):
        return {k: _normalize_response(v) for k, v in result.items()}
    return str(result)[:500]


def _execute(
    service_name: str,
    operation_name: str,
    parameters: dict[str, Any],
    network_interface: str = "eth0",
) -> dict[str, Any]:
    err = ensure_dds(network_interface)
    if err:
        return {"ok": False, "error": err}

    try:
        client = _get_client(service_name)
    except Exception as e:
        return {"ok": False, "error": f"client init failed: {e}"}

    fn = getattr(client, operation_name, None)
    if fn is None or not callable(fn):
        return {
            "ok": False,
            "error": f"unknown operation '{service_name}.{operation_name}'",
            "available_operations": list_operations(service_name),
        }

    try:
        sig = inspect.signature(fn)
        sig.bind(**(parameters or {}))
    except TypeError as e:
        return {
            "ok": False,
            "error": f"parameter mismatch: {e}",
            "expected": describe_operation(service_name, operation_name),
        }

    try:
        with _CALL_LOCK:
            raw = fn(**(parameters or {}))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return {"ok": True, "result": _normalize_response(raw)}


def _gate(service_name: str, operation_name: str, high_danger: bool, tool_context: ToolContext | None) -> str | None:
    """Operator approval for one mutative RPC, before it is dispatched.

    Args:
        service_name: A key of :data:`SERVICES`.
        operation_name: The client method about to be called.
        high_danger: Whether the pair is in :data:`HIGH_DANGER_OPS`; only
            changes the wording the operator sees.
        tool_context: The agent tool context supplying ``interrupt()``.

    Returns:
        A refusal message, or None to let the RPC proceed.
    """
    target = f"{service_name}.{operation_name}"
    what = "can drop or walk the robot" if high_danger else "commands the robot"
    return gate_motion(
        "use_unitree",
        operation_name,
        target,
        f"{target!r} {what}; it needs operator approval before it is sent.",
        tool_context,
        allow_env=COMMAND_ALLOW_ENV,
    )


@tool(context=True)
def use_unitree(
    service_name: str,
    operation_name: str,
    parameters: dict[str, Any] | None = None,
    label: str = "",
    network_interface: str = "eth0",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Universal interface to every Unitree SDK2 client method.

    Like ``use_aws`` but for the Unitree G1. ONE tool covers the entire
    unitree_sdk2_python surface - no per-method @tool wrapper needed.

    SERVICES: loco (FSM/posture/walking), arm (gestures), audio (TTS/LED),
    motion_switcher (controller select), vui (head LED/volume),
    robot_state (service list/switch) - or 'meta' for discovery.

    META OPERATIONS (no DDS, no robot, always safe - use these first!):
      use_unitree('meta', 'list_services')
      use_unitree('meta', 'list_operations', {'service_name': 'loco'})
      use_unitree('meta', 'describe_operation',
                  {'service_name': 'loco', 'operation_name': 'SetFsmId'})

    HIGH-DANGER OPS (flagged in every response): loco.ZeroTorque (robot
    collapses), loco.SetFsmId (fsm_id=0 collapses), loco.SetVelocity /
    loco.Move (walking), motion_switcher.ReleaseMode (uncontrolled).
    Prefer the FSM-gated driver verbs (g1_send_action, g1_run_policy) for
    routine motion; use_unitree is the raw escape hatch.

    OPERATOR APPROVAL: every mutative op (and every high-danger op) stops
    for a human before the RPC is sent. Pre-approve with
    STRANDS_UNITREE_COMMAND_ALLOW=loco.SetVelocity,audio.TtsMaker (or '*');
    BYPASS_TOOL_CONSENT=true lifts the gate. Reads, meta ops and
    loco.StopMove are never gated.

    EXAMPLES:
      use_unitree('audio', 'TtsMaker', {'text': 'Hello', 'speaker_id': 0})
      use_unitree('audio', 'LedControl', {'R': 255, 'G': 0, 'B': 0})
      use_unitree('motion_switcher', 'CheckMode', {})
      use_unitree('loco', 'StopMove', {})   # emergency stop

    Args:
        service_name: One of {loco, arm, audio, motion_switcher, vui,
            robot_state} - or 'meta' for discovery operations.
        operation_name: PascalCase method name on the client class
            (e.g. 'SetFsmId', 'ExecuteAction', 'TtsMaker'), or one of
            {list_services, list_operations, describe_operation} when
            service_name='meta'.
        parameters: Kwargs to pass to the method. For meta ops, the
            lookup target (e.g. {'service_name': 'loco'}).
        label: Optional human-readable description, echoed in the response.
        network_interface: DDS interface. Default 'eth0'.
        tool_context: Supplied by the agent runtime; carries the operator
            interrupt the mutative ops are approved through. Without it a
            mutative op is refused unless pre-approved via
            STRANDS_UNITREE_COMMAND_ALLOW or BYPASS_TOOL_CONSENT=true.

    Returns:
        Dict with status/message plus service, operation, label, result,
        mutative and high_danger flags. On error: the message plus
        available_operations or the expected signature where useful.

        The mutative and high_danger flags are present on BOTH outcomes,
        because a call that failed is the one whose classification a caller
        most needs. An RPC that times out is not evidence the command never
        landed - a loco.SetVelocity answering RPC_CLIENT_API_TIMEOUT is
        consistent with a robot that is walking - and an absent flag cannot be
        told apart from ``False``, so ``.get("high_danger")`` would read a
        failed ZeroTorque exactly like a failed GetFsmId.
    """
    params = parameters or {}
    label = label or f"{service_name}.{operation_name}"

    if service_name == "meta" or operation_name in ("list_services", "list_operations", "describe_operation"):
        try:
            if operation_name == "list_services":
                data: Any = list_services()
            elif operation_name == "list_operations":
                target_svc = params.get("service_name") or service_name
                if target_svc == "meta":
                    return {"status": "error", "message": "list_operations needs a service_name in parameters"}
                data = list_operations(target_svc)
            elif operation_name == "describe_operation":
                target_svc = params.get("service_name") or service_name
                target_op = params.get("operation_name")
                if not target_svc or target_svc == "meta" or not target_op:
                    return {
                        "status": "error",
                        "message": "describe_operation needs parameters {service_name, operation_name}",
                    }
                data = describe_operation(target_svc, target_op)
            else:
                return {
                    "status": "error",
                    "message": f"unknown meta operation '{operation_name}'. Valid: list_services, list_operations, describe_operation",
                }
            return {
                "status": "success",
                "message": f"meta.{operation_name} ok",
                "service": "meta",
                "operation": operation_name,
                "result": data,
            }
        except Exception as e:
            return {"status": "error", "message": f"meta operation failed: {e}"}

    if service_name not in SERVICES:
        return {
            "status": "error",
            "message": f"unknown service '{service_name}'. Valid: {sorted(SERVICES)} (or 'meta' for discovery)",
        }

    high_danger = (service_name, operation_name) in HIGH_DANGER_OPS
    mutative = _is_mutative(operation_name)

    if high_danger or mutative:
        logger.warning(
            "use_unitree: %s.%s is %s",
            service_name,
            operation_name,
            "HIGH_DANGER" if high_danger else "mutative",
        )

    # What the call WAS, as opposed to how it went. This classification is a
    # property of the operation name, so it is known before the RPC is attempted
    # and stays true however the RPC ends - see the Returns section above for why
    # a failed call is the outcome that needs it most.
    classification = {
        "service": service_name,
        "operation": operation_name,
        "label": label,
        "mutative": mutative,
        "high_danger": high_danger,
    }

    if (high_danger or mutative) and (service_name, operation_name) not in _UNGATED_STOP_OPS:
        refusal = _gate(service_name, operation_name, high_danger, tool_context)
        if refusal is not None:
            # Nothing was dispatched: the gate runs before ``_execute`` touches
            # the bus, so a refused ZeroTorque is exactly as inert as a read.
            return {
                "status": "error",
                "message": f"{service_name}.{operation_name} refused: {refusal}",
                "dispatched": False,
                **classification,
            }

    res = _execute(service_name, operation_name, params, network_interface=network_interface)

    if not res.get("ok"):
        return {
            "status": "error",
            "message": f"{service_name}.{operation_name} failed: {res.get('error')}",
            # ``_execute``'s own diagnostics first, so a key it happens to carry
            # cannot shadow the classification.
            **{k: v for k, v in res.items() if k not in ("ok", "error")},
            **classification,
        }

    return {
        "status": "success",
        "message": f"{service_name}.{operation_name} ok",
        "parameters": params,
        "result": res["result"],
        **classification,
    }
