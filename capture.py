"""Measure every mesh.iot posture flag on this tree; dump JSON for the figure."""

import json
import math
import sys
import tempfile
from pathlib import Path

import strands_robots.mesh.iot.bootstrap as bootstrap_mod
import strands_robots.mesh.iot.provision as provision_mod

TREE = str(Path(provision_mod.__file__).parents[3])
print("TREE:", TREE)


class _NotFound(Exception):
    pass


class _Iot:
    class _Exceptions:
        ResourceNotFoundException = _NotFound

    class _Meta:
        region_name = "us-west-2"

    exceptions = _Exceptions()
    meta = _Meta()

    def __init__(self):
        self.policies = {}
        self.attached = []
        self.things = []
        self.certs = 0

    def describe_thing(self, **kw):
        raise _NotFound()

    def create_thing(self, thingName, **kw):
        self.things.append(thingName)
        return {"thingArn": "arn:aws:iot:us-west-2:1:thing/t"}

    def get_policy(self, **kw):
        raise _NotFound()

    def create_policy(self, policyName, policyDocument, **kw):
        self.policies[policyName] = json.loads(policyDocument)
        return {"policyArn": f"arn:aws:iot:us-west-2:1:policy/{policyName}"}

    def list_thing_principals(self, **kw):
        return {"principals": []}

    def create_keys_and_certificate(self, **kw):
        self.certs += 1
        return {
            "certificateArn": "arn:aws:iot:us-west-2:1:cert/abc",
            "certificateId": "abc",
            "certificatePem": "PEM",
            "keyPair": {"PrivateKey": "KEY"},
        }

    def attach_policy(self, policyName, target):
        self.attached.append(policyName)

    def attach_thing_principal(self, **kw):
        pass

    def describe_endpoint(self, **kw):
        return {"endpointAddress": "x.iot.us-west-2.amazonaws.com"}

    def touched(self):
        return bool(self.things or self.policies or self.attached or self.certs)


def provision(value, supplied=True):
    """Provision with allow_estop_publish=value; report the posture reached."""
    iot = _Iot()

    class _B3:
        @staticmethod
        def client(name, region_name=None):
            return iot

    ca, b3 = provision_mod._ensure_ca, provision_mod._require_boto3
    provision_mod._ensure_ca = lambda p: p.write_text("CA")
    provision_mod._require_boto3 = lambda: _B3()
    try:
        with tempfile.TemporaryDirectory() as d:
            kw = {"allow_estop_publish": value} if supplied else {}
            res = provision_mod.provision_robot("fleet-arm-01", cert_dir=Path(d), **kw)
        sids = [st.get("Sid") for st in iot.policies[res.policy_name]["Statement"]]
        return {
            "verdict": "provisioned",
            "policy": res.policy_name,
            "grants_estop": "AllowSafetyEstop" in sids,
            "statements": len(sids),
            "touched": iot.touched(),
        }
    except ValueError as exc:
        return {"verdict": "refused", "detail": str(exc)[:120], "touched": iot.touched()}
    except Exception as exc:  # noqa: BLE001 - the outcome is the measurement
        return {"verdict": f"{type(exc).__name__}", "detail": str(exc)[:120], "touched": iot.touched()}
    finally:
        provision_mod._ensure_ca, provision_mod._require_boto3 = ca, b3


class _Sts:
    class _Meta:
        region_name = "us-west-2"

    meta = _Meta()

    def get_caller_identity(self):
        return {"Account": "111122223333"}


def bootstrap(**kw):
    """Call bootstrap_account; report whether the account create path was entered."""
    entered = []

    class _B3:
        @staticmethod
        def client(name, region_name=None):
            if name == "sts":
                return _Sts()
            entered.append(name)
            raise AssertionError("create path")

    b3 = bootstrap_mod._require_boto3
    bootstrap_mod._require_boto3 = lambda: _B3()
    try:
        bootstrap_mod.bootstrap_account(**kw)
        return {"verdict": "previewed", "entered_create": False}
    except AssertionError:
        return {"verdict": "ENTERED CREATE PATH", "entered_create": True}
    except ValueError as exc:
        return {"verdict": "refused", "detail": str(exc)[:100], "entered_create": bool(entered)}
    except Exception as exc:  # noqa: BLE001
        return {"verdict": type(exc).__name__, "detail": str(exc)[:100], "entered_create": bool(entered)}
    finally:
        bootstrap_mod._require_boto3 = b3


# Values an operator reaches for when opting out, plus the two declared spellings.
ESTOP_CASES = [
    ("True", True),
    ("False", False),
    ('"false"', "false"),
    ('"no"', "no"),
    ('"off"', "off"),
    ('"0"', "0"),
    ("1", 1),
    ("nan", math.nan),
]
BOOT_CASES = [
    ("confirm=True, dry_run=False", {"confirm": True, "dry_run": False}),
    ("confirm=False, dry_run=False", {"confirm": False, "dry_run": False}),
    ('confirm="false", dry_run=False', {"confirm": "false", "dry_run": False}),
    ('confirm="no", dry_run=False', {"confirm": "no", "dry_run": False}),
    ("dry_run=True (default)", {}),
    ('dry_run="false", confirm=False', {"dry_run": "false", "confirm": False}),
]

out = {
    "tree": TREE,
    "estop": [{"label": lab, **provision(v)} for lab, v in ESTOP_CASES],
    "boot": [{"label": lab, **bootstrap(**kw)} for lab, kw in BOOT_CASES],
}
Path(sys.argv[1]).write_text(json.dumps(out, indent=2))

for r in out["estop"]:
    print(f"  estop {r['label']:<10} -> {r['verdict']:<12} policy={r.get('policy', '-'):<26} "
          f"grants={r.get('grants_estop')} touched={r['touched']}")
for r in out["boot"]:
    print(f"  boot  {r['label']:<32} -> {r['verdict']}")
