"""Every resource the IoT bootstrap ensures lands in its created/skipped ledger.

``BootstrappedAccount.created`` + ``.skipped`` are the only record an operator
has of what a provisioning run touched, and the closing log line counts them.
Three helpers were ensuring a resource without recording it:

* ``_grant_iot_invoke_lambda`` recorded neither the ``lambda:InvokeFunction``
  grant it created for the E-stop fan-out nor the one it found already present,
  while its sibling ``_grant_iot_invoke_provisioning_hook`` recorded both. A
  fresh bootstrap created thirteen resources and reported twelve, and the one
  missing from the audit trail was a permission grant.
* ``_ensure_iot_action_role`` and ``_ensure_provisioning_role`` recorded the role
  they create but not the existing role they reuse, so a resumed bootstrap
  reported fewer reused resources than it had reused.

The end-to-end classes drive the real helpers against fake AWS clients, which is
what makes the ledger counts observable; ``TestTheLedgerContractHoldsForEveryHelper``
pins the invariant structurally so a helper added later cannot drift out of it.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Collection
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh.iot import bootstrap as boot_mod
from strands_robots.mesh.iot.bootstrap import (
    ESTOP_INVOKE_PERMISSION,
    ESTOP_LAMBDA_NAME,
    ESTOP_LAMBDA_ROLE,
    IOT_ACTION_ROLE,
    LOG_GROUP_NAME,
    PROVISIONING_HOOK_INVOKE_PERMISSION,
    PROVISIONING_HOOK_LAMBDA_NAME,
    PROVISIONING_HOOK_ROLE,
    PROVISIONING_ROLE,
    PROVISIONING_TEMPLATE,
    RULE_ESTOP_FANOUT,
    RULE_SAFETY_TO_DYNAMODB,
    SAFETY_TABLE_NAME,
    BootstrappedAccount,
)

ACCOUNT = "111122223333"
REGION = "us-west-2"

# The ledger entry each resource of the fake account is recorded under. Built
# from the module's own name constants so a renamed resource updates both the
# fixture and the expectation together.
LEDGER_NAME = {
    "log": f"logs:{LOG_GROUP_NAME}",
    "table": f"dynamodb:{SAFETY_TABLE_NAME}",
    "lam_role": f"iam:{ESTOP_LAMBDA_ROLE}",
    "estop_lambda": f"lambda:{ESTOP_LAMBDA_NAME}",
    "action_role": f"iam:{IOT_ACTION_ROLE}",
    "safety_rule": f"iot-rule:{RULE_SAFETY_TO_DYNAMODB}",
    "estop_rule": f"iot-rule:{RULE_ESTOP_FANOUT}",
    "estop_permission": ESTOP_INVOKE_PERMISSION,
    "hook_role": f"iam:{PROVISIONING_HOOK_ROLE}",
    "hook_lambda": f"lambda:{PROVISIONING_HOOK_LAMBDA_NAME}",
    "prov_role": f"iam:{PROVISIONING_ROLE}",
    "prov_template": f"iot-prov-template:{PROVISIONING_TEMPLATE}",
    "hook_permission": PROVISIONING_HOOK_INVOKE_PERMISSION,
}

ROLE_KEY = {
    ESTOP_LAMBDA_ROLE: "lam_role",
    IOT_ACTION_ROLE: "action_role",
    PROVISIONING_HOOK_ROLE: "hook_role",
    PROVISIONING_ROLE: "prov_role",
}
LAMBDA_KEY = {
    ESTOP_LAMBDA_NAME: "estop_lambda",
    PROVISIONING_HOOK_LAMBDA_NAME: "hook_lambda",
}
RULE_KEY = {
    RULE_SAFETY_TO_DYNAMODB: "safety_rule",
    RULE_ESTOP_FANOUT: "estop_rule",
}
PERMISSION_KEY = {
    "strands-mesh-iot-invoke": "estop_permission",
    "strands-mesh-iot-provisioning-invoke": "hook_permission",
}


def _exceptions(*names: str) -> SimpleNamespace:
    """A boto3-style ``client.exceptions`` namespace of real exception classes.

    ``MagicMock`` cannot stand in here: ``except <MagicMock>`` raises
    ``TypeError: catching classes that do not inherit from BaseException``.
    """
    return SimpleNamespace(**{n: type(n, (Exception,), {}) for n in names})


class _FakeAws:
    """boto3 stand-in whose account already holds the resources in ``present``.

    Keys are those of :data:`LEDGER_NAME`. Every client is a ``MagicMock`` with
    the describe/get calls wired to answer for that account, so the real
    ``_ensure_*`` / ``_grant_*`` helpers run unmodified and the ledger they build
    is the value under test.
    """

    def __init__(self, present: Collection[str], stale: Collection[str] = ()) -> None:
        self.present = set(present)
        # Lambdas whose deployed description carries no current version tag.
        # _ensure_estop_lambda only honours force_update for a stale
        # deployment, so its update path is unreachable without this.
        self.stale = set(stale)
        self.requested: list[str] = []

    # boto3 module surface
    def client(self, svc: str, **_: Any) -> Any:
        self.requested.append(svc)
        return getattr(self, f"_{svc.replace('-', '_')}")()

    def _sts(self) -> Any:
        c = MagicMock(name="sts")
        c.get_caller_identity.return_value = {"Account": ACCOUNT}
        c.meta.region_name = REGION
        return c

    def _iam(self) -> Any:
        c = MagicMock(name="iam")
        c.exceptions = _exceptions("NoSuchEntityException")

        def get_role(RoleName: str, **_kw: Any) -> Any:  # noqa: N803 - boto3 casing
            if ROLE_KEY[RoleName] not in self.present:
                raise c.exceptions.NoSuchEntityException(RoleName)
            return {"Role": {"Arn": f"arn:aws:iam::{ACCOUNT}:role/{RoleName}"}}

        c.get_role.side_effect = get_role
        c.create_role.side_effect = lambda RoleName, **_kw: {  # noqa: N803
            "Role": {"Arn": f"arn:aws:iam::{ACCOUNT}:role/{RoleName}"}
        }
        return c

    def _lambda(self) -> Any:
        c = MagicMock(name="lambda")
        c.exceptions = _exceptions(
            "ResourceNotFoundException",
            "ResourceConflictException",
            "InvalidParameterValueException",
        )
        descriptions = {
            ESTOP_LAMBDA_NAME: f"strands-mesh: E-stop [v{boot_mod._LAMBDA_VERSION}]",
            PROVISIONING_HOOK_LAMBDA_NAME: f"strands-mesh: hook [v{boot_mod._PROVISIONING_HOOK_VERSION}]",
        }
        stale_descriptions = {
            ESTOP_LAMBDA_NAME: "strands-mesh: E-stop [v0]",
            PROVISIONING_HOOK_LAMBDA_NAME: "strands-mesh: hook [v0]",
        }

        def get_function(FunctionName: str, **_kw: Any) -> Any:  # noqa: N803
            if LAMBDA_KEY[FunctionName] not in self.present:
                raise c.exceptions.ResourceNotFoundException(FunctionName)
            table = stale_descriptions if LAMBDA_KEY[FunctionName] in self.stale else descriptions
            return {
                "Configuration": {
                    "Description": table[FunctionName],
                    "FunctionArn": f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:{FunctionName}",
                }
            }

        def add_permission(StatementId: str, **_kw: Any) -> Any:  # noqa: N803
            if PERMISSION_KEY[StatementId] in self.present:
                raise c.exceptions.ResourceConflictException(StatementId)
            return {"Statement": StatementId}

        c.get_function.side_effect = get_function
        c.add_permission.side_effect = add_permission
        c.create_function.side_effect = lambda FunctionName, **_kw: {  # noqa: N803
            "FunctionArn": f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:{FunctionName}"
        }
        return c

    def _dynamodb(self) -> Any:
        c = MagicMock(name="dynamodb")
        c.exceptions = _exceptions("ResourceNotFoundException")
        arn = f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/{SAFETY_TABLE_NAME}"

        def describe_table(**_kw: Any) -> Any:
            if "table" not in self.present:
                raise c.exceptions.ResourceNotFoundException(SAFETY_TABLE_NAME)
            return {"Table": {"TableArn": arn}}

        c.describe_table.side_effect = describe_table
        c.create_table.return_value = {"TableDescription": {"TableArn": arn}}
        return c

    def _iot(self) -> Any:
        c = MagicMock(name="iot")
        c.exceptions = _exceptions(
            "ResourceNotFoundException",
            "UnauthorizedException",
            "InvalidRequestException",
        )

        def get_topic_rule(ruleName: str, **_kw: Any) -> Any:  # noqa: N803
            if RULE_KEY[ruleName] not in self.present:
                raise c.exceptions.ResourceNotFoundException(ruleName)
            return {"ruleArn": f"arn:aws:iot:{REGION}:{ACCOUNT}:rule/{ruleName}"}

        def describe_provisioning_template(**_kw: Any) -> Any:
            if "prov_template" not in self.present:
                raise c.exceptions.ResourceNotFoundException(PROVISIONING_TEMPLATE)
            return {"templateName": PROVISIONING_TEMPLATE}

        c.get_topic_rule.side_effect = get_topic_rule
        c.describe_provisioning_template.side_effect = describe_provisioning_template
        return c

    def _logs(self) -> Any:
        c = MagicMock(name="logs")
        group = {
            "logGroupName": LOG_GROUP_NAME,
            "arn": f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:{LOG_GROUP_NAME}",
        }

        def describe_log_groups(**_kw: Any) -> Any:
            # After create_log_group the helper describes again for the ARN, so
            # the group has to exist from that point on.
            return {"logGroups": [group] if "log" in self.present else []}

        def create_log_group(**_kw: Any) -> Any:
            self.present.add("log")
            return {}

        c.describe_log_groups.side_effect = describe_log_groups
        c.create_log_group.side_effect = create_log_group
        return c


@pytest.fixture
def no_sleep(monkeypatch):
    """IAM propagation waits total 23s across the bootstrap; skip them."""
    monkeypatch.setattr(boot_mod.time, "sleep", lambda *_a, **_k: None)


def _bootstrap(monkeypatch, present: set[str]) -> BootstrappedAccount:
    fake = _FakeAws(present)
    monkeypatch.setattr(boot_mod, "_require_boto3", lambda: fake)
    return boot_mod.bootstrap_account(region=REGION, confirm=True, dry_run=False)


# Every resource a bootstrap of an empty account brings up, in the order
# bootstrap_account ensures them.
FRESH_ACCOUNT_RESOURCES = (
    "log",
    "table",
    "lam_role",
    "estop_lambda",
    "action_role",
    "safety_rule",
    "estop_rule",
    "estop_permission",
    "hook_role",
    "hook_lambda",
    "prov_role",
    "prov_template",
    "hook_permission",
)

# A fully provisioned account. The two roles nested inside the safety rule and
# the provisioning template are absent because those parents return before
# reaching them, so this run does not ensure them at all.
FULL_ACCOUNT_RESOURCES = tuple(k for k in FRESH_ACCOUNT_RESOURCES if k not in {"action_role", "prov_role"})


class TestEveryResourceABootstrapEnsuresIsRecorded:
    """The ledger accounts for every resource, on the create and the reuse path."""

    def test_a_fresh_account_records_every_resource_it_creates(self, monkeypatch, no_sleep):
        out = _bootstrap(monkeypatch, present=set())

        expected = [LEDGER_NAME[k] for k in FRESH_ACCOUNT_RESOURCES]
        assert out.created == expected
        assert out.skipped == []

    def test_a_fully_provisioned_account_records_every_resource_it_reuses(self, monkeypatch, no_sleep):
        out = _bootstrap(monkeypatch, present=set(LEDGER_NAME))

        assert sorted(out.skipped) == sorted(LEDGER_NAME[k] for k in FULL_ACCOUNT_RESOURCES)
        assert out.created == []

    def test_a_resumed_bootstrap_records_the_roles_it_reuses(self, monkeypatch, no_sleep):
        # The realistic re-run: an earlier attempt left the roles and Lambdas
        # behind but died before the rules and the provisioning template. The
        # two nested roles are reached here, on their reuse path.
        present = set(LEDGER_NAME) - {"safety_rule", "estop_rule", "prov_template"}
        out = _bootstrap(monkeypatch, present=present)

        assert LEDGER_NAME["action_role"] in out.skipped
        assert LEDGER_NAME["prov_role"] in out.skipped
        assert LEDGER_NAME["estop_permission"] in out.skipped
        # The rules it did have to create are recorded as created, not reused.
        assert sorted(out.created) == sorted(LEDGER_NAME[k] for k in ("safety_rule", "estop_rule", "prov_template"))

    @pytest.mark.parametrize(
        "present",
        [
            pytest.param(set(), id="fresh"),
            pytest.param(set(LEDGER_NAME), id="fully-provisioned"),
            pytest.param(set(LEDGER_NAME) - {"safety_rule", "prov_template"}, id="resumed"),
        ],
    )
    def test_no_resource_is_recorded_as_both_created_and_reused(self, monkeypatch, no_sleep, present):
        out = _bootstrap(monkeypatch, present=present)

        assert set(out.created) & set(out.skipped) == set()
        assert len(out.created) == len(set(out.created))
        assert len(out.skipped) == len(set(out.skipped))


class TestTheGrantHelpersRecordTheirPermission:
    """Both ``add_permission`` helpers record the statement, either way."""

    @pytest.mark.parametrize(
        ("helper", "statement_id", "entry"),
        [
            ("_grant_iot_invoke_lambda", "strands-mesh-iot-invoke", ESTOP_INVOKE_PERMISSION),
            (
                "_grant_iot_invoke_provisioning_hook",
                "strands-mesh-iot-provisioning-invoke",
                PROVISIONING_HOOK_INVOKE_PERMISSION,
            ),
        ],
    )
    def test_a_granted_permission_is_recorded_as_created(self, helper, statement_id, entry):
        lam = MagicMock()
        lam.exceptions = _exceptions("ResourceConflictException")
        account = BootstrappedAccount(region=REGION, account_id=ACCOUNT)

        getattr(boot_mod, helper)(lam, "arn:aws:lambda:::function:f", account)

        assert lam.add_permission.call_args.kwargs["StatementId"] == statement_id
        assert account.created == [entry]
        assert account.skipped == []

    @pytest.mark.parametrize(
        ("helper", "entry"),
        [
            ("_grant_iot_invoke_lambda", ESTOP_INVOKE_PERMISSION),
            ("_grant_iot_invoke_provisioning_hook", PROVISIONING_HOOK_INVOKE_PERMISSION),
        ],
    )
    def test_an_already_granted_permission_is_recorded_as_reused(self, helper, entry):
        lam = MagicMock()
        lam.exceptions = _exceptions("ResourceConflictException")
        lam.add_permission.side_effect = lam.exceptions.ResourceConflictException("exists")
        account = BootstrappedAccount(region=REGION, account_id=ACCOUNT)

        getattr(boot_mod, helper)(lam, "arn:aws:lambda:::function:f", account)

        assert account.skipped == [entry]
        assert account.created == []


class TestTheRoleHelpersRecordAReusedRole:
    """Every ``_ensure_*_role`` helper records the role it finds already there."""

    @pytest.mark.parametrize(
        ("helper", "role_name"),
        [
            ("_ensure_lambda_role", ESTOP_LAMBDA_ROLE),
            ("_ensure_iot_action_role", IOT_ACTION_ROLE),
            ("_ensure_provisioning_hook_role", PROVISIONING_HOOK_ROLE),
            ("_ensure_provisioning_role", PROVISIONING_ROLE),
        ],
    )
    def test_an_existing_role_is_recorded_as_reused_and_its_arn_returned(self, helper, role_name):
        arn = f"arn:aws:iam::{ACCOUNT}:role/{role_name}"
        iam = MagicMock()
        iam.exceptions = _exceptions("NoSuchEntityException")
        iam.get_role.return_value = {"Role": {"Arn": arn}}
        account = BootstrappedAccount(region=REGION, account_id=ACCOUNT)

        result = getattr(boot_mod, helper)(iam, account)

        assert result == arn
        assert account.skipped == [f"iam:{role_name}"]
        assert account.created == []
        iam.create_role.assert_not_called()


class TestTheLedgerContractHoldsForEveryHelper:
    """Structural: no helper can mutate the account without recording it.

    A behavioural test only covers the helpers a scenario reaches. This grades
    the module itself, so a helper added later inherits the contract.
    """

    MUTATING_CALLS = (
        "create_",
        "put_",
        "add_permission",
        "attach_",
        "update_",
        "register_",
        "tag_",
    )

    def _mutating_helpers(self) -> dict[str, str]:
        """``{function name: unparsed body}`` for every helper that mutates AWS."""
        tree = ast.parse(Path(inspect.getfile(boot_mod)).read_text(encoding="utf-8"))
        out = {}
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if not (node.name.startswith("_ensure") or node.name.startswith("_grant")):
                continue
            body = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
            if any(call in body for call in self.MUTATING_CALLS):
                out[node.name] = body
        return out

    def test_the_scan_reaches_every_helper(self):
        helpers = self._mutating_helpers()
        # Non-vacuity: an empty or near-empty scan would pass the rule below
        # while measuring nothing.
        assert len(helpers) >= 13, sorted(helpers)
        assert "_grant_iot_invoke_lambda" in helpers
        assert "_ensure_provisioning_role" in helpers

    def test_every_mutating_helper_records_both_outcomes(self):
        missing = {
            name: {
                "created": "created.append" in body,
                "skipped": "skipped.append" in body,
            }
            for name, body in self._mutating_helpers().items()
            if not ("created.append" in body and "skipped.append" in body)
        }
        assert missing == {}, (
            "every helper that ensures a resource must record it in "
            f"BootstrappedAccount on both its create and its reuse path: {missing}"
        )

    def test_the_permission_ledger_names_are_single_sourced(self):
        # A permission statement has no ARN, so its ledger name is its only
        # handle; a repeated literal would let the two append sites disagree.
        source = Path(inspect.getfile(boot_mod)).read_text(encoding="utf-8")
        for name, value in (
            ("ESTOP_INVOKE_PERMISSION", ESTOP_INVOKE_PERMISSION),
            ("PROVISIONING_HOOK_INVOKE_PERMISSION", PROVISIONING_HOOK_INVOKE_PERMISSION),
        ):
            assert f'{name} = "{value}"' in source
            assert source.count(f'"{value}"') == 1, f"{value!r} is spelled out more than once"


class TestTheLedgerIsNotWidened:
    """Boundaries this change deliberately leaves alone."""

    def test_a_dry_run_records_nothing(self, monkeypatch, capsys):
        fake = _FakeAws(set())
        monkeypatch.setattr(boot_mod, "_require_boto3", lambda: fake)

        out = boot_mod.bootstrap_account(region=REGION)  # dry_run defaults True

        # A preview ensures nothing, so it records nothing - the printed
        # preview is its output, not the ledger.
        assert out.created == []
        assert out.skipped == []
        assert "[dry_run]" in capsys.readouterr().err

    def test_an_updated_lambda_is_recorded_as_created_not_reused(self, monkeypatch, no_sleep):
        # force_update replaces the deployed code, so it is a change to the
        # account and belongs in created under the existing "(updated)"
        # spelling, not in skipped. Both Lambdas are stale here because
        # _ensure_estop_lambda only honours force_update for a stale
        # deployment (_ensure_provisioning_hook_lambda updates either way).
        fake = _FakeAws(set(LEDGER_NAME), stale={"estop_lambda", "hook_lambda"})
        monkeypatch.setattr(boot_mod, "_require_boto3", lambda: fake)

        out = boot_mod.bootstrap_account(region=REGION, confirm=True, dry_run=False, force_update=True)

        for name in (ESTOP_LAMBDA_NAME, PROVISIONING_HOOK_LAMBDA_NAME):
            assert f"lambda:{name} (updated)" in out.created
            assert f"lambda:{name}" not in out.skipped

    def test_a_stale_lambda_left_alone_is_still_recorded_as_reused(self, monkeypatch, no_sleep):
        # Without force_update a stale deployment is warned about but not
        # touched, so it stays a reuse - the ledger reports what happened,
        # not what the operator was advised to do.
        fake = _FakeAws(set(LEDGER_NAME), stale={"estop_lambda", "hook_lambda"})
        monkeypatch.setattr(boot_mod, "_require_boto3", lambda: fake)

        out = boot_mod.bootstrap_account(region=REGION, confirm=True, dry_run=False)

        for name in (ESTOP_LAMBDA_NAME, PROVISIONING_HOOK_LAMBDA_NAME):
            assert f"lambda:{name}" in out.skipped
            assert f"lambda:{name} (updated)" not in out.created


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
