"""Regression tests: both contact readers identify a contact the same way.

Two methods report contacts and each holds something the other does not:

* ``get_contacts``      - geometry, ``normal_force``, ``exclude``, ``body1``/``body2``
* ``get_contact_forces`` - the only source of ``friction_force`` / ``full_wrench``

so a slip check has to read the tangential force from the second and then say WHICH
body pair slipped. That was impossible: for UNNAMED geoms (most robot-link collision
geoms are unnamed) the two disagreed about the contact's identity, and the second
carried no body fields at all:

    get_contacts        geom1='a/link5/geom_31'   body1='a/link5'
    get_contact_forces  geom1='geom_31'           (no body1/body2 keys)

Same contact, two identities. Records could not be joined, per-body force could not
be summed, and a caller matching on a body name found nothing in the force reader.

``get_contact_forces`` now labels geoms via the same ``<body>/geom_<id>`` fallback and
reports ``body1``/``body2``, so the two agree pair-for-pair.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="contact_reader_identity_parity", mesh=False)
    s.create_world()
    assert s.add_robot(name="a", data_config="panda")["status"] == "success"
    assert s.add_object(name="b", shape="box", size=[0.12] * 3, position=[0, 0, 0.061], mass=2.0)["status"] == "success"
    assert s.step(n_steps=3000)["status"] == "success"
    yield s
    s.destroy()


def _json(result):
    payload = [b["json"] for b in result["content"] if "json" in b][0]
    return payload.get("contacts", payload)


def _pairs(records, a="geom1", b="geom2"):
    return {tuple(sorted((r[a], r[b]))) for r in records}


def test_both_readers_see_the_same_contacts(sim) -> None:
    plain = _json(sim.get_contacts())
    forces = _json(sim.get_contact_forces())
    assert len(plain) == len(forces)


def test_geom_labels_agree(sim) -> None:
    """The core defect: 'a/link5/geom_31' vs bare 'geom_31'."""
    plain = _json(sim.get_contacts())
    forces = _json(sim.get_contact_forces())
    assert _pairs(plain) == _pairs(forces)


def test_the_force_reader_reports_bodies(sim) -> None:
    forces = _json(sim.get_contact_forces())
    assert forces, "expected active contacts"
    for record in forces:
        assert "body1" in record and "body2" in record


def test_body_labels_agree(sim) -> None:
    plain = _json(sim.get_contacts())
    forces = _json(sim.get_contact_forces())
    assert _pairs(plain, "body1", "body2") == _pairs(forces, "body1", "body2")


def test_an_unnamed_geom_is_body_prefixed(sim) -> None:
    """Robot-link collision geoms are unnamed; a bare id is unjoinable."""
    forces = _json(sim.get_contact_forces())
    synthesized = [r for r in forces if "/geom_" in r["geom1"] or "/geom_" in r["geom2"]]
    assert synthesized, "expected at least one unnamed robot geom in self-contact"
    for record in synthesized:
        for key in ("geom1", "geom2"):
            if "/geom_" in record[key]:
                body = record[key].rsplit("/geom_", 1)[0]
                assert mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0, body


def test_normal_forces_agree_pair_for_pair(sim) -> None:
    def summed(records):
        out: dict[tuple[str, str], float] = {}
        for r in records:
            key = tuple(sorted((r["geom1"], r["geom2"])))
            out[key] = out.get(key, 0.0) + abs(float(r["normal_force"]))
        return out

    plain, forces = summed(_json(sim.get_contacts())), summed(_json(sim.get_contact_forces()))
    assert set(plain) == set(forces)
    for key in plain:
        assert plain[key] == pytest.approx(forces[key], abs=1e-6), key


def test_a_per_body_slip_check_is_now_possible(sim) -> None:
    """The end-to-end point: friction and identity readable from one record.

    A 3 N push on a 2 kg block that cannot slide (mu*m*g = 19.6 N) must show a
    3 N tangential reaction on the block/ground pair alongside its 19.62 N normal.
    """
    assert sim.apply_force(body_name="b", force=[3.0, 0.0, 0.0])["status"] == "success"
    assert sim.step(n_steps=1500)["status"] == "success"

    aggregate: dict[tuple[str, str], list[float]] = {}
    for record in _json(sim.get_contact_forces()):
        key = tuple(sorted((record["body1"], record["body2"])))
        entry = aggregate.setdefault(key, [0.0, 0.0])
        entry[0] += float(np.linalg.norm(record["friction_force"]))
        entry[1] += abs(float(record["normal_force"]))

    ground_pair = tuple(sorted(("b", "world")))
    assert ground_pair in aggregate, aggregate.keys()
    friction, normal = aggregate[ground_pair]
    assert friction == pytest.approx(3.0, abs=0.05), friction
    assert normal == pytest.approx(2.0 * 9.81, abs=0.05), normal


def test_the_full_wrench_is_still_reported(sim) -> None:
    """Do not regress the fields the force reader already had."""
    for record in _json(sim.get_contact_forces()):
        assert len(record["full_wrench"]) == 6
        assert len(record["friction_force"]) == 2
        assert bool(np.all(np.isfinite(record["full_wrench"])))
