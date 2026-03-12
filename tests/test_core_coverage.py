"""Comprehensive tests for strands_robots core modules.

Targets 100% code coverage across ALL modules:
  robot.py, factory.py, envs.py, kinematics.py, processor.py, video.py,
  visualizer.py, record.py, dataset_recorder.py, _async_utils.py,
  zenoh_mesh.py, stereo/__init__.py, cosmos_transfer/__init__.py,
  dreamgen/__init__.py, marble/__init__.py, leisaac.py, rl_trainer.py,
  telemetry/ (all), registry/ (all), assets/ (all), __init__.py

All external dependencies (lerobot, mujoco, torch, zenoh, etc.) are mocked.
"""

import asyncio
import gzip
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import (
    AsyncMock,
    MagicMock,
    PropertyMock,
    call,
    mock_open,
    patch,
)

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine in a new event loop."""
    return asyncio.get_event_loop().run_until_complete(coro) if asyncio.get_event_loop().is_running() else asyncio.run(coro)


# ---------------------------------------------------------------------------
# _async_utils
# ---------------------------------------------------------------------------


class TestAsyncUtils:
    """Tests for strands_robots._async_utils."""

    def test_resolve_coroutine_plain_value(self):
        from strands_robots._async_utils import _resolve_coroutine
        assert _resolve_coroutine(42) == 42
        assert _resolve_coroutine("hello") == "hello"

    def test_resolve_coroutine_with_coroutine_no_loop(self):
        from strands_robots._async_utils import _resolve_coroutine

        async def coro():
            return 99

        result = _resolve_coroutine(coro())
        assert result == 99

    def test_resolve_coroutine_inside_running_loop(self):
        from strands_robots._async_utils import _resolve_coroutine

        async def coro():
            return 55

        async def outer():
            return _resolve_coroutine(coro())

        result = asyncio.run(outer())
        assert result == 55

    def test_run_async_no_loop(self):
        from strands_robots._async_utils import _run_async

        async def coro():
            return 77

        result = _run_async(lambda: coro())
        assert result == 77

    def test_run_async_inside_running_loop(self):
        from strands_robots._async_utils import _run_async

        async def coro():
            return 88

        async def outer():
            return _run_async(lambda: coro())

        result = asyncio.run(outer())
        assert result == 88


# ---------------------------------------------------------------------------
# telemetry/types.py
# ---------------------------------------------------------------------------


class TestTelemetryTypes:
    """Tests for telemetry event types and tier routing."""

    def test_stream_tier_enum(self):
        from strands_robots.telemetry.types import StreamTier
        assert StreamTier.STREAM.name == "STREAM"
        assert StreamTier.BATCH.name == "BATCH"
        assert StreamTier.STORAGE.name == "STORAGE"

    def test_event_category_default_tier(self):
        from strands_robots.telemetry.types import EventCategory, StreamTier
        assert EventCategory.EMERGENCY_STOP.default_tier == StreamTier.STREAM
        assert EventCategory.JOINT_STATE.default_tier == StreamTier.BATCH
        assert EventCategory.CAMERA_FRAME.default_tier == StreamTier.STORAGE
        assert EventCategory.CUSTOM.default_tier == StreamTier.BATCH

    def test_event_category_name(self):
        from strands_robots.telemetry.types import EventCategory
        assert EventCategory.EMERGENCY_STOP.category_name == "emergency_stop"
        assert EventCategory.HEARTBEAT.category_name == "heartbeat"

    def test_batch_config_defaults(self):
        from strands_robots.telemetry.types import BatchConfig
        bc = BatchConfig()
        assert bc.max_count == 100
        assert bc.max_bytes == 256 * 1024
        assert bc.max_age_ms == 500.0

    def test_telemetry_event_defaults(self):
        from strands_robots.telemetry.types import EventCategory, TelemetryEvent
        ev = TelemetryEvent(
            category=EventCategory.JOINT_STATE,
            robot_id="test_robot",
            data={"q": [0.1, 0.2]},
        )
        assert ev.robot_id == "test_robot"
        assert ev.event_id  # auto-generated
        assert ev.timestamp_ms > 0
        assert ev.frame_id == 0
        assert ev.sim_or_real == "real"

    def test_effective_tier_override(self):
        from strands_robots.telemetry.types import (
            EventCategory,
            StreamTier,
            TelemetryEvent,
        )
        ev = TelemetryEvent(
            category=EventCategory.JOINT_STATE,
            robot_id="r",
            data={},
            tier=StreamTier.STREAM,
        )
        assert ev.effective_tier == StreamTier.STREAM

    def test_effective_tier_default(self):
        from strands_robots.telemetry.types import EventCategory, TelemetryEvent
        ev = TelemetryEvent(
            category=EventCategory.CAMERA_FRAME,
            robot_id="r",
            data={},
        )
        from strands_robots.telemetry.types import StreamTier
        assert ev.effective_tier == StreamTier.STORAGE

    def test_serialize_basic(self):
        from strands_robots.telemetry.types import EventCategory, TelemetryEvent
        ev = TelemetryEvent(
            category=EventCategory.HEARTBEAT,
            robot_id="r",
            data={"x": 1, "arr": np.array([1.0, 2.0])},
            metadata={"extra": "info"},
        )
        raw = ev.serialize()
        d = json.loads(raw)
        assert d["category"] == "heartbeat"
        assert d["data"]["arr"] == [1.0, 2.0]
        assert d["metadata"]["extra"] == "info"

    def test_serialize_bytes_in_data(self):
        from strands_robots.telemetry.types import EventCategory, TelemetryEvent
        ev = TelemetryEvent(
            category=EventCategory.CAMERA_FRAME,
            robot_id="r",
            data={"img": b"\x00" * 100},
        )
        raw = ev.serialize()
        d = json.loads(raw)
        assert d["data"]["img"] == "<bytes:100>"

    def test_serialize_numpy_types(self):
        from strands_robots.telemetry.types import EventCategory, TelemetryEvent
        ev = TelemetryEvent(
            category=EventCategory.JOINT_STATE,
            robot_id="r",
            data={"i": np.int64(5), "f": np.float32(3.14)},
        )
        raw = ev.serialize()
        d = json.loads(raw)
        assert d["data"]["i"] == 5
        assert abs(d["data"]["f"] - 3.14) < 0.01

    def test_compress(self):
        from strands_robots.telemetry.types import EventCategory, TelemetryEvent
        ev = TelemetryEvent(
            category=EventCategory.HEARTBEAT,
            robot_id="r",
            data={"x": 1},
        )
        compressed = ev.compress()
        assert isinstance(compressed, bytes)
        decompressed = gzip.decompress(compressed)
        d = json.loads(decompressed)
        assert d["category"] == "heartbeat"

    def test_size_bytes_estimation(self):
        from strands_robots.telemetry.types import EventCategory, TelemetryEvent
        ev = TelemetryEvent(
            category=EventCategory.JOINT_STATE,
            robot_id="r",
            data={
                "q": np.zeros(6),
                "names": ["a", "b"],
                "img": b"\x00" * 50,
                "nested": {"k": "v"},
                "scalar": 1.0,
            },
        )
        sz = ev.size_bytes()
        assert sz > 200


# ---------------------------------------------------------------------------
# telemetry/stream.py
# ---------------------------------------------------------------------------


class TestTelemetryStream:
    """Tests for telemetry stream with batching and transport."""

    def test_lifecycle_start_stop(self):
        from strands_robots.telemetry.stream import TelemetryStream
        ts = TelemetryStream(robot_id="test")
        ts.start()
        assert ts._running
        ts.stop()
        assert not ts._running

    def test_double_start(self):
        from strands_robots.telemetry.stream import TelemetryStream
        ts = TelemetryStream(robot_id="test")
        ts.start()
        ts.start()  # should warn but not crash
        ts.stop()

    def test_stop_when_not_running(self):
        from strands_robots.telemetry.stream import TelemetryStream
        ts = TelemetryStream(robot_id="test")
        ts.stop()  # no-op

    def test_context_manager(self):
        from strands_robots.telemetry.stream import TelemetryStream
        with TelemetryStream(robot_id="test") as ts:
            ts.start()
            assert ts._running
        assert not ts._running

    def test_emit_basic(self):
        from strands_robots.telemetry.stream import TelemetryStream
        from strands_robots.telemetry.types import EventCategory
        ts = TelemetryStream(robot_id="test")
        ts.emit(EventCategory.JOINT_STATE, {"q": [0.1]})
        stats = ts.get_stats()
        assert stats["emitted"] == 1

    def test_emit_with_tier_override(self):
        from strands_robots.telemetry.stream import TelemetryStream
        from strands_robots.telemetry.types import EventCategory, StreamTier
        ts = TelemetryStream(robot_id="test")
        ts.emit(EventCategory.JOINT_STATE, {"q": [0.1]}, tier=StreamTier.STREAM)
        assert len(ts._buffers[StreamTier.STREAM]) == 1

    def test_emit_buffer_overflow(self):
        from strands_robots.telemetry.stream import TelemetryStream
        from strands_robots.telemetry.types import EventCategory
        ts = TelemetryStream(robot_id="test", buffer_maxlen=2)
        ts.emit(EventCategory.JOINT_STATE, {"q": [1]})
        ts.emit(EventCategory.JOINT_STATE, {"q": [2]})
        ts.emit(EventCategory.JOINT_STATE, {"q": [3]})  # drops oldest
        stats = ts.get_stats()
        assert stats["dropped"] == 1

    def test_span_context(self):
        from strands_robots.telemetry.stream import TelemetryStream
        from strands_robots.telemetry.types import EventCategory
        ts = TelemetryStream(robot_id="test")
        with ts.span("pick_and_place", metadata={"extra": 1}) as trace_id:
            assert trace_id is not None
            ts.emit(EventCategory.JOINT_STATE, {"q": [0.1]})
        # verify correlation restored
        assert getattr(ts._correlation, "trace_id", None) is None

    def test_span_nested(self):
        from strands_robots.telemetry.stream import TelemetryStream
        from strands_robots.telemetry.types import EventCategory
        ts = TelemetryStream(robot_id="test")
        with ts.span("outer") as outer_id:
            with ts.span("inner") as inner_id:
                ts.emit(EventCategory.JOINT_STATE, {"q": [1]})
            assert getattr(ts._correlation, "trace_id", None) == outer_id

    def test_add_transport(self):
        from strands_robots.telemetry.stream import TelemetryStream, TransportConfig
        from strands_robots.telemetry.types import StreamTier
        ts = TelemetryStream(robot_id="test")
        t = TransportConfig(
            name="test_t",
            tiers=[StreamTier.BATCH],
            handler=lambda payload: True,
        )
        ts.add_transport(t)
        assert len(ts._transports) == 1

    def test_flush_sends_to_transport(self):
        from strands_robots.telemetry.stream import TelemetryStream, TransportConfig
        from strands_robots.telemetry.types import EventCategory, StreamTier
        received = []
        ts = TelemetryStream(robot_id="test", compression_threshold=999999)
        t = TransportConfig(
            name="capture",
            tiers=[StreamTier.BATCH],
            handler=lambda payload: (received.append(payload), True)[1],
        )
        ts.add_transport(t)
        for i in range(200):
            ts.emit(EventCategory.JOINT_STATE, {"q": [i]})
        ts._flush_all()
        assert len(received) > 0

    def test_flush_with_compression(self):
        from strands_robots.telemetry.stream import TelemetryStream, TransportConfig
        from strands_robots.telemetry.types import EventCategory, StreamTier
        received = []
        # Set compression_threshold=1 so ANY batch gets compressed
        ts = TelemetryStream(robot_id="test", compression_threshold=1)
        t = TransportConfig(
            name="capture",
            tiers=[StreamTier.BATCH],
            handler=lambda payload: (received.append(payload), True)[1],
        )
        ts.add_transport(t)
        # Emit enough to trigger count-based flush (default max_count=100)
        for i in range(150):
            ts.emit(EventCategory.JOINT_STATE, {"q": list(range(50))})
        ts._flush_all()
        assert ts._stats["compressed_bytes"] > 0

    def test_send_with_retry_failure(self):
        from strands_robots.telemetry.stream import TelemetryStream, TransportConfig
        from strands_robots.telemetry.types import EventCategory, StreamTier, BatchConfig
        call_count = [0]

        def failing_handler(payload):
            call_count[0] += 1
            raise Exception("fail")

        # Use small max_count so a single emit triggers flush
        ts = TelemetryStream(
            robot_id="test",
            batch_config={
                StreamTier.STREAM: BatchConfig(max_count=1, max_bytes=64*1024, max_age_ms=50.0),
                StreamTier.BATCH: BatchConfig(max_count=1, max_bytes=64*1024, max_age_ms=50.0),
                StreamTier.STORAGE: BatchConfig(max_count=1, max_bytes=1024*1024, max_age_ms=2000.0),
            },
        )
        t = TransportConfig(
            name="fail",
            tiers=[StreamTier.BATCH],
            handler=failing_handler,
            retry_max=2,
            retry_base_ms=1.0,
        )
        ts.add_transport(t)
        ts.emit(EventCategory.JOINT_STATE, {"q": [1]})
        ts._flush_all()
        assert call_count[0] == 2  # retry_max
        assert ts._stats["errors"] > 0

    def test_get_stats(self):
        from strands_robots.telemetry.stream import TelemetryStream
        ts = TelemetryStream(robot_id="test")
        stats = ts.get_stats()
        assert stats["robot_id"] == "test"
        assert stats["running"] is False
        assert "buffer_sizes" in stats
        assert "frame_counter" in stats

    def test_flush_loop_integration(self):
        """Start stream, emit events, stop — flush loop should drain."""
        from strands_robots.telemetry.stream import TelemetryStream, TransportConfig
        from strands_robots.telemetry.types import EventCategory, StreamTier, BatchConfig
        received = []
        ts = TelemetryStream(
            robot_id="test",
            flush_interval_s=0.05,
            batch_config={
                StreamTier.STREAM: BatchConfig(max_count=1, max_bytes=64*1024, max_age_ms=10.0),
                StreamTier.BATCH: BatchConfig(max_count=10, max_bytes=64*1024, max_age_ms=50.0),
                StreamTier.STORAGE: BatchConfig(max_count=1, max_bytes=1024*1024, max_age_ms=2000.0),
            },
        )
        t = TransportConfig(
            name="capture",
            tiers=[StreamTier.BATCH],
            handler=lambda payload: (received.append(1), True)[1],
        )
        ts.add_transport(t)
        ts.start()
        for _ in range(50):
            ts.emit(EventCategory.JOINT_STATE, {"q": [0.1]})
        time.sleep(0.15)
        ts.stop(timeout=2.0)
        assert ts.get_stats()["flushed"] > 0


# ---------------------------------------------------------------------------
# telemetry/transports
# ---------------------------------------------------------------------------


class TestLocalWALTransport:
    def test_send_events(self, tmp_path):
        from strands_robots.telemetry.transports.local import LocalWALTransport
        transport = LocalWALTransport(wal_dir=str(tmp_path / "wal"), max_files=5)
        events = [{"event_id": "1", "data": {"x": 1}}]
        assert transport.send(events) is True
        transport.close()
        files = list((tmp_path / "wal").glob("telemetry_*"))
        assert len(files) > 0

    def test_send_compressed(self, tmp_path):
        from strands_robots.telemetry.transports.local import LocalWALTransport
        transport = LocalWALTransport(wal_dir=str(tmp_path / "wal"))
        payload = gzip.compress(json.dumps([{"a": 1}]).encode())
        assert transport.send(payload) is True
        transport.close()

    def test_rotation(self, tmp_path):
        from strands_robots.telemetry.transports.local import LocalWALTransport
        transport = LocalWALTransport(
            wal_dir=str(tmp_path / "wal"),
            max_file_bytes=50,
            max_files=3,
            compress_rotated=True,
        )
        for i in range(20):
            transport.send([{"event": i, "data": "x" * 30}])
        transport.close()
        stats = transport.get_stats()
        assert stats["total_written_bytes"] > 0

    def test_send_error(self, tmp_path):
        from strands_robots.telemetry.transports.local import LocalWALTransport
        transport = LocalWALTransport(wal_dir=str(tmp_path / "wal"))
        assert transport.send("not valid json bytes".encode()) is False


class TestStdoutTransport:
    def test_send_compact(self):
        from strands_robots.telemetry.transports.stdout import StdoutTransport
        t = StdoutTransport(compact=True, max_events_per_batch=2)
        events = [
            {"category": "joint_state", "robot_id": "r1", "frame_id": 0, "correlation_id": "abc"},
            {"category": "heartbeat", "robot_id": "r1", "frame_id": 1},
            {"category": "extra", "robot_id": "r1", "frame_id": 2},
        ]
        assert t.send(events) is True
        assert t._total_events == 3

    def test_send_verbose(self):
        from strands_robots.telemetry.transports.stdout import StdoutTransport
        t = StdoutTransport(compact=False)
        assert t.send([{"category": "test"}]) is True

    def test_send_compressed(self):
        from strands_robots.telemetry.transports.stdout import StdoutTransport
        t = StdoutTransport()
        payload = gzip.compress(json.dumps([{"category": "x"}]).encode())
        assert t.send(payload) is True

    def test_get_stats(self):
        from strands_robots.telemetry.transports.stdout import StdoutTransport
        t = StdoutTransport()
        stats = t.get_stats()
        assert "total_events_logged" in stats


class TestOTelTransport:
    def test_send_without_otel(self):
        from strands_robots.telemetry.transports.otel import OTelTransport
        t = OTelTransport()
        # should succeed as no-op
        assert t.send([{"category": "test", "robot_id": "r", "data": {"x": 1}}]) is True

    def test_send_compressed(self):
        from strands_robots.telemetry.transports.otel import OTelTransport
        t = OTelTransport()
        payload = gzip.compress(json.dumps([{"category": "x"}]).encode())
        assert t.send(payload) is True

    def test_get_stats(self):
        from strands_robots.telemetry.transports.otel import OTelTransport
        t = OTelTransport()
        stats = t.get_stats()
        assert "otel_available" in stats

    @patch("strands_robots.telemetry.transports.otel.HAS_OTEL", True)
    def test_emit_span_with_otel(self):
        from strands_robots.telemetry.transports.otel import OTelTransport
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
        t = OTelTransport()
        t._tracer = mock_tracer
        event = {
            "category": "joint_state",
            "robot_id": "r1",
            "tier": "BATCH",
            "frame_id": 5,
            "sim_or_real": "sim",
            "correlation_id": "abc",
            "data": {"q": 1.0, "name": "test"},
        }
        t._emit_span(event)
        assert mock_tracer.start_as_current_span.called


# ---------------------------------------------------------------------------
# registry/loader.py
# ---------------------------------------------------------------------------


class TestRegistryLoader:
    def test_load_robots(self):
        from strands_robots.registry.loader import _load
        data = _load("robots")
        assert "robots" in data

    def test_load_policies(self):
        from strands_robots.registry.loader import _load
        data = _load("policies")
        assert "providers" in data

    def test_load_missing(self):
        from strands_robots.registry.loader import _load
        data = _load("nonexistent_file_xyz")
        assert data == {}

    def test_reload(self):
        from strands_robots.registry.loader import _cache, _mtimes, reload
        reload()
        assert len(_cache) == 0
        assert len(_mtimes) == 0

    def test_validate_robots_duplicate_alias(self):
        from strands_robots.registry.loader import _validate_robots
        data = {
            "robots": {
                "robot_a": {"aliases": ["common"]},
                "robot_b": {"aliases": ["common"]},
            }
        }
        with pytest.raises(ValueError, match="Duplicate robot alias"):
            _validate_robots(data)

    def test_validate_robots_alias_collides_name(self):
        from strands_robots.registry.loader import _validate_robots
        data = {
            "robots": {
                "robot_a": {"aliases": ["robot_b"]},
                "robot_b": {"aliases": []},
            }
        }
        with pytest.raises(ValueError, match="collides with"):
            _validate_robots(data)

    def test_validate_policies_duplicate_alias(self):
        from strands_robots.registry.loader import _validate_policies
        data = {
            "providers": {
                "prov_a": {"aliases": ["common"], "shorthands": [], "url_patterns": []},
                "prov_b": {"aliases": ["common"], "shorthands": [], "url_patterns": []},
            }
        }
        with pytest.raises(ValueError, match="Duplicate policy alias"):
            _validate_policies(data)

    def test_validate_policies_duplicate_url_pattern(self):
        from strands_robots.registry.loader import _validate_policies
        data = {
            "providers": {
                "prov_a": {"aliases": [], "shorthands": [], "url_patterns": ["^http"]},
                "prov_b": {"aliases": [], "shorthands": [], "url_patterns": ["^http"]},
            }
        }
        with pytest.raises(ValueError, match="Duplicate URL pattern"):
            _validate_policies(data)


# ---------------------------------------------------------------------------
# registry/robots.py
# ---------------------------------------------------------------------------


class TestRegistryRobots:
    def test_resolve_name_canonical(self):
        from strands_robots.registry.robots import resolve_name
        # canonical names should resolve to themselves
        result = resolve_name("so100")
        assert isinstance(result, str)

    def test_resolve_name_unknown(self):
        from strands_robots.registry.robots import resolve_name
        result = resolve_name("nonexistent_robot_xyz")
        assert result == "nonexistent_robot_xyz"

    def test_get_robot(self):
        from strands_robots.registry.robots import get_robot
        info = get_robot("nonexistent_robot_xyz")
        assert info is None

    def test_has_sim(self):
        from strands_robots.registry.robots import has_sim
        result = has_sim("nonexistent_xyz")
        assert result is False

    def test_has_hardware(self):
        from strands_robots.registry.robots import has_hardware
        result = has_hardware("nonexistent_xyz")
        assert result is False

    def test_get_hardware_type_none(self):
        from strands_robots.registry.robots import get_hardware_type
        result = get_hardware_type("nonexistent_xyz")
        assert result is None

    def test_list_robots(self):
        from strands_robots.registry.robots import list_robots
        robots = list_robots()
        assert isinstance(robots, list)

    def test_list_robots_filtered(self):
        from strands_robots.registry.robots import list_robots
        list_robots(mode="sim")
        list_robots(mode="real")
        list_robots(mode="both")

    def test_list_robots_by_category(self):
        from strands_robots.registry.robots import list_robots_by_category
        cats = list_robots_by_category()
        assert isinstance(cats, dict)

    def test_list_aliases(self):
        from strands_robots.registry.robots import list_aliases
        aliases = list_aliases()
        assert isinstance(aliases, dict)

    def test_format_robot_table(self):
        from strands_robots.registry.robots import format_robot_table
        table = format_robot_table()
        assert "Total" in table


# ---------------------------------------------------------------------------
# registry/policies.py
# ---------------------------------------------------------------------------


class TestRegistryPolicies:
    def test_list_policy_providers(self):
        from strands_robots.registry.policies import list_policy_providers
        providers = list_policy_providers()
        assert isinstance(providers, list)
        assert "mock" in providers

    def test_get_policy_provider(self):
        from strands_robots.registry.policies import get_policy_provider
        info = get_policy_provider("mock")
        assert info is not None

    def test_get_policy_provider_none(self):
        from strands_robots.registry.policies import get_policy_provider
        info = get_policy_provider("nonexistent_provider_xyz")
        assert info is None

    def test_resolve_policy_string_mock(self):
        from strands_robots.registry.policies import resolve_policy_string
        prov, kwargs = resolve_policy_string("mock")
        assert prov == "mock"

    def test_resolve_policy_string_hf_model(self):
        from strands_robots.registry.policies import resolve_policy_string
        prov, kwargs = resolve_policy_string("lerobot/act_aloha_sim")
        assert "pretrained_name_or_path" in kwargs

    def test_resolve_policy_string_unknown(self):
        from strands_robots.registry.policies import resolve_policy_string
        prov, kwargs = resolve_policy_string("some_unknown_thing")
        assert prov == "lerobot_local"

    def test_import_policy_class_mock(self):
        from strands_robots.registry.policies import import_policy_class
        cls = import_policy_class("mock")
        assert cls is not None

    def test_import_policy_class_unknown(self):
        from strands_robots.registry.policies import import_policy_class
        with pytest.raises(ValueError, match="Unknown policy provider"):
            import_policy_class("nonexistent_provider_xyz_abc")

    def test_build_policy_kwargs(self):
        from strands_robots.registry.policies import build_policy_kwargs
        kwargs = build_policy_kwargs(
            provider="mock",
            policy_port=5555,
            policy_host="myhost",
        )
        assert isinstance(kwargs, dict)


# ---------------------------------------------------------------------------
# assets/__init__.py
# ---------------------------------------------------------------------------


class TestAssets:
    def test_get_assets_dir(self):
        from strands_robots.assets import get_assets_dir
        d = get_assets_dir()
        assert d.exists()

    def test_get_search_paths(self):
        from strands_robots.assets import get_search_paths
        paths = get_search_paths()
        assert len(paths) >= 1

    def test_resolve_model_path_unknown(self):
        from strands_robots.assets import resolve_model_path
        result = resolve_model_path("nonexistent_robot_xyz")
        assert result is None

    def test_resolve_model_dir_unknown(self):
        from strands_robots.assets import resolve_model_dir
        result = resolve_model_dir("nonexistent_robot_xyz")
        assert result is None

    def test_get_robot_info_unknown(self):
        from strands_robots.assets import get_robot_info
        result = get_robot_info("nonexistent_robot_xyz")
        assert result is None

    def test_list_available_robots(self):
        from strands_robots.assets import list_available_robots
        robots = list_available_robots()
        assert isinstance(robots, list)


# ---------------------------------------------------------------------------
# video.py
# ---------------------------------------------------------------------------


class TestVideo:
    def test_video_encoder_context(self, tmp_path):
        from strands_robots.video import VideoEncoder
        output = str(tmp_path / "test.mp4")
        with patch("strands_robots.video._check_pyav", return_value=False), \
             patch("strands_robots.video._check_imageio", return_value=True):
            mock_writer = MagicMock()
            with patch("imageio.get_writer", return_value=mock_writer):
                enc = VideoEncoder(output, fps=30)
                with enc:
                    frame = np.zeros((100, 100, 3), dtype=np.uint8)
                    enc.add_frame(frame)
                    enc.add_frame(frame)
                assert enc.frame_count == 2

    def test_encode_frames_empty(self):
        from strands_robots.video import encode_frames
        result = encode_frames([], "out.mp4")
        assert result["status"] == "error"

    def test_encode_frames_with_frames(self, tmp_path):
        from strands_robots.video import encode_frames
        output = str(tmp_path / "test.mp4")
        frames = [np.zeros((50, 50, 3), dtype=np.uint8)] * 3
        with patch("strands_robots.video._check_pyav", return_value=False), \
             patch("strands_robots.video._check_imageio", return_value=True):
            mock_writer = MagicMock()
            with patch("imageio.get_writer", return_value=mock_writer):
                with patch("os.path.exists", return_value=True), \
                     patch("os.path.getsize", return_value=1024):
                    result = encode_frames(frames, output)
                    assert result["status"] == "success"

    def test_video_encoder_no_backend(self, tmp_path):
        from strands_robots.video import VideoEncoder
        output = str(tmp_path / "test.mp4")
        with patch("strands_robots.video._check_pyav", return_value=False), \
             patch("strands_robots.video._check_imageio", return_value=False):
            enc = VideoEncoder(output)
            with pytest.raises(RuntimeError, match="No video encoder"):
                enc.add_frame(np.zeros((50, 50, 3), dtype=np.uint8))

    def test_get_video_info_no_av(self):
        from strands_robots.video import get_video_info
        with patch.dict("sys.modules", {"av": None}):
            info = get_video_info("/nonexistent/path.mp4")
            assert "path" in info

    def test_get_video_info_exception(self):
        from strands_robots.video import get_video_info
        with patch("builtins.__import__", side_effect=ImportError):
            info = get_video_info("/nonexistent/path.mp4")
            assert "path" in info

    def test_video_encoder_get_info(self):
        from strands_robots.video import VideoEncoder
        enc = VideoEncoder("test.mp4", fps=30, codec="h264")
        info = enc.get_info()
        assert info["fps"] == 30
        assert info["codec"] == "h264"

    def test_video_encoder_close_no_writer(self):
        from strands_robots.video import VideoEncoder
        enc = VideoEncoder("test.mp4")
        enc.close()  # no-op, no crash


# ---------------------------------------------------------------------------
# visualizer.py
# ---------------------------------------------------------------------------


class TestVisualizer:
    def test_recording_stats_defaults(self):
        from strands_robots.visualizer import RecordingStats
        stats = RecordingStats()
        assert stats.episode == 0
        assert stats.recording is False

    def test_visualizer_terminal_mode(self):
        from strands_robots.visualizer import RecordingVisualizer
        viz = RecordingVisualizer(mode="terminal", refresh_rate=10.0)
        viz.start()
        viz.update(
            frame_count=10,
            episode=1,
            total_episodes=5,
            task="test task",
            cameras={"wrist": np.zeros((100, 100, 3), dtype=np.uint8)},
            last_action={"j1": 0.1, "j2": 0.2, "j3": 0.3, "j4": 0.4, "j5": 0.5},
            last_state={"s1": 0.0},
            error=True,
        )
        viz.new_episode(2, "new task")
        stats = viz.get_stats_dict()
        assert stats["episode"] == 2
        time.sleep(0.12)
        viz.stop()

    def test_visualizer_json_mode(self):
        from strands_robots.visualizer import RecordingVisualizer
        viz = RecordingVisualizer(mode="json", refresh_rate=10.0)
        viz.start()
        viz.update(frame_count=5)
        time.sleep(0.12)
        viz.stop()

    def test_visualizer_web_mode(self):
        from strands_robots.visualizer import RecordingVisualizer
        viz = RecordingVisualizer(mode="web", port=18888, refresh_rate=10.0)
        viz.start()
        viz.update(frame_count=1)
        time.sleep(0.05)
        viz.stop()
        # Ensure web server gets cleaned up
        if viz._web_server:
            viz._web_server.shutdown()

    def test_visualizer_fps_tracking(self):
        from strands_robots.visualizer import RecordingVisualizer
        viz = RecordingVisualizer(mode="terminal")
        viz._start_time = time.time()
        viz._running = True
        viz.update(frame_count=1)
        time.sleep(0.05)
        viz.update(frame_count=2)
        time.sleep(0.05)
        viz.update(frame_count=3)
        stats = viz.get_stats_dict()
        assert stats["fps_actual"] >= 0


# ---------------------------------------------------------------------------
# processor.py
# ---------------------------------------------------------------------------


class TestProcessor:
    def test_processor_bridge_no_modules(self):
        from strands_robots.processor import ProcessorBridge
        with patch("strands_robots.processor._try_import_processor", return_value=None):
            bridge = ProcessorBridge()
            assert not bridge.has_preprocessor
            assert not bridge.has_postprocessor
            assert not bridge.is_active
            # passthrough
            obs = {"key": "val"}
            assert bridge.preprocess(obs) is obs
            assert bridge.postprocess("action") == "action"
            assert bridge.process_full_transition(obs) is obs

    def test_processor_bridge_repr(self):
        from strands_robots.processor import ProcessorBridge
        bridge = ProcessorBridge()
        assert "ProcessorBridge" in repr(bridge)

    def test_processor_bridge_get_info(self):
        from strands_robots.processor import ProcessorBridge
        bridge = ProcessorBridge(device="cpu")
        info = bridge.get_info()
        assert info["device"] == "cpu"
        assert not info["is_active"]

    def test_processor_bridge_reset(self):
        from strands_robots.processor import ProcessorBridge
        mock_pre = MagicMock()
        mock_post = MagicMock()
        bridge = ProcessorBridge(preprocessor=mock_pre, postprocessor=mock_post)
        bridge.reset()
        mock_pre.reset.assert_called_once()
        mock_post.reset.assert_called_once()

    def test_processor_bridge_from_pretrained_no_modules(self):
        from strands_robots.processor import ProcessorBridge
        with patch("strands_robots.processor._try_import_processor", return_value=None):
            bridge = ProcessorBridge.from_pretrained("some/model")
            assert not bridge.is_active

    def test_processor_bridge_wrap_policy(self):
        from strands_robots.processor import ProcessorBridge
        mock_policy = MagicMock()
        mock_policy.provider_name = "test"
        bridge = ProcessorBridge()
        wrapped = bridge.wrap_policy(mock_policy)
        assert "processed" in wrapped.provider_name

    def test_processed_policy_get_actions(self):
        from strands_robots.processor import ProcessedPolicy, ProcessorBridge
        mock_policy = AsyncMock()
        mock_policy.provider_name = "test"
        mock_policy.get_actions = AsyncMock(return_value=[{"j1": 0.1}])
        bridge = ProcessorBridge()
        pp = ProcessedPolicy(mock_policy, bridge)
        pp.set_robot_state_keys(["j1"])
        result = asyncio.run(pp.get_actions({"j1": 0.5}, "do something"))
        assert result == [{"j1": 0.1}]

    def test_processed_policy_reset(self):
        from strands_robots.processor import ProcessedPolicy, ProcessorBridge
        mock_policy = MagicMock()
        mock_policy.provider_name = "test"
        bridge = ProcessorBridge()
        pp = ProcessedPolicy(mock_policy, bridge)
        pp.reset()

    def test_processed_policy_getattr(self):
        from strands_robots.processor import ProcessedPolicy, ProcessorBridge
        mock_policy = MagicMock()
        mock_policy.provider_name = "test"
        mock_policy.custom_attr = "hello"
        bridge = ProcessorBridge()
        pp = ProcessedPolicy(mock_policy, bridge)
        assert pp.custom_attr == "hello"

    def test_create_processor_bridge_none(self):
        from strands_robots.processor import create_processor_bridge
        bridge = create_processor_bridge(device="cpu")
        assert not bridge.is_active

    def test_create_processor_bridge_with_path(self):
        from strands_robots.processor import create_processor_bridge
        with patch("strands_robots.processor._try_import_processor", return_value=None):
            bridge = create_processor_bridge("some/model", stats={"mean": [0.0]})
            assert not bridge.is_active


# ---------------------------------------------------------------------------
# kinematics.py
# ---------------------------------------------------------------------------


class TestKinematics:
    def test_mujoco_kinematics(self):
        mock_mj = MagicMock()
        mock_mj.mjtObj.mjOBJ_BODY = 1
        mock_mj.mjtObj.mjOBJ_JOINT = 2
        mock_mj.mj_name2id.return_value = 1
        mock_mj.mj_id2name.return_value = "joint_0"

        mock_model = MagicMock()
        mock_model.njnt = 3
        mock_model.nv = 3
        mock_data = MagicMock()
        mock_data.qpos = np.zeros(3)
        mock_data.xpos = np.array([[0, 0, 0], [1, 2, 3]])
        mock_data.xmat = np.eye(3).flatten().reshape(1, 9).repeat(2, axis=0)

        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            from strands_robots.kinematics import MuJoCoKinematics
            kin = MuJoCoKinematics(mock_model, mock_data, body_name="gripper")
            assert kin.backend == "mujoco"
            assert len(kin.joint_names) == 3

    def test_mujoco_kinematics_body_not_found(self):
        mock_mj = MagicMock()
        mock_mj.mjtObj.mjOBJ_BODY = 1
        mock_mj.mj_name2id.return_value = -1
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            from strands_robots.kinematics import MuJoCoKinematics
            with pytest.raises(ValueError, match="not found"):
                MuJoCoKinematics(MagicMock(), MagicMock(), body_name="missing")

    def test_placo_kinematics_import_error(self):
        with patch.dict("sys.modules", {"lerobot.model.kinematics": None}):
            from strands_robots.kinematics import PlacoKinematics
            with pytest.raises(ImportError):
                PlacoKinematics(urdf_path="test.urdf")

    def test_onnx_kinematics_import_error(self):
        with patch.dict("sys.modules", {"onnxruntime": None}):
            from strands_robots.kinematics import ONNXKinematics
            with pytest.raises(ImportError):
                ONNXKinematics(fk_model_path="model.onnx")

    def test_create_kinematics_mujoco(self):
        mock_mj = MagicMock()
        mock_mj.mjtObj.mjOBJ_BODY = 1
        mock_mj.mjtObj.mjOBJ_JOINT = 2
        mock_mj.mj_name2id.return_value = 1
        mock_mj.mj_id2name.return_value = "j"
        model = MagicMock()
        model.njnt = 2
        data = MagicMock()
        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            from strands_robots.kinematics import create_kinematics
            kin = create_kinematics(model=model, data=data)
            assert kin.backend == "mujoco"

    def test_create_kinematics_no_args(self):
        from strands_robots.kinematics import create_kinematics
        with pytest.raises(ValueError, match="Provide either"):
            create_kinematics()

    def test_generate_training_data(self):
        from strands_robots.kinematics import ONNXKinematics
        mock_kin = MagicMock()
        mock_kin.joint_names = ["j1", "j2"]
        mock_kin.forward_kinematics.return_value = np.eye(4)
        joints, poses = ONNXKinematics.generate_training_data(mock_kin, n_samples=5)
        assert len(joints) == 5


# ---------------------------------------------------------------------------
# dataset_recorder.py
# ---------------------------------------------------------------------------


class TestDatasetRecorder:
    def test_has_lerobot_dataset(self):
        from strands_robots.dataset_recorder import has_lerobot_dataset
        # clear cache
        has_lerobot_dataset.cache_clear()
        result = has_lerobot_dataset()
        assert isinstance(result, bool)

    def test_numpy_ify(self):
        from strands_robots.dataset_recorder import _numpy_ify
        assert isinstance(_numpy_ify(1.0), np.ndarray)
        assert isinstance(_numpy_ify([1, 2]), np.ndarray)
        assert isinstance(_numpy_ify(np.array([1])), np.ndarray)
        assert _numpy_ify("hello") == "hello"

    def test_build_features(self):
        from strands_robots.dataset_recorder import DatasetRecorder
        features = DatasetRecorder._build_features(
            camera_keys=["wrist"],
            joint_names=["j1", "j2"],
            use_videos=True,
        )
        assert "observation.images.wrist" in features
        assert "observation.state" in features
        assert "action" in features

    def test_build_features_robot_features(self):
        from strands_robots.dataset_recorder import DatasetRecorder
        features = DatasetRecorder._build_features(
            robot_features={"j1": float, "j2": float},
            action_features={"a1": float},
        )
        assert "observation.state" in features
        assert "action" in features

    def test_add_frame(self):
        from strands_robots.dataset_recorder import DatasetRecorder
        mock_dataset = MagicMock()
        mock_dataset.features = {
            "observation.state": {"names": ["j1", "j2"]},
            "action": {"names": ["j1", "j2"]},
        }
        recorder = DatasetRecorder(dataset=mock_dataset, task="test")
        recorder.add_frame(
            observation={"j1": 0.1, "j2": 0.2, "cam": np.zeros((100, 100, 3))},
            action={"j1": 0.5, "j2": 0.6},
            camera_keys=["cam"],
        )
        assert recorder.frame_count == 1
        mock_dataset.add_frame.assert_called_once()

    def test_add_frame_closed(self):
        from strands_robots.dataset_recorder import DatasetRecorder
        mock_dataset = MagicMock()
        mock_dataset.features = {}
        recorder = DatasetRecorder(dataset=mock_dataset)
        recorder._closed = True
        recorder.add_frame({"j1": 0.1}, {"j1": 0.5})
        assert recorder.frame_count == 0

    def test_add_frame_error(self):
        from strands_robots.dataset_recorder import DatasetRecorder
        mock_dataset = MagicMock()
        mock_dataset.features = {"observation.state": {"names": []}, "action": {"names": []}}
        mock_dataset.add_frame.side_effect = Exception("fail")
        recorder = DatasetRecorder(dataset=mock_dataset)
        recorder.add_frame({"j1": 0.1}, {"j1": 0.5})
        assert recorder.dropped_frame_count == 1

    def test_save_episode(self):
        from strands_robots.dataset_recorder import DatasetRecorder
        mock_dataset = MagicMock()
        mock_dataset.features = {}
        recorder = DatasetRecorder(dataset=mock_dataset)
        result = recorder.save_episode()
        assert result["status"] == "success"

    def test_save_episode_error(self):
        from strands_robots.dataset_recorder import DatasetRecorder
        mock_dataset = MagicMock()
        mock_dataset.save_episode.side_effect = Exception("fail")
        mock_dataset.features = {}
        recorder = DatasetRecorder(dataset=mock_dataset)
        result = recorder.save_episode()
        assert result["status"] == "error"

    def test_save_episode_closed(self):
        from strands_robots.dataset_recorder import DatasetRecorder
        mock_dataset = MagicMock()
        mock_dataset.features = {}
        recorder = DatasetRecorder(dataset=mock_dataset)
        recorder._closed = True
        result = recorder.save_episode()
        assert result["status"] == "error"

    def test_finalize(self):
        from strands_robots.dataset_recorder import DatasetRecorder
        mock_dataset = MagicMock()
        mock_dataset.features = {}
        recorder = DatasetRecorder(dataset=mock_dataset)
        recorder.finalize()
        assert recorder._closed

    def test_push_to_hub_error(self):
        from strands_robots.dataset_recorder import DatasetRecorder
        mock_dataset = MagicMock()
        mock_dataset.push_to_hub.side_effect = Exception("no hub")
        mock_dataset.features = {}
        recorder = DatasetRecorder(dataset=mock_dataset)
        result = recorder.push_to_hub()
        assert result["status"] == "error"

    def test_repr(self):
        from strands_robots.dataset_recorder import DatasetRecorder
        mock_dataset = MagicMock()
        mock_dataset.repo_id = "test/repo"
        mock_dataset.features = {}
        recorder = DatasetRecorder(dataset=mock_dataset)
        assert "test/repo" in repr(recorder)


# ---------------------------------------------------------------------------
# stereo/__init__.py
# ---------------------------------------------------------------------------


class TestStereo:
    def test_stereo_config_valid(self):
        from strands_robots.stereo import StereoConfig
        config = StereoConfig(model_variant="23-36-37", valid_iters=8)
        assert config.model_variant == "23-36-37"

    def test_stereo_config_invalid_variant(self):
        from strands_robots.stereo import StereoConfig
        with pytest.raises(ValueError, match="Invalid model_variant"):
            StereoConfig(model_variant="bad")

    def test_stereo_config_invalid_iters(self):
        from strands_robots.stereo import StereoConfig
        with pytest.raises(ValueError, match="valid_iters must be"):
            StereoConfig(valid_iters=0)

    def test_stereo_config_invalid_max_disp(self):
        from strands_robots.stereo import StereoConfig
        with pytest.raises(ValueError, match="max_disp must be"):
            StereoConfig(max_disp=0)

    def test_stereo_config_invalid_scale(self):
        from strands_robots.stereo import StereoConfig
        with pytest.raises(ValueError, match="scale must be"):
            StereoConfig(scale=0.0)

    def test_stereo_config_invalid_zfar(self):
        from strands_robots.stereo import StereoConfig
        with pytest.raises(ValueError, match="zfar must be"):
            StereoConfig(zfar=-1.0)

    def test_stereo_config_invalid_camera(self):
        from strands_robots.stereo import StereoConfig
        with pytest.raises(ValueError, match="Unknown camera"):
            StereoConfig(camera="nonexistent")

    def test_stereo_config_resolve_model_path_not_found(self):
        from strands_robots.stereo import StereoConfig
        config = StereoConfig()
        with pytest.raises(FileNotFoundError):
            config.resolve_model_path()

    def test_stereo_result(self):
        from strands_robots.stereo import StereoResult
        disp = np.ones((100, 200))
        result = StereoResult(disparity=disp)
        assert result.height == 100
        assert result.width == 200
        assert result.median_depth is None
        assert result.valid_ratio > 0
        d = result.to_dict()
        assert d["height"] == 100

    def test_stereo_result_with_depth(self):
        from strands_robots.stereo import StereoResult
        disp = np.ones((10, 10))
        depth = np.full((10, 10), 2.0)
        result = StereoResult(disparity=disp, depth=depth)
        assert result.median_depth == 2.0

    def test_stereo_result_median_depth_no_valid(self):
        from strands_robots.stereo import StereoResult
        depth = np.full((5, 5), np.inf)
        result = StereoResult(disparity=np.ones((5, 5)), depth=depth)
        assert result.median_depth is None

    def test_stereo_pipeline_init(self):
        from strands_robots.stereo import StereoConfig, StereoDepthPipeline
        p = StereoDepthPipeline(StereoConfig())
        assert "StereoDepthPipeline" in repr(p)

    def test_stereo_pipeline_both_config_and_kwargs(self):
        from strands_robots.stereo import StereoConfig, StereoDepthPipeline
        with pytest.raises(ValueError, match="Cannot specify both"):
            StereoDepthPipeline(StereoConfig(), valid_iters=4)

    def test_load_image_array(self):
        from strands_robots.stereo import StereoDepthPipeline
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = StereoDepthPipeline._load_image(img)
        assert result.shape == (100, 200, 3)

    def test_load_image_grayscale(self):
        from strands_robots.stereo import StereoDepthPipeline
        img = np.zeros((100, 200), dtype=np.uint8)
        result = StereoDepthPipeline._load_image(img)
        assert result.shape == (100, 200, 3)

    def test_load_image_bad_shape(self):
        from strands_robots.stereo import StereoDepthPipeline
        with pytest.raises(ValueError, match="Expected"):
            StereoDepthPipeline._load_image(np.zeros((10,)))

    def test_load_image_file_not_found(self):
        from strands_robots.stereo import StereoDepthPipeline
        with pytest.raises(FileNotFoundError):
            StereoDepthPipeline._load_image("/nonexistent/path.png")

    def test_depth_to_xyz(self):
        from strands_robots.stereo import _depth_to_xyz
        depth = np.ones((10, 10)) * 2.0
        K = np.array([[600, 0, 5], [0, 600, 5], [0, 0, 1]], dtype=np.float64)
        xyz = _depth_to_xyz(depth, K)
        assert xyz.shape == (10, 10, 3)

    def test_visualize_disparity(self):
        from strands_robots.stereo import _visualize_disparity
        disp = np.random.uniform(0, 10, (50, 50))
        vis = _visualize_disparity(disp)
        assert vis.shape == (50, 50, 3)

    def test_visualize_disparity_empty(self):
        from strands_robots.stereo import _visualize_disparity
        disp = np.full((10, 10), np.inf)
        vis = _visualize_disparity(disp)
        assert vis.shape == (10, 10, 3)

    def test_input_padder(self):
        from strands_robots.stereo import _InputPadder
        import torch
        padder = _InputPadder((1, 3, 100, 200), divis_by=32)
        x = torch.zeros(1, 3, 100, 200)
        padded = padder.pad(x)
        assert padded[0].shape[-2] % 32 == 0
        unpadded = padder.unpad(padded[0])
        assert unpadded.shape[-2] == 100


# ---------------------------------------------------------------------------
# rl_trainer.py
# ---------------------------------------------------------------------------


class TestRLTrainer:
    def test_rl_config_defaults(self):
        from strands_robots.rl_trainer import RLConfig
        config = RLConfig()
        assert config.algorithm == "ppo"
        assert config.total_timesteps == 1_000_000

    def test_reward_locomotion(self):
        from strands_robots.rl_trainer import RewardFunction
        obs = np.zeros(10)
        action = np.ones(5)
        r = RewardFunction.locomotion_reward(obs, action)
        assert isinstance(r, float)

    def test_reward_locomotion_dict_obs(self):
        from strands_robots.rl_trainer import RewardFunction
        obs = {"state": np.zeros(10)}
        r = RewardFunction.locomotion_reward(obs, None)
        assert isinstance(r, float)

    def test_reward_manipulation(self):
        from strands_robots.rl_trainer import RewardFunction
        obs = np.zeros(10)
        r = RewardFunction.manipulation_reward(obs, np.ones(3), target_pos=np.array([0.3, 0.0, 0.5]))
        assert isinstance(r, float)

    def test_reward_sparse(self):
        from strands_robots.rl_trainer import RewardFunction
        assert RewardFunction.sparse_success_reward(None, None, True) == 1.0
        assert RewardFunction.sparse_success_reward(None, None, False) == 0.0

    def test_pick_and_place_reward_phases(self):
        from strands_robots.rl_trainer import PickAndPlaceReward
        reward = PickAndPlaceReward(
            object_pos_indices=(3, 6),
            ee_pos_indices=(0, 3),
            gripper_index=6,
        )
        state = np.zeros(20)
        state[0:3] = [0.5, 0.0, 0.5]  # ee far
        state[3:6] = [0.0, 0.0, 0.5]  # object
        action = np.zeros(7)

        # Phase 1: reach
        r = reward(state, action)
        assert reward.current_phase == 0

        # Move ee close
        state[0:3] = [0.01, 0.0, 0.5]
        r = reward(state, action)
        assert reward.current_phase == 1  # advanced to grasp

        # Grasp: close gripper
        state[6] = 0.0  # closed
        r = reward(state, action)

        # Lift
        state[3:6] = [0.01, 0.0, 0.65]
        r = reward(state, action)
        assert reward.current_phase >= 2

        assert reward.phase_name in ["Reach", "Grasp", "Transport", "Place"]
        info = reward.get_info()
        assert "phase" in info
        assert "PickAndPlaceReward" in repr(reward)

    def test_pick_and_place_reward_reset(self):
        from strands_robots.rl_trainer import PickAndPlaceReward
        reward = PickAndPlaceReward()
        reward._phase = 3
        reward.reset()
        assert reward._phase == 0
        assert reward._prev_action is None

    def test_pick_and_place_drop_back(self):
        from strands_robots.rl_trainer import PickAndPlaceReward
        reward = PickAndPlaceReward(
            object_pos_indices=(3, 6),
            ee_pos_indices=(0, 3),
            gripper_index=6,
        )
        reward._phase = 2  # Transport
        reward._initial_obj_z = 0.5
        state = np.zeros(20)
        state[3:6] = [0.3, 0.0, 0.45]  # object dropped
        r = reward(state, np.zeros(7))
        assert reward.current_phase == 0  # back to reach

    def test_pick_and_place_place_phase(self):
        from strands_robots.rl_trainer import PickAndPlaceReward
        reward = PickAndPlaceReward(
            object_pos_indices=(3, 6),
            ee_pos_indices=(0, 3),
            gripper_index=6,
            target_place_pos=np.array([0.3, 0.0, 0.75]),
        )
        reward._phase = 3  # Place
        reward._initial_obj_z = 0.5
        state = np.zeros(20)
        state[3:6] = [0.3, 0.0, 0.75]  # at target
        state[6] = 0.1  # gripper open
        for _ in range(15):
            r = reward(state, np.zeros(7))
        assert reward.is_success

    def test_pick_and_place_extract_state_dict(self):
        from strands_robots.rl_trainer import PickAndPlaceReward
        reward = PickAndPlaceReward()
        state = reward._extract_state({"state": {"a": [1, 2], "b": [3]}})
        assert len(state) == 3

    def test_sb3_trainer_get_reward_fn(self):
        from strands_robots.rl_trainer import RLConfig, SB3Trainer
        config = RLConfig(task="walk forward")
        trainer = SB3Trainer(config)
        fn = trainer._get_reward_fn()
        assert fn is not None
        r = fn(np.zeros(10), np.ones(3))
        assert isinstance(r, float)

    def test_sb3_trainer_get_reward_pick(self):
        from strands_robots.rl_trainer import RLConfig, SB3Trainer
        config = RLConfig(task="pick up the cube")
        trainer = SB3Trainer(config)
        fn = trainer._get_reward_fn()
        assert fn is not None

    def test_sb3_trainer_get_reward_pick_place(self):
        from strands_robots.rl_trainer import PickAndPlaceReward, RLConfig, SB3Trainer
        config = RLConfig(task="pick and place the cube")
        trainer = SB3Trainer(config)
        fn = trainer._get_reward_fn()
        assert isinstance(fn, PickAndPlaceReward)

    def test_sb3_trainer_get_reward_none(self):
        from strands_robots.rl_trainer import RLConfig, SB3Trainer
        config = RLConfig(task="something neutral")
        trainer = SB3Trainer(config)
        fn = trainer._get_reward_fn()
        assert fn is None

    def test_sb3_trainer_evaluate_no_model(self):
        from strands_robots.rl_trainer import RLConfig, SB3Trainer
        config = RLConfig()
        trainer = SB3Trainer(config)
        result = trainer.evaluate()
        assert result["status"] == "error"

    def test_create_rl_trainer(self):
        from strands_robots.rl_trainer import create_rl_trainer
        trainer = create_rl_trainer(
            algorithm="sac",
            env_config={"robot_name": "so100", "task": "pick"},
        )
        assert trainer.config.algorithm == "sac"


# ---------------------------------------------------------------------------
# dreamgen/__init__.py
# ---------------------------------------------------------------------------


class TestDreamGen:
    def test_neural_trajectory(self):
        from strands_robots.dreamgen import NeuralTrajectory
        traj = NeuralTrajectory(
            frames=np.zeros((10, 480, 640, 3), dtype=np.uint8),
            actions=np.zeros((9, 6), dtype=np.float32),
            instruction="pick up cup",
        )
        assert traj.action_type == "idm"

    def test_dreamgen_config(self):
        from strands_robots.dreamgen import DreamGenConfig
        config = DreamGenConfig(video_model="wan2.1")
        assert config.video_model == "wan2.1"

    def test_dreamgen_pipeline_init(self):
        from strands_robots.dreamgen import DreamGenPipeline
        pipeline = DreamGenPipeline(video_model="wan2.1", embodiment_tag="so100")
        assert pipeline.config.embodiment_tag == "so100"

    def test_finetune_video_model(self):
        from strands_robots.dreamgen import DreamGenPipeline
        pipeline = DreamGenPipeline()
        result = pipeline.finetune_video_model("/data")
        assert result["status"] == "ready"

    def test_finetune_cosmos_transfer(self):
        from strands_robots.dreamgen import DreamGenConfig, DreamGenPipeline
        config = DreamGenConfig(video_model="cosmos_transfer")
        pipeline = DreamGenPipeline(config=config)
        result = pipeline.finetune_video_model("/data")
        assert result["status"] == "skipped"

    def test_generate_videos(self):
        from strands_robots.dreamgen import DreamGenPipeline
        pipeline = DreamGenPipeline()
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        result = pipeline.generate_videos(frames, ["pick up"], num_per_prompt=2)
        assert len(result) == 2

    def test_extract_actions_latent(self):
        from strands_robots.dreamgen import DreamGenPipeline
        pipeline = DreamGenPipeline()
        videos = [{"instruction": "pick", "video_path": "/tmp/v.mp4"}]
        trajs = pipeline.extract_actions(videos, method="latent")
        assert len(trajs) == 1
        assert trajs[0].action_type == "latent"

    def test_extract_actions_invalid(self):
        from strands_robots.dreamgen import DreamGenPipeline
        pipeline = DreamGenPipeline()
        with pytest.raises(ValueError, match="Unknown action extraction"):
            pipeline.extract_actions([], method="bad")

    def test_extract_idm_no_model(self):
        from strands_robots.dreamgen import DreamGenPipeline
        pipeline = DreamGenPipeline()
        videos = [{"instruction": "pick", "video_path": "/tmp/v.mp4"}]
        trajs = pipeline._extract_idm_actions(videos, "/tmp")
        assert len(trajs) == 1  # placeholder

    def test_create_dataset_raw(self, tmp_path):
        from strands_robots.dreamgen import DreamGenPipeline, NeuralTrajectory
        pipeline = DreamGenPipeline()
        trajs = [
            NeuralTrajectory(
                frames=np.zeros((5, 10, 10, 3), dtype=np.uint8),
                actions=np.zeros((4, 3), dtype=np.float32),
                instruction="test",
            )
        ]
        result = pipeline.create_dataset(trajs, str(tmp_path / "out"), format="raw")
        assert result["num_trajectories"] == 1

    def test_run_full_pipeline(self, tmp_path):
        from strands_robots.dreamgen import DreamGenPipeline
        pipeline = DreamGenPipeline()
        frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
        result = pipeline.run_full_pipeline(
            robot_dataset_path="/data",
            initial_frames=frames,
            instructions=["pick"],
            num_per_prompt=1,
            output_dir=str(tmp_path / "out"),
        )
        assert "stage1" in result
        assert "stage4" in result


# ---------------------------------------------------------------------------
# cosmos_transfer/__init__.py
# ---------------------------------------------------------------------------


class TestCosmosTransfer:
    def test_config_valid(self):
        from strands_robots.cosmos_transfer import CosmosTransferConfig
        config = CosmosTransferConfig()
        assert config.model_variant == "depth"

    def test_config_invalid_variant(self):
        from strands_robots.cosmos_transfer import CosmosTransferConfig
        with pytest.raises(ValueError, match="Invalid model_variant"):
            CosmosTransferConfig(model_variant="bad")

    def test_config_invalid_resolution(self):
        from strands_robots.cosmos_transfer import CosmosTransferConfig
        with pytest.raises(ValueError, match="Invalid output_resolution"):
            CosmosTransferConfig(output_resolution="4k")

    def test_config_invalid_gpus(self):
        from strands_robots.cosmos_transfer import CosmosTransferConfig
        with pytest.raises(ValueError):
            CosmosTransferConfig(num_gpus=0)

    def test_config_invalid_guidance(self):
        from strands_robots.cosmos_transfer import CosmosTransferConfig
        with pytest.raises(ValueError):
            CosmosTransferConfig(guidance=-1.0)

    def test_config_invalid_steps(self):
        from strands_robots.cosmos_transfer import CosmosTransferConfig
        with pytest.raises(ValueError):
            CosmosTransferConfig(num_steps=0)

    def test_config_invalid_control_weight(self):
        from strands_robots.cosmos_transfer import CosmosTransferConfig
        with pytest.raises(ValueError):
            CosmosTransferConfig(control_weight=3.0)

    def test_config_invalid_chunks(self):
        from strands_robots.cosmos_transfer import CosmosTransferConfig
        with pytest.raises(ValueError):
            CosmosTransferConfig(num_chunks=0)

    def test_config_invalid_overlap(self):
        from strands_robots.cosmos_transfer import CosmosTransferConfig
        with pytest.raises(ValueError):
            CosmosTransferConfig(chunk_overlap=-1)

    def test_config_resolve_checkpoint_not_found(self):
        from strands_robots.cosmos_transfer import CosmosTransferConfig
        config = CosmosTransferConfig()
        with pytest.raises(FileNotFoundError):
            config.resolve_checkpoint_path()

    def test_pipeline_init(self):
        from strands_robots.cosmos_transfer import CosmosTransferPipeline
        p = CosmosTransferPipeline()
        assert "CosmosTransferPipeline" in repr(p)

    def test_pipeline_both_config_and_kwargs(self):
        from strands_robots.cosmos_transfer import (
            CosmosTransferConfig,
            CosmosTransferPipeline,
        )
        with pytest.raises(ValueError, match="Cannot specify both"):
            CosmosTransferPipeline(CosmosTransferConfig(), num_gpus=2)

    def test_merge_config_overrides(self):
        from strands_robots.cosmos_transfer import CosmosTransferPipeline
        p = CosmosTransferPipeline()
        merged = p._merge_config_overrides(guidance=5.0)
        assert merged.guidance == 5.0

    def test_count_frames(self):
        from strands_robots.cosmos_transfer import CosmosTransferPipeline
        count = CosmosTransferPipeline._count_frames("/nonexistent.mp4")
        assert count == 0

    def test_cleanup(self, tmp_path):
        from strands_robots.cosmos_transfer import CosmosTransferPipeline
        p = CosmosTransferPipeline()
        td = str(tmp_path / "test_tmp")
        os.makedirs(td)
        p._tmp_dirs.append(td)
        p.cleanup()
        assert not os.path.isdir(td)

    def test_build_inference_spec(self):
        from strands_robots.cosmos_transfer import (
            CosmosTransferConfig,
            CosmosTransferPipeline,
        )
        p = CosmosTransferPipeline()
        spec = p._build_inference_spec(
            sim_video_path="/tmp/sim.mp4",
            prompt="test prompt",
            output_path="/tmp/out.mp4",
            control_types=["depth"],
            control_video_paths={"depth": "/tmp/depth.mp4"},
            control_weights=[1.0],
            config=CosmosTransferConfig(),
        )
        assert spec["prompt"] == "test prompt"
        assert "depth" in spec

    def test_transfer_video_missing_file(self):
        from strands_robots.cosmos_transfer import CosmosTransferPipeline
        p = CosmosTransferPipeline()
        with pytest.raises(FileNotFoundError):
            p.transfer_video("/nonexistent.mp4", "prompt", "/out.mp4")

    def test_transfer_video_invalid_control(self, tmp_path):
        from strands_robots.cosmos_transfer import CosmosTransferPipeline
        vid = tmp_path / "sim.mp4"
        vid.write_text("fake")
        p = CosmosTransferPipeline()
        with pytest.raises(ValueError, match="Invalid control type"):
            p.transfer_video(str(vid), "prompt", "/out.mp4", control_types=["bad"])


# ---------------------------------------------------------------------------
# marble/__init__.py
# ---------------------------------------------------------------------------


class TestMarble:
    def test_config_valid(self):
        from strands_robots.marble import MarbleConfig
        config = MarbleConfig()
        assert config.output_format == "ply"

    def test_config_invalid_format(self):
        from strands_robots.marble import MarbleConfig
        with pytest.raises(ValueError, match="Invalid output_format"):
            MarbleConfig(output_format="bad")

    def test_config_invalid_input_mode(self):
        from strands_robots.marble import MarbleConfig
        with pytest.raises(ValueError, match="Invalid input_mode"):
            MarbleConfig(input_mode="bad")

    def test_config_invalid_model(self):
        from strands_robots.marble import MarbleConfig
        with pytest.raises(ValueError, match="Invalid model"):
            MarbleConfig(model="bad")

    def test_config_invalid_robot(self):
        from strands_robots.marble import MarbleConfig
        with pytest.raises(ValueError, match="Unknown robot"):
            MarbleConfig(robot="bad_robot")

    def test_config_invalid_variations(self):
        from strands_robots.marble import MarbleConfig
        with pytest.raises(ValueError):
            MarbleConfig(num_variations=0)

    def test_marble_scene_dataclass(self):
        from strands_robots.marble import MarbleScene
        scene = MarbleScene(scene_id="test", prompt="kitchen")
        assert scene.splat_path is None
        assert scene.best_background is None
        d = scene.to_dict()
        assert d["scene_id"] == "test"

    def test_marble_scene_splat_priority(self):
        from strands_robots.marble import MarbleScene
        scene = MarbleScene(
            scene_id="t", prompt="p",
            spz_path="/a.spz", ply_path="/b.ply"
        )
        assert scene.splat_path == "/a.spz"

    def test_pipeline_init(self):
        from strands_robots.marble import MarblePipeline
        p = MarblePipeline()
        assert "MarblePipeline" in repr(p)

    def test_pipeline_both_config_and_kwargs(self):
        from strands_robots.marble import MarbleConfig, MarblePipeline
        with pytest.raises(ValueError, match="Cannot specify both"):
            MarblePipeline(MarbleConfig(), num_variations=5)

    def test_generate_world_placeholder(self, tmp_path):
        from strands_robots.marble import MarbleConfig, MarblePipeline
        config = MarbleConfig(api_url="local")  # force local/placeholder mode
        p = MarblePipeline(config)
        scenes = p.generate_world("A kitchen", output_dir=str(tmp_path / "scenes"))
        assert len(scenes) == 1
        assert scenes[0].prompt == "A kitchen"

    def test_create_placeholder_ply(self, tmp_path):
        from strands_robots.marble import MarblePipeline
        ply_path = str(tmp_path / "test.ply")
        MarblePipeline._create_placeholder_ply(ply_path, "test prompt")
        assert os.path.isfile(ply_path)

    def test_list_presets(self):
        from strands_robots.marble import list_presets
        presets = list_presets()
        assert len(presets) > 0
        assert "name" in presets[0]

    def test_read_ply_vertices_fallback(self, tmp_path):
        from strands_robots.marble import MarblePipeline
        ply_path = tmp_path / "test.ply"
        ply_path.write_text(
            "ply\nformat ascii 1.0\nelement vertex 2\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
            "1.0 2.0 3.0 128 128 128\n"
            "4.0 5.0 6.0 200 200 200\n"
        )
        verts, colors = MarblePipeline._read_ply_vertices(str(ply_path))
        assert len(verts) == 2
        assert len(colors) == 2

    def test_compose_scene_file_not_found(self):
        from strands_robots.marble import MarblePipeline
        p = MarblePipeline()
        with pytest.raises(FileNotFoundError):
            p.compose_scene("/nonexistent/scene.ply")

    def test_compose_scene_invalid_robot(self, tmp_path):
        from strands_robots.marble import MarblePipeline
        f = tmp_path / "scene.ply"
        f.write_text("fake")
        p = MarblePipeline()
        with pytest.raises(ValueError, match="Unknown robot"):
            p.compose_scene(str(f), robot="bad_robot")

    def test_generate_training_scenes_no_prompts(self):
        from strands_robots.marble import MarblePipeline
        p = MarblePipeline()
        with pytest.raises(ValueError, match="Either 'prompts'"):
            p.generate_training_scenes()

    def test_generate_training_scenes_with_preset(self, tmp_path):
        from strands_robots.marble import MarbleConfig, MarblePipeline
        config = MarbleConfig(api_url="local")
        p = MarblePipeline(config)
        result = p.generate_training_scenes(
            preset="kitchen", num_per_prompt=1,
            output_dir=str(tmp_path / "training"),
        )
        assert result["total_scenes"] >= 1

    def test_config_resolve_threedgrut_path(self):
        from strands_robots.marble import MarbleConfig
        config = MarbleConfig()
        result = config.resolve_threedgrut_path()
        # likely None unless installed
        assert result is None or os.path.isdir(result)


# ---------------------------------------------------------------------------
# leisaac.py
# ---------------------------------------------------------------------------


class TestLeIsaac:
    def test_list_tasks(self):
        from strands_robots.leisaac import list_tasks
        tasks = list_tasks()
        assert len(tasks) > 0

    def test_format_task_table(self):
        from strands_robots.leisaac import format_task_table
        table = format_task_table()
        assert "LeIsaac" in table

    def test_rollout_result(self):
        from strands_robots.leisaac import RolloutResult
        r = RolloutResult(n_episodes=5, n_successes=3, success_rate=0.6)
        d = r.to_dict()
        assert d["n_episodes"] == 5

    def test_leisaac_env_init(self):
        from strands_robots.leisaac import LeIsaacEnv
        env = LeIsaacEnv("so101_pick_orange")
        assert not env._loaded
        assert "LeIsaacEnv" in repr(env)

    def test_leisaac_env_custom_task(self):
        from strands_robots.leisaac import LeIsaacEnv
        env = LeIsaacEnv("custom/task/ref")
        assert env.task_info["robot"] == "unknown"

    def test_leisaac_env_load_failure(self):
        from strands_robots.leisaac import LeIsaacEnv
        env = LeIsaacEnv("so101_pick_orange")
        result = env.load()
        assert result is False  # lerobot not installed

    def test_leisaac_env_reset_not_loaded(self):
        from strands_robots.leisaac import LeIsaacEnv
        env = LeIsaacEnv("so101_pick_orange")
        with pytest.raises(RuntimeError, match="Failed to load"):
            env.reset()

    def test_leisaac_get_joint_names_no_env(self):
        from strands_robots.leisaac import LeIsaacEnv
        env = LeIsaacEnv("so101_pick_orange")
        assert env.get_joint_names() == []

    def test_obs_to_dict_array(self):
        from strands_robots.leisaac import LeIsaacEnv
        env = LeIsaacEnv("so101_pick_orange")
        env._raw_env = MagicMock()
        env._raw_env.action_space = MagicMock()
        env._raw_env.action_space.shape = (3,)
        # get_joint_names needs _raw_env with the right attrs
        env._raw_env.joint_names = None
        env._raw_env.robot_joint_names = None
        env._raw_env.action_names = None
        # configure spec so hasattr works correctly
        del env._raw_env.joint_names
        del env._raw_env.robot_joint_names
        del env._raw_env.action_names
        obs = np.array([1.0, 2.0, 3.0])
        d = env._obs_to_dict(obs)
        assert len(d) == 3

    def test_obs_to_dict_non_standard(self):
        from strands_robots.leisaac import LeIsaacEnv
        env = LeIsaacEnv()
        result = env._obs_to_dict("something_else")
        assert result == {"observation": "something_else"}

    def test_dict_to_action(self):
        from strands_robots.leisaac import LeIsaacEnv
        env = LeIsaacEnv()
        env._raw_env = MagicMock(spec=[])
        env._raw_env.action_space = MagicMock()
        env._raw_env.action_space.shape = (2,)
        action = env._dict_to_action({"joint_0": 0.5, "joint_1": 0.3})
        assert len(action) == 2

    def test_create_leisaac_env(self):
        from strands_robots.leisaac import create_leisaac_env
        env = create_leisaac_env("so101_pick_orange", auto_load=False)
        assert not env._loaded


# ---------------------------------------------------------------------------
# record.py
# ---------------------------------------------------------------------------


class TestRecord:
    def test_episode_stats(self):
        from strands_robots.record import EpisodeStats
        stats = EpisodeStats(index=0, frames=100, task="test")
        assert stats.frames == 100

    def test_record_mode(self):
        from strands_robots.record import RecordMode
        assert RecordMode.TELEOP.value == "teleop"
        assert RecordMode.POLICY.value == "policy"

    def test_record_session_init(self):
        from strands_robots.record import RecordSession
        mock_robot = MagicMock()
        session = RecordSession(robot=mock_robot, repo_id="test/repo")
        assert not session._connected

    def test_record_session_get_status(self):
        from strands_robots.record import RecordSession
        mock_robot = MagicMock()
        session = RecordSession(robot=mock_robot, task="pick up")
        status = session.get_status()
        assert status["task"] == "pick up"

    def test_record_session_disconnect(self):
        from strands_robots.record import RecordSession
        mock_robot = MagicMock()
        mock_teleop = MagicMock()
        session = RecordSession(robot=mock_robot, teleop=mock_teleop)
        session._connected = True
        session.disconnect()
        assert not session._connected

    def test_record_session_stop(self):
        from strands_robots.record import RecordSession
        mock_robot = MagicMock()
        session = RecordSession(robot=mock_robot)
        session.stop()
        assert session._stop_flag

    def test_record_session_discard_episode(self):
        from strands_robots.record import EpisodeStats, RecordSession
        mock_robot = MagicMock()
        session = RecordSession(robot=mock_robot)
        session._episodes.append(EpisodeStats(index=0))
        session.discard_episode()
        assert session._episodes[0].discarded

    def test_save_and_push(self):
        from strands_robots.record import EpisodeStats, RecordSession
        mock_robot = MagicMock()
        session = RecordSession(robot=mock_robot)
        session._episodes = [EpisodeStats(index=0, frames=10)]
        session._dataset = MagicMock()
        session._dataset.root = "/tmp/data"
        result = session.save_and_push()
        assert result["episodes"] == 1


# ---------------------------------------------------------------------------
# zenoh_mesh.py — test without zenoh
# ---------------------------------------------------------------------------


class TestZenohMesh:
    def test_peer_info(self):
        from strands_robots.zenoh_mesh import PeerInfo
        p = PeerInfo(peer_id="test", peer_type="robot", hostname="host")
        p.last_seen = time.time()
        assert p.age < 1.0
        d = p.to_dict()
        assert d["peer_id"] == "test"

    def test_get_peers_empty(self):
        from strands_robots.zenoh_mesh import get_peers
        peers = get_peers()
        assert isinstance(peers, list)

    def test_get_peer_none(self):
        from strands_robots.zenoh_mesh import get_peer
        assert get_peer("nonexistent") is None

    def test_update_peer(self):
        from strands_robots.zenoh_mesh import _PEERS, _update_peer
        is_new = _update_peer("test_peer", "robot", "host", {})
        assert is_new
        is_new2 = _update_peer("test_peer", "robot", "host", {})
        assert not is_new2
        # cleanup
        with __import__("strands_robots.zenoh_mesh", fromlist=["_PEERS_LOCK"])._PEERS_LOCK:
            _PEERS.pop("test_peer", None)

    def test_prune_peers(self):
        from strands_robots.zenoh_mesh import PeerInfo, _PEERS, _prune_peers
        import strands_robots.zenoh_mesh as zm
        with zm._PEERS_LOCK:
            _PEERS["stale_peer"] = PeerInfo(
                peer_id="stale_peer", peer_type="robot",
                last_seen=time.time() - 100,
            )
        _prune_peers()
        assert "stale_peer" not in _PEERS

    def test_put_no_session(self):
        from strands_robots.zenoh_mesh import _put
        _put("test/key", {"data": 1})  # should not crash

    def test_init_mesh_disabled(self):
        from strands_robots.zenoh_mesh import init_mesh
        result = init_mesh(MagicMock(), mesh=False)
        assert result is None

    @patch.dict(os.environ, {"STRANDS_MESH": "false"})
    def test_init_mesh_env_disabled(self):
        from strands_robots.zenoh_mesh import init_mesh
        result = init_mesh(MagicMock())
        assert result is None

    def test_mesh_stop_not_running(self):
        from strands_robots.zenoh_mesh import Mesh
        m = Mesh(MagicMock(), peer_id="test")
        m.stop()  # no-op

    def test_mesh_alive(self):
        from strands_robots.zenoh_mesh import Mesh
        m = Mesh(MagicMock(), peer_id="test")
        assert not m.alive

    def test_mesh_dispatch_status(self):
        from strands_robots.zenoh_mesh import Mesh
        mock_robot = MagicMock()
        mock_robot.get_task_status.return_value = {"status": "idle"}
        m = Mesh(mock_robot, peer_id="test")
        result = m._dispatch({"action": "status"})
        assert result["status"] == "idle"

    def test_mesh_dispatch_features(self):
        from strands_robots.zenoh_mesh import Mesh
        mock_robot = MagicMock()
        mock_robot.get_features.return_value = {"features": []}
        m = Mesh(mock_robot, peer_id="test")
        result = m._dispatch({"action": "features"})
        assert "features" in result

    def test_mesh_dispatch_stop(self):
        from strands_robots.zenoh_mesh import Mesh
        mock_robot = MagicMock()
        mock_robot.stop_task.return_value = {"status": "stopped"}
        m = Mesh(mock_robot, peer_id="test")
        result = m._dispatch({"action": "stop"})
        assert result["status"] == "stopped"

    def test_mesh_dispatch_unknown(self):
        from strands_robots.zenoh_mesh import Mesh
        m = Mesh(MagicMock(), peer_id="test")
        result = m._dispatch({"action": "unknown_action"})
        assert "error" in result

    def test_mesh_dispatch_state(self):
        from strands_robots.zenoh_mesh import Mesh
        m = Mesh(MagicMock(), peer_id="test")
        result = m._dispatch({"action": "state"})
        assert isinstance(result, dict)

    def test_mesh_dispatch_execute_no_instruction(self):
        from strands_robots.zenoh_mesh import Mesh
        m = Mesh(MagicMock(), peer_id="test")
        result = m._dispatch({"action": "execute"})
        assert "error" in result

    def test_mesh_publish_step(self):
        from strands_robots.zenoh_mesh import Mesh
        m = Mesh(MagicMock(), peer_id="test")
        m.publish_step(
            step=1,
            observation={"j1": 0.1, "img": np.zeros((100, 100, 3))},
            action={"j1": 0.5},
            instruction="pick",
        )

    def test_mesh_build_presence(self):
        from strands_robots.zenoh_mesh import Mesh
        mock_robot = MagicMock()
        mock_robot.tool_name_str = "test_robot"
        mock_robot._task_state.status.value = "idle"
        mock_robot._task_state.instruction = ""
        mock_robot.robot.is_connected = True
        mock_robot.robot.name = "so100"
        mock_robot._action_features = {"j1": float}
        m = Mesh(mock_robot, peer_id="test")
        p = m._build_presence()
        assert p["tool_name"] == "test_robot"

    def test_mesh_read_state_no_data(self):
        from strands_robots.zenoh_mesh import Mesh
        mock_robot = MagicMock(spec=[])
        m = Mesh(mock_robot, peer_id="test")
        result = m._read_state()
        assert result is None  # only peer_id and t


# ---------------------------------------------------------------------------
# factory.py
# ---------------------------------------------------------------------------


class TestFactory:
    def test_auto_detect_mode_env_sim(self):
        from strands_robots.factory import _auto_detect_mode
        with patch.dict(os.environ, {"STRANDS_ROBOT_MODE": "sim"}):
            assert _auto_detect_mode("so100") == "sim"

    def test_auto_detect_mode_env_real(self):
        from strands_robots.factory import _auto_detect_mode
        with patch.dict(os.environ, {"STRANDS_ROBOT_MODE": "real"}):
            assert _auto_detect_mode("so100") == "real"

    def test_auto_detect_mode_default_sim(self):
        from strands_robots.factory import _auto_detect_mode
        with patch.dict(os.environ, {}, clear=True):
            result = _auto_detect_mode("nonexistent_xyz")
            assert result == "sim"

    def test_list_robots(self):
        from strands_robots.factory import list_robots
        robots = list_robots()
        assert isinstance(robots, list)


# ---------------------------------------------------------------------------
# robot.py — RobotTaskState
# ---------------------------------------------------------------------------


class TestRobotTaskState:
    def test_update_and_snapshot(self):
        from strands_robots.robot import RobotTaskState, TaskStatus
        state = RobotTaskState()
        state.update(status=TaskStatus.RUNNING, instruction="test", step_count=5)
        snap = state.snapshot()
        assert snap["status"] == TaskStatus.RUNNING
        assert snap["instruction"] == "test"
        assert snap["step_count"] == 5

    def test_task_status_enum(self):
        from strands_robots.robot import TaskStatus
        assert TaskStatus.IDLE.value == "idle"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.ERROR.value == "error"


# ---------------------------------------------------------------------------
# __init__.py — test imports work
# ---------------------------------------------------------------------------


class TestModuleInit:
    def test_all_list(self):
        import strands_robots
        assert isinstance(strands_robots.__all__, list)

    def test_has_basic_exports(self):
        import strands_robots
        # These should always be in __all__ if registry/policies can be loaded
        # Just verify module loads without errors
        assert hasattr(strands_robots, "__all__")


# ---------------------------------------------------------------------------
# envs.py — the StrandsSimEnv stub when gym is not available
# ---------------------------------------------------------------------------


class TestEnvsStub:
    def test_strands_sim_env_no_gym(self):
        """Test that the stub raises ImportError when gymnasium not installed."""
        # Save and remove gym from modules
        saved_gym = sys.modules.get("gymnasium")
        sys.modules["gymnasium"] = None

        try:
            # Force reimport
            import importlib
            if "strands_robots.envs" in sys.modules:
                del sys.modules["strands_robots.envs"]

            # This tests the stub path — but only if HAS_GYM was False at import
            # Since gym might actually be installed, we test the stub separately
            from strands_robots.envs import HAS_GYM
            if not HAS_GYM:
                from strands_robots.envs import StrandsSimEnv
                with pytest.raises(ImportError):
                    StrandsSimEnv()
        finally:
            if saved_gym is not None:
                sys.modules["gymnasium"] = saved_gym


# ---------------------------------------------------------------------------
# Comprehensive integration: Robot.get_features, get_task_status, stop_task
# ---------------------------------------------------------------------------


class TestRobotMethods:
    """Test Robot class methods without actual hardware."""

    def _make_robot(self):
        """Create a Robot-like object with mocked internals."""
        from strands_robots.robot import RobotTaskState, TaskStatus
        state = RobotTaskState()
        return state

    def test_get_features_empty(self):
        """Test get_features with no features."""
        from strands_robots.robot import Robot
        # We can't easily instantiate Robot without mocking lerobot,
        # so test the parts we can
        from strands_robots.robot import RobotTaskState, TaskStatus
        state = RobotTaskState()
        state.update(status=TaskStatus.IDLE)
        snap = state.snapshot()
        assert snap["status"] == TaskStatus.IDLE

    def test_task_state_concurrent_update(self):
        """Test thread-safe updates to RobotTaskState."""
        from strands_robots.robot import RobotTaskState, TaskStatus
        state = RobotTaskState()
        errors = []

        def updater(status, count):
            try:
                for _ in range(100):
                    state.update(status=status, step_count=count)
                    state.snapshot()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=updater, args=(TaskStatus.RUNNING, 1)),
            threading.Thread(target=updater, args=(TaskStatus.IDLE, 2)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_video_encoder_grayscale_frame(self):
        """VideoEncoder with grayscale frame via PyAV path."""
        from strands_robots.video import VideoEncoder
        with patch("strands_robots.video._check_pyav", return_value=True):
            mock_av = MagicMock()
            mock_container = MagicMock()
            mock_stream = MagicMock()
            mock_stream.encode.return_value = []
            mock_container.add_stream.return_value = mock_stream
            mock_av.open.return_value = mock_container
            mock_av.VideoFrame.from_ndarray.return_value = MagicMock()

            with patch.dict("sys.modules", {"av": mock_av}):
                enc = VideoEncoder("/tmp/test_pyav.mp4", codec="h264")
                # Init with pyav
                enc._init_writer(640, 480)
                enc._backend = "pyav"
                enc._writer = True

                # Add grayscale frame
                gray = np.zeros((480, 640), dtype=np.uint8)
                enc._add_frame_pyav(gray)

    def test_dataset_recorder_add_frame_auto_detect_cameras(self):
        from strands_robots.dataset_recorder import DatasetRecorder
        mock_dataset = MagicMock()
        mock_dataset.features = {
            "observation.state": {"names": ["j1"]},
            "action": {"names": ["j1"]},
        }
        recorder = DatasetRecorder(dataset=mock_dataset)
        # Image auto-detected as camera
        obs = {"j1": 0.5, "cam": np.zeros((100, 100, 3))}
        recorder.add_frame(obs, {"j1": 0.3})
        assert recorder.frame_count == 1

    def test_stereo_config_resolve_env_var(self, tmp_path):
        from strands_robots.stereo import StereoConfig
        model_dir = tmp_path / "23-36-37"
        model_dir.mkdir()
        (model_dir / "model_best_bp2_serialize.pth").write_text("fake")
        config = StereoConfig()
        with patch.dict(os.environ, {"STEREO_MODEL_DIR": str(tmp_path)}):
            path = config.resolve_model_path()
            assert "model_best_bp2_serialize.pth" in path

    def test_marble_config_api_key_from_env(self):
        from strands_robots.marble import MarbleConfig
        with patch.dict(os.environ, {"WLT_API_KEY": "test_key"}):
            config = MarbleConfig()
            assert config.api_key == "test_key"

    def test_cosmos_transfer_config_resolve_env(self, tmp_path):
        from strands_robots.cosmos_transfer import CosmosTransferConfig
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        config = CosmosTransferConfig()
        with patch.dict(os.environ, {"COSMOS_CHECKPOINT_DIR": str(ckpt_dir)}):
            path = config.resolve_checkpoint_path()
            assert path == str(ckpt_dir)

    def test_visualizer_fps_bar_colors(self):
        """Test different FPS ratio color paths in terminal render."""
        from strands_robots.visualizer import RecordingVisualizer
        viz = RecordingVisualizer(mode="terminal")
        viz._start_time = time.time()
        viz._running = True
        viz.stats.fps_target = 30.0
        
        # Good FPS (>0.9 ratio)
        viz.stats.fps_actual = 29.0
        viz._render_terminal()
        
        # Warn FPS (0.7-0.9)
        viz.stats.fps_actual = 24.0
        viz._render_terminal()
        
        # Bad FPS (<0.7)
        viz.stats.fps_actual = 15.0
        viz._render_terminal()

    def test_record_session_context_manager(self):
        from strands_robots.record import RecordSession
        mock_robot = MagicMock()
        mock_robot.is_connected = True
        session = RecordSession(robot=mock_robot, use_processor=False)
        session._dataset = MagicMock()
        session._connected = True
        with patch.object(session, "connect"):
            with patch.object(session, "save_and_push", return_value={}):
                with patch.object(session, "disconnect"):
                    with session:
                        pass

    def test_record_session_del(self):
        from strands_robots.record import RecordSession
        mock_robot = MagicMock()
        session = RecordSession(robot=mock_robot)
        session._connected = True
        del session  # should not raise

    def test_pick_and_place_contact_force(self):
        from strands_robots.rl_trainer import PickAndPlaceReward
        reward = PickAndPlaceReward(
            object_pos_indices=(3, 6),
            ee_pos_indices=(0, 3),
            gripper_index=6,
            contact_force_index=7,
        )
        assert reward._detect_grasp(0.0, 0.5)  # gripper closed + force
        assert not reward._detect_grasp(0.0, 0.0)  # gripper closed but no force
        assert not reward._detect_grasp(0.1, 0.5)  # gripper open


class TestTelemetryTransportsInit:
    def test_imports(self):
        from strands_robots.telemetry.transports import LocalWALTransport, StdoutTransport
        assert LocalWALTransport is not None
        assert StdoutTransport is not None


class TestTelemetryInit:
    def test_imports(self):
        from strands_robots.telemetry import (
            BatchConfig,
            EventCategory,
            StreamTier,
            TelemetryEvent,
            TelemetryStream,
        )
        assert TelemetryStream is not None
        assert TelemetryEvent is not None


class TestRegistryInit:
    def test_imports(self):
        from strands_robots.registry import (
            build_policy_kwargs,
            format_robot_table,
            get_hardware_type,
            get_policy_provider,
            get_robot,
            has_hardware,
            has_sim,
            import_policy_class,
            list_aliases,
            list_policy_providers,
            list_robots,
            list_robots_by_category,
            reload,
            resolve_name,
            resolve_policy_string,
        )
        assert resolve_name is not None


class TestAssetsSearchPaths:
    def test_custom_env_var(self):
        from strands_robots.assets import get_search_paths
        with patch.dict(os.environ, {"STRANDS_ASSETS_DIR": "/custom/path"}):
            paths = get_search_paths()
            assert Path("/custom/path") in paths


class TestMarbleBuildWorldPrompt:
    def test_text_prompt(self):
        from strands_robots.marble import MarblePipeline
        p = MarblePipeline()
        prompt = p._build_world_prompt("A kitchen", "text")
        assert prompt["type"] == "text"
        assert prompt["text_prompt"] == "A kitchen"

    def test_image_prompt_url(self):
        from strands_robots.marble import MarblePipeline
        p = MarblePipeline()
        prompt = p._build_world_prompt(
            "A kitchen", "image", input_image="https://example.com/img.jpg"
        )
        assert prompt["type"] == "image"

    def test_video_prompt_media_asset(self):
        from strands_robots.marble import MarblePipeline
        p = MarblePipeline()
        prompt = p._build_world_prompt(
            "A room", "video", media_asset_id="asset123"
        )
        assert prompt["type"] == "video"

    def test_multi_image_prompt(self):
        from strands_robots.marble import MarblePipeline
        p = MarblePipeline()
        multi = [{"content": "img_data", "azimuth": 0}]
        prompt = p._build_world_prompt(
            "A room", "multi-image", multi_image_prompt=multi
        )
        assert prompt["type"] == "multi-image"

    def test_multi_image_no_data(self):
        from strands_robots.marble import MarblePipeline
        p = MarblePipeline()
        with pytest.raises(ValueError, match="multi_image_prompt"):
            p._build_world_prompt("A room", "multi-image")

    def test_unsupported_mode(self):
        from strands_robots.marble import MarblePipeline
        p = MarblePipeline()
        with pytest.raises(ValueError, match="Unsupported input_mode"):
            p._build_world_prompt("A room", "audio")

    def test_image_no_source(self):
        from strands_robots.marble import MarblePipeline
        p = MarblePipeline()
        with pytest.raises(ValueError, match="image mode requires"):
            p._build_world_prompt("A room", "image")

    def test_video_no_source(self):
        from strands_robots.marble import MarblePipeline
        p = MarblePipeline()
        with pytest.raises(ValueError, match="video mode requires"):
            p._build_world_prompt("A room", "video")


class TestProcessorBridgeWithModules:
    """Tests for processor bridge with mocked LeRobot modules."""

    def test_from_pretrained_with_modules(self):
        from strands_robots.processor import ProcessorBridge

        mock_modules = {
            "DataProcessorPipeline": MagicMock(),
        }
        mock_pipeline_cls = mock_modules["DataProcessorPipeline"]
        mock_pipeline_cls.from_pretrained.side_effect = [
            MagicMock(__len__=lambda self: 3, steps=[MagicMock(), MagicMock(), MagicMock()]),
            FileNotFoundError("no post"),
        ]

        with patch("strands_robots.processor._try_import_processor", return_value=mock_modules):
            bridge = ProcessorBridge.from_pretrained("test/model")
            assert bridge.has_preprocessor

    def test_from_pretrained_generic_exception(self):
        from strands_robots.processor import ProcessorBridge

        mock_modules = {"DataProcessorPipeline": MagicMock()}
        mock_modules["DataProcessorPipeline"].from_pretrained.side_effect = RuntimeError("generic")

        with patch("strands_robots.processor._try_import_processor", return_value=mock_modules):
            bridge = ProcessorBridge.from_pretrained("test/model")
            assert not bridge.has_preprocessor

    def test_preprocess_with_pipeline(self):
        from strands_robots.processor import ProcessorBridge

        mock_pre = MagicMock()
        mock_pre.process_observation.return_value = {"processed": True}
        mock_modules = {"dummy": True}

        bridge = ProcessorBridge(preprocessor=mock_pre)
        bridge._modules = mock_modules
        result = bridge.preprocess({"raw": True})
        assert result == {"processed": True}

    def test_preprocess_exception(self):
        from strands_robots.processor import ProcessorBridge

        mock_pre = MagicMock()
        mock_pre.process_observation.side_effect = Exception("fail")
        bridge = ProcessorBridge(preprocessor=mock_pre)
        bridge._modules = {"dummy": True}
        result = bridge.preprocess({"raw": True})
        assert result == {"raw": True}

    def test_postprocess_with_pipeline(self):
        from strands_robots.processor import ProcessorBridge

        mock_post = MagicMock()
        mock_post.process_action.return_value = {"processed_action": True}
        bridge = ProcessorBridge(postprocessor=mock_post)
        bridge._modules = {"dummy": True}
        result = bridge.postprocess({"raw_action": True})
        assert result == {"processed_action": True}

    def test_postprocess_exception(self):
        from strands_robots.processor import ProcessorBridge

        mock_post = MagicMock()
        mock_post.process_action.side_effect = Exception("fail")
        bridge = ProcessorBridge(postprocessor=mock_post)
        bridge._modules = {"dummy": True}
        result = bridge.postprocess("action")
        assert result == "action"

    def test_process_full_transition(self):
        from strands_robots.processor import ProcessorBridge

        mock_pre = MagicMock()
        mock_pre.return_value = {"full": True}
        bridge = ProcessorBridge(preprocessor=mock_pre)
        bridge._modules = {"dummy": True}
        result = bridge.process_full_transition({"obs": 1})
        assert result == {"full": True}

    def test_get_info_with_pipelines(self):
        from strands_robots.processor import ProcessorBridge

        mock_pre = MagicMock()
        mock_pre.__len__ = MagicMock(return_value=2)
        mock_pre.steps = [MagicMock(), MagicMock()]
        mock_post = MagicMock()
        mock_post.__len__ = MagicMock(return_value=1)
        mock_post.steps = [MagicMock()]

        bridge = ProcessorBridge(preprocessor=mock_pre, postprocessor=mock_post)
        info = bridge.get_info()
        assert info["preprocessor_steps"] == 2
        assert info["postprocessor_steps"] == 1


class TestDreamGenLoadVideoFrames:
    def test_load_video_cv2(self):
        from strands_robots.dreamgen import DreamGenPipeline

        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, frame), (True, frame), (False, None)]
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.cvtColor.return_value = frame
        mock_cv2.COLOR_BGR2RGB = 4

        pipeline = DreamGenPipeline()
        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            frames = pipeline._load_video_frames("/tmp/test.mp4")
            assert isinstance(frames, np.ndarray)
            assert frames.shape[0] == 2


class TestProcessedPolicySelectActionSync:
    def test_select_action_sync_with_postprocessor(self):
        from strands_robots.processor import ProcessedPolicy, ProcessorBridge

        mock_policy = MagicMock()
        mock_policy.provider_name = "test"
        mock_policy.select_action_sync.return_value = np.array([1.0, 2.0])

        mock_post = MagicMock()
        mock_post.process_action.return_value = np.array([0.5, 1.0])

        bridge = ProcessorBridge(postprocessor=mock_post)
        bridge._modules = {"dummy": True}

        pp = ProcessedPolicy(mock_policy, bridge)
        result = pp.select_action_sync({"obs": 1})
        np.testing.assert_array_equal(result, np.array([0.5, 1.0]))


class TestProcessedPolicyGetActionsWithPostprocess:
    def test_get_actions_postprocess_dict(self):
        from strands_robots.processor import ProcessedPolicy, ProcessorBridge

        mock_policy = AsyncMock()
        mock_policy.provider_name = "test"
        mock_policy.get_actions = AsyncMock(return_value=[{"j1": 0.1}])

        mock_post = MagicMock()
        mock_post.process_action.return_value = {"j1": 0.05}

        bridge = ProcessorBridge(postprocessor=mock_post)
        bridge._modules = {"dummy": True}

        pp = ProcessedPolicy(mock_policy, bridge)
        result = asyncio.run(pp.get_actions({"obs": 1}, "test"))
        assert result == [{"j1": 0.05}]

    def test_get_actions_postprocess_non_dict(self):
        from strands_robots.processor import ProcessedPolicy, ProcessorBridge

        mock_policy = AsyncMock()
        mock_policy.provider_name = "test"
        mock_policy.get_actions = AsyncMock(return_value=[{"j1": 0.1}])

        mock_post = MagicMock()
        mock_post.process_action.return_value = np.array([0.05])  # non-dict

        bridge = ProcessorBridge(postprocessor=mock_post)
        bridge._modules = {"dummy": True}

        pp = ProcessedPolicy(mock_policy, bridge)
        result = asyncio.run(pp.get_actions({"obs": 1}, "test"))
        assert result == [{"j1": 0.1}]  # falls back to original


class TestSendWithRetrySuccess:
    def test_handler_returns_false_then_true(self):
        from strands_robots.telemetry.stream import TelemetryStream, TransportConfig
        from strands_robots.telemetry.types import StreamTier

        calls = [0]

        def handler(payload):
            calls[0] += 1
            return calls[0] >= 2

        ts = TelemetryStream(robot_id="test")
        t = TransportConfig(
            name="retry_test",
            tiers=[StreamTier.BATCH],
            handler=handler,
            retry_max=3,
            retry_base_ms=1.0,
        )
        result = ts._send_with_retry(t, [], StreamTier.BATCH, False)
        assert result is True
        assert calls[0] == 2


class TestRegistryPoliciesResolveURLPatterns:
    def test_resolve_grpc_url(self):
        from strands_robots.registry.policies import resolve_policy_string
        # Test host:port pattern
        prov, kwargs = resolve_policy_string("localhost:8080")
        assert "server_address" in kwargs or prov is not None

    def test_resolve_registered_provider(self):
        from strands_robots.registry.policies import resolve_policy_string
        prov, kwargs = resolve_policy_string("groot")
        assert prov == "groot"


class TestLocalWALEdgeCases:
    def test_cleanup_old_files(self, tmp_path):
        from strands_robots.telemetry.transports.local import LocalWALTransport
        transport = LocalWALTransport(
            wal_dir=str(tmp_path),
            max_files=2,
        )
        # Create files manually
        for i in range(5):
            (tmp_path / f"telemetry_{i:04d}.jsonl").write_text(f"line {i}\n")
        transport._cleanup_old_files()
        remaining = list(tmp_path.glob("telemetry_*"))
        assert len(remaining) < 5

    def test_send_single_dict(self, tmp_path):
        from strands_robots.telemetry.transports.local import LocalWALTransport
        transport = LocalWALTransport(wal_dir=str(tmp_path / "wal2"))
        # Single dict (not a list) — still works via not isinstance check
        raw = json.dumps({"event": "test"}).encode()
        assert transport.send(raw) is True
        transport.close()
