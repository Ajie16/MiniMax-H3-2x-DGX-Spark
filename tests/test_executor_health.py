"""Generate-time health must not mark a busy Ray actor dead."""

from __future__ import annotations

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from h3_multinode.executor import (
    RayDiffusionExecutor,
    interpret_health_rpc,
    is_busy_actor_timeout,
)


class GetTimeoutError(Exception):
    """Stand-in for ray.exceptions.GetTimeoutError (name-matched)."""


class _Ping:
    def remote(self):
        return "ref"


class _Actor:
    def __init__(self):
        self.ping = _Ping()


def test_short_ping_timeout_is_busy_not_dead():
    assert is_busy_actor_timeout(TimeoutError("ray.get timed out"))
    assert is_busy_actor_timeout(GetTimeoutError("timed out"))
    assert interpret_health_rpc(TimeoutError("ray.get timed out"), None) == "busy"
    assert interpret_health_rpc(GetTimeoutError("timed out"), None) == "busy"
    assert interpret_health_rpc(None, [0, 1]) == "ok"
    assert interpret_health_rpc(None, [0]) == "dead"
    assert interpret_health_rpc(RuntimeError("ActorDiedError"), None) == "dead"


def test_check_health_timeout_does_not_mark_executor_dead():
    executor = RayDiffusionExecutor.__new__(RayDiffusionExecutor)
    executor._closed = False
    executor._is_failed = False
    executor._rpc_in_flight = 0
    executor._failure_callbacks = []

    class _Ray:
        def get(self, _refs, timeout=None):
            raise TimeoutError(f"timed out after {timeout}s")

    executor._ray = _Ray()
    executor._actors = [_Actor(), _Actor()]
    executor.check_health()
    assert executor._is_failed is False
    assert executor.is_dead is False


def test_check_health_skips_ping_while_generate_rpc_in_flight():
    executor = RayDiffusionExecutor.__new__(RayDiffusionExecutor)
    executor._closed = False
    executor._is_failed = False
    executor._rpc_in_flight = 1
    executor._failure_callbacks = []
    executor._actors = [object(), object()]

    class _Ray:
        def get(self, _refs, timeout=None):
            raise AssertionError("ping must not run during generate")

    executor._ray = _Ray()
    executor.check_health()
    assert executor._is_failed is False


def test_check_health_actor_crash_marks_executor_dead():
    executor = RayDiffusionExecutor.__new__(RayDiffusionExecutor)
    executor._closed = False
    executor._is_failed = False
    executor._rpc_in_flight = 0
    executor._failure_callbacks = []

    class _Ray:
        def get(self, _refs, timeout=None):
            raise RuntimeError("actor died")

    executor._ray = _Ray()
    executor._actors = [_Actor(), _Actor()]
    with pytest.raises(EngineDeadError):
        executor.check_health()
    assert executor._is_failed is True
