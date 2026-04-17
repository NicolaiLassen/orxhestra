"""Tests for gather_with_event_queue concurrency utility."""

from __future__ import annotations

import asyncio

import pytest

from orxhestra.concurrency import gather_with_event_queue


@pytest.mark.asyncio
async def test_empty_coros_yields_nothing():
    queue: asyncio.Queue = asyncio.Queue()
    items = [item async for item in gather_with_event_queue([], queue)]
    assert items == []


@pytest.mark.asyncio
async def test_results_in_original_order():
    """Tasks completing out of order still yield results in original order."""
    queue: asyncio.Queue = asyncio.Queue()

    async def fast():
        return "fast"

    async def slow():
        await asyncio.sleep(0.05)
        return "slow"

    items = [item async for item in gather_with_event_queue([slow(), fast()], queue)]
    # Results should be in original order: slow first, fast second
    assert items == ["slow", "fast"]


@pytest.mark.asyncio
async def test_queue_events_yielded_before_results():
    """Events pushed to the queue during execution appear before task results."""
    queue: asyncio.Queue = asyncio.Queue()

    async def worker():
        queue.put_nowait("event_1")
        queue.put_nowait("event_2")
        await asyncio.sleep(0.02)
        return "result"

    items = [item async for item in gather_with_event_queue([worker()], queue)]

    # Events should come first, result last
    assert items[0] == "event_1"
    assert items[-1] == "result"
    assert "event_2" in items


@pytest.mark.asyncio
async def test_queue_events_interleaved_across_tasks():
    """Multiple tasks pushing events — all events arrive before results."""
    queue: asyncio.Queue = asyncio.Queue()

    async def task_a():
        queue.put_nowait("a_progress")
        await asyncio.sleep(0.03)
        return "a_done"

    async def task_b():
        queue.put_nowait("b_progress")
        await asyncio.sleep(0.01)
        return "b_done"

    items = [item async for item in gather_with_event_queue([task_a(), task_b()], queue)]

    events = [i for i in items if isinstance(i, str) and "progress" in i]
    results = [i for i in items if isinstance(i, str) and "done" in i]

    assert set(events) == {"a_progress", "b_progress"}
    assert results == ["a_done", "b_done"]  # original order

    # All events before all results
    last_event_idx = max(items.index(e) for e in events)
    first_result_idx = min(items.index(r) for r in results)
    assert last_event_idx < first_result_idx


@pytest.mark.asyncio
async def test_events_pushed_after_task_completion_still_drained():
    """Events pushed just before a task returns are still yielded."""
    queue: asyncio.Queue = asyncio.Queue()

    async def worker():
        queue.put_nowait("late_event")
        return "done"

    items = [item async for item in gather_with_event_queue([worker()], queue)]
    assert "late_event" in items
    assert "done" in items
