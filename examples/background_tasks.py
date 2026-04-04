"""Background tasks example — task lifecycle with TaskStore.

Demonstrates the task tools that let agents spawn, list, inspect,
and stop background tasks. This example exercises the TaskStore
directly without needing an LLM.

Run::

    python examples/background_tasks.py
"""

from __future__ import annotations

import asyncio

from orxhestra.tools.task_tools import BackgroundTask, TaskStore


async def _simulate_work(task: BackgroundTask, seconds: float) -> None:
    """Simulate a background job that writes output when done."""
    try:
        await asyncio.sleep(seconds)
        task.output = f"Finished after {seconds}s"
        task.status = "completed"
    except asyncio.CancelledError:
        task.status = "stopped"
        task.output = "Task was stopped."


async def main() -> None:
    """Run through the full task lifecycle."""
    store = TaskStore()

    # --- Create tasks ---
    task_a = BackgroundTask(id="a1", subject="Data export", description="Export CSV")
    task_b = BackgroundTask(id="b2", subject="Send emails", description="Notify users")

    store.add(task_a)
    store.add(task_b)

    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    task_a._task = loop.create_task(_simulate_work(task_a, 0.3))
    task_b._task = loop.create_task(_simulate_work(task_b, 5.0))

    print("Created tasks:")
    for t in store.list_all():
        print(f"  {t.id} [{t.status}] {t.subject}")

    # --- Wait for the fast task to finish ---
    await asyncio.sleep(0.5)

    print(f"\nTask a1 status : {store.get('a1').status}")
    print(f"Task a1 output : {store.get('a1').output}")

    # --- Stop the slow task ---
    slow: BackgroundTask = store.get("b2")
    if slow._task and not slow._task.done():
        slow._task.cancel()
        try:
            await slow._task
        except asyncio.CancelledError:
            pass

    print(f"\nTask b2 status : {slow.status}")
    print(f"Task b2 output : {slow.output}")

    # --- List final state ---
    print("\nFinal task list:")
    for t in store.list_all():
        print(f"  {t.id} [{t.status}] {t.subject} — {t.output}")


if __name__ == "__main__":
    asyncio.run(main())
