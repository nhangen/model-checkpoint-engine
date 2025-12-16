# Optimized cleanup scheduler - zero redundancy design

import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .retention_manager import RetentionManager, RetentionRule


def _current_time() -> float:
    # Shared time function
    return time.time()


class ScheduleType(Enum):
    # Optimized schedule type enum

    INTERVAL = "interval"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class ScheduledTask:
    # Optimized scheduled task

    name: str
    schedule_type: ScheduleType
    retention_rules: List[str] = field(default_factory=list)
    interval_seconds: Optional[float] = None
    hour: int = 2  # Default to 2 AM
    day_of_week: int = 0  # Monday
    day_of_month: int = 1
    enabled: bool = True
    dry_run: bool = True
    last_run: float = 0.0
    next_run: float = 0.0
    run_count: int = 0
    custom_function: Optional[Callable] = None


class CleanupScheduler:
    # Optimized cleanup scheduler with thread-safe execution

    def __init__(self, retention_manager: RetentionManager):
        """
        Initialize cleanup scheduler

        Args:
            retention_manager: Retention manager instance
        """
        self.retention_manager = retention_manager

        # Optimized: Thread-safe state management
        self._tasks: Dict[str, ScheduledTask] = {}
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Optimized: Pre-defined common schedules
        self._default_schedules = {
            "daily_cleanup": ScheduledTask(
                name="daily_cleanup",
                schedule_type=ScheduleType.DAILY,
                retention_rules=["keep_recent_7_days"],
                hour=2,
                dry_run=False,
            ),
            "weekly_deep_clean": ScheduledTask(
                name="weekly_deep_clean",
                schedule_type=ScheduleType.WEEKLY,
                retention_rules=["keep_best_10", "limit_size_1gb"],
                hour=3,
                day_of_week=0,  # Monday
                dry_run=False,
            ),
            "hourly_size_check": ScheduledTask(
                name="hourly_size_check",
                schedule_type=ScheduleType.INTERVAL,
                retention_rules=["limit_size_1gb"],
                interval_seconds=3600.0,  # 1 hour
                dry_run=True,
            ),
        }

    def add_scheduled_task(self, task: ScheduledTask) -> None:
        # Add a scheduled task - thread-safe
        with self._lock:
            task.next_run = self._calculate_next_run(task)
            self._tasks[task.name] = task

    def get_scheduled_task(self, name: str) -> Optional[ScheduledTask]:
        # Get scheduled task by name
        with self._lock:
            return self._tasks.get(name) or self._default_schedules.get(name)

    def apply_default_schedule(self, schedule_name: str) -> bool:
        # Apply a default schedule
        if schedule_name in self._default_schedules:
            self.add_scheduled_task(self._default_schedules[schedule_name])
            return True
        return False

    def remove_scheduled_task(self, name: str) -> bool:
        # Remove a scheduled task - thread-safe
        with self._lock:
            return self._tasks.pop(name, None) is not None

    def start_scheduler(self) -> bool:
        # Start the scheduler thread - optimized startup
        with self._lock:
            if self._running:
                return False

            self._running = True
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop, daemon=True
            )
            self._scheduler_thread.start()
            return True

    def stop_scheduler(self) -> bool:
        # Stop the scheduler thread - optimized shutdown
        with self._lock:
            if not self._running:
                return False

            self._running = False

        # Wait for thread to finish
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)

        return True

    def _scheduler_loop(self) -> None:
        # Main scheduler loop - optimized polling
        while self._running:
            try:
                current_time = _current_time()
                tasks_to_run = []

                # Optimized: Find tasks to run in single pass
                with self._lock:
                    for task in self._tasks.values():
                        if task.enabled and current_time >= task.next_run:
                            tasks_to_run.append(task)

                # Execute tasks outside of lock
                for task in tasks_to_run:
                    self._execute_task(task)

                # Optimized: Sleep with shorter intervals for responsiveness
                time.sleep(10.0)  # Check every 10 seconds

            except Exception as e:
                print(f"Scheduler loop error: {e}")
                time.sleep(30.0)  # Longer sleep on error

    def _execute_task(self, task: ScheduledTask) -> None:
        # Execute a scheduled task - optimized execution
        try:
            current_time = _current_time()

            if task.custom_function:
                # Execute custom function
                task.custom_function(self.retention_manager)
            else:
                # Execute retention rules
                candidates = self.retention_manager.find_cleanup_candidates(
                    rules=task.retention_rules
                )
                self.retention_manager.execute_cleanup(candidates, dry_run=task.dry_run)

            # Update task state
            with self._lock:
                task.last_run = current_time
                task.run_count += 1
                task.next_run = self._calculate_next_run(task)

        except Exception as e:
            print(f"Task execution error for '{task.name}': {e}")

    def _calculate_next_run(self, task: ScheduledTask) -> float:
        # Calculate next run time - optimized calculation
        current_time = _current_time()

        if task.schedule_type == ScheduleType.INTERVAL:
            if task.interval_seconds:
                return current_time + task.interval_seconds
            else:
                return current_time + 3600  # Default 1 hour

        elif task.schedule_type == ScheduleType.DAILY:
            # Optimized: Calculate next daily run
            import datetime

            now = datetime.datetime.fromtimestamp(current_time)
            next_run = now.replace(hour=task.hour, minute=0, second=0, microsecond=0)

            # If we've passed today's time, schedule for tomorrow
            if next_run <= now:
                next_run += datetime.timedelta(days=1)

            return next_run.timestamp()

        elif task.schedule_type == ScheduleType.WEEKLY:
            # Optimized: Calculate next weekly run
            import datetime

            now = datetime.datetime.fromtimestamp(current_time)

            # Calculate days until next occurrence
            days_ahead = task.day_of_week - now.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7

            next_run = now + datetime.timedelta(days=days_ahead)
            next_run = next_run.replace(
                hour=task.hour, minute=0, second=0, microsecond=0
            )

            return next_run.timestamp()

        elif task.schedule_type == ScheduleType.MONTHLY:
            # Optimized: Calculate next monthly run
            import datetime

            now = datetime.datetime.fromtimestamp(current_time)

            try:
                next_run = now.replace(
                    day=task.day_of_month,
                    hour=task.hour,
                    minute=0,
                    second=0,
                    microsecond=0,
                )

                # If we've passed this month's date, go to next month
                if next_run <= now:
                    if now.month == 12:
                        next_run = next_run.replace(year=now.year + 1, month=1)
                    else:
                        next_run = next_run.replace(month=now.month + 1)

                return next_run.timestamp()

            except ValueError:
                # Handle invalid day (e.g., Feb 30)
                return current_time + 86400  # Try again tomorrow

        else:
            # Default fallback
            return current_time + 3600

    def get_task_status(self) -> Dict[str, Any]:
        # Get status of all scheduled tasks - optimized reporting
        with self._lock:
            current_time = _current_time()

            status = {
                "scheduler_running": self._running,
                "total_tasks": len(self._tasks),
                "enabled_tasks": sum(1 for t in self._tasks.values() if t.enabled),
                "tasks": [],
            }

            # Optimized: Single pass through all tasks
            for task in self._tasks.values():
                time_until_next = max(0, task.next_run - current_time)
                time_since_last = (
                    current_time - task.last_run if task.last_run > 0 else 0
                )

                task_info = {
                    "name": task.name,
                    "schedule_type": task.schedule_type.value,
                    "enabled": task.enabled,
                    "dry_run": task.dry_run,
                    "run_count": task.run_count,
                    "last_run_ago_seconds": time_since_last,
                    "next_run_in_seconds": time_until_next,
                    "retention_rules": task.retention_rules,
                }
                status["tasks"].append(task_info)

            return status

    def run_task_now(self, task_name: str) -> Dict[str, Any]:
        """
        Run a specific task immediately - optimized execution

        Args:
            task_name: Name of task to run

        Returns:
            Execution results
        """
        task = self.get_scheduled_task(task_name)
        if not task:
            return {"error": f'Task "{task_name}" not found'}

        try:
            current_time = _current_time()

            if task.custom_function:
                # Execute custom function
                task.custom_function(self.retention_manager)
                result = {"status": "completed", "type": "custom_function"}
            else:
                # Execute retention rules
                candidates = self.retention_manager.find_cleanup_candidates(
                    rules=task.retention_rules
                )
                result = self.retention_manager.execute_cleanup(
                    candidates, dry_run=task.dry_run
                )

            # Update task state
            with self._lock:
                if task.name in self._tasks:  # Only update if it's a user task
                    task.last_run = current_time
                    task.run_count += 1

            result["executed_at"] = current_time
            result["task_name"] = task_name

            return result

        except Exception as e:
            return {"error": f"Task execution failed: {e}", "task_name": task_name}

    def simulate_schedule(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Simulate scheduled runs for next N days - optimized simulation

        Args:
            days: Number of days to simulate

        Returns:
            List of simulated executions
        """
        current_time = _current_time()
        end_time = current_time + (days * 24 * 3600)

        executions = []

        with self._lock:
            # Optimized: Calculate all execution times
            for task in self._tasks.values():
                if not task.enabled:
                    continue

                next_run = task.next_run
                temp_task = ScheduledTask(
                    name=task.name,
                    schedule_type=task.schedule_type,
                    retention_rules=task.retention_rules,
                    interval_seconds=task.interval_seconds,
                    hour=task.hour,
                    day_of_week=task.day_of_week,
                    day_of_month=task.day_of_month,
                )

                # Generate all runs within the time window
                while next_run <= end_time:
                    executions.append(
                        {
                            "task_name": task.name,
                            "scheduled_time": next_run,
                            "days_from_now": (next_run - current_time) / (24 * 3600),
                            "retention_rules": task.retention_rules,
                            "dry_run": task.dry_run,
                        }
                    )

                    # Calculate next occurrence
                    temp_task.next_run = next_run
                    next_run = self._calculate_next_run(temp_task)

        # Sort by execution time
        executions.sort(key=lambda x: x["scheduled_time"])

        return executions

    def export_schedule_config(
        self, format_type: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        # Export scheduler configuration
        with self._lock:
            config_data = {
                "scheduler_status": {
                    "running": self._running,
                    "total_tasks": len(self._tasks),
                },
                "tasks": {},
            }

            for name, task in self._tasks.items():
                config_data["tasks"][name] = {
                    "schedule_type": task.schedule_type.value,
                    "retention_rules": task.retention_rules,
                    "interval_seconds": task.interval_seconds,
                    "hour": task.hour,
                    "day_of_week": task.day_of_week,
                    "day_of_month": task.day_of_month,
                    "enabled": task.enabled,
                    "dry_run": task.dry_run,
                    "run_count": task.run_count,
                    "last_run": task.last_run,
                }

        if format_type == "json":
            return json.dumps(config_data, indent=2, default=str)
        else:
            return config_data

    def load_schedule_config(self, config_data: Dict[str, Any]) -> bool:
        # Load scheduler configuration from data
        try:
            with self._lock:
                self._tasks.clear()

                for name, task_config in config_data.get("tasks", {}).items():
                    task = ScheduledTask(
                        name=name,
                        schedule_type=ScheduleType(task_config["schedule_type"]),
                        retention_rules=task_config.get("retention_rules", []),
                        interval_seconds=task_config.get("interval_seconds"),
                        hour=task_config.get("hour", 2),
                        day_of_week=task_config.get("day_of_week", 0),
                        day_of_month=task_config.get("day_of_month", 1),
                        enabled=task_config.get("enabled", True),
                        dry_run=task_config.get("dry_run", True),
                        run_count=task_config.get("run_count", 0),
                        last_run=task_config.get("last_run", 0.0),
                    )

                    task.next_run = self._calculate_next_run(task)
                    self._tasks[name] = task

            return True

        except Exception as e:
            print(f"Failed to load schedule configuration: {e}")
            return False
