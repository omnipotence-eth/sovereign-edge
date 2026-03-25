"""Base interface that all agent squads must implement."""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import TaskRequest, TaskResult


class BaseSquad(ABC):
    """Interface that all squads must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Squad identifier (e.g., 'spiritual')."""
        ...

    @abstractmethod
    async def process(self, task: TaskRequest) -> TaskResult:
        """Process a task and return a result."""
        ...

    @abstractmethod
    async def morning_brief(self) -> str:
        """Generate content for the morning digest. Called at 05:00 CT."""
        ...

    async def health_check(self) -> bool:
        """Return True if squad is operational."""
        return True
