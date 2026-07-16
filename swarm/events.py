"""
Compatibility entrypoint for the Swarm event system.

The project contains a timestamped event module from a prior generation step,
while the runtime imports `swarm.events`. Keep this thin wrapper so existing
imports work without renaming the generated source file.
"""
from .events_20260428_231035 import Event, EventType

__all__ = ["Event", "EventType"]
