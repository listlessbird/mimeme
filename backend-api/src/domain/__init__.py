"""Core backend behavior modules.

The domain package owns behavior that should stay stable across runtime
adapters. HTTP routes, Temporal workflows, Temporal activities, and external
runtime integrations should translate into this package rather than duplicate
business rules in place.

Domain modules should be added only when they move real behavior behind a small
interface. Avoid placeholder modules and hypothetical adapter interfaces.
"""

