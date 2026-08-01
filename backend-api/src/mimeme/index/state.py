from __future__ import annotations

from mimeme.index.model import Phase, State


class InvalidTransition(Exception):
    pass


_ALLOWED: dict[Phase, set[Phase]] = {
    Phase.NEW: {Phase.PREPARED, Phase.CANCELLED, Phase.FAILED},
    Phase.PREPARED: {Phase.BUILT, Phase.CANCELLED, Phase.FAILED},
    Phase.BUILT: {Phase.ACTIVE, Phase.CANCELLED, Phase.FAILED},
    Phase.ACTIVE: {Phase.RELEASED},
    Phase.CANCELLED: {Phase.RELEASED},
    Phase.FAILED: {Phase.RELEASED},
    Phase.RELEASED: set(),
}


def transition(state: State, phase: Phase) -> State:
    if phase not in _ALLOWED[state.phase]:
        raise InvalidTransition(f"cannot move from {state.phase.value} to {phase.value}")
    return state.model_copy(update={"phase": phase})
