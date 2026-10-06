"""AgentCore Platform v1.0"""

# src/graph/context_bridge.py — carries the caller's input_context across the
# outer→inner graph boundary.
#
# Why this exists: the framework's GraphNode.execute() invokes the inner graph as
# `subgraph.invoke(user_input, session_id=..., ctx=...)` and does NOT forward the
# outer state's input_context, so an inner node reading state["input_context"]
# would always see {} through the full nested graph — and every declared caller
# parameter would be silently ignored rather than rejected. The sanctioned
# subclass hooks bridge it:
#
#   MedicalAssessmentWorkflowGraphNode.extract_input(state)  [before subgraph.invoke]
#       → set_caller_input_context(state["input_context"])
#   MedicalAssessmentDomainGraph._extra_initial_state()      [inside subgraph.invoke]
#       → returns {"input_context": get_caller_input_context()}
#
# A ContextVar keeps the hand-off correct per thread/task, so concurrent
# invocations in one process cannot see each other's context.

from contextvars import ContextVar
from typing import Any, Dict, Optional

_CALLER_INPUT_CONTEXT: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "ins_c2_051_caller_input_context", default=None
)


def set_caller_input_context(input_context: Optional[Dict[str, Any]]) -> None:
    """Stash the outer graph's input_context for the imminent inner-graph invoke."""
    _CALLER_INPUT_CONTEXT.set(dict(input_context) if input_context else {})


def get_caller_input_context() -> Dict[str, Any]:
    """Read (without consuming) the stashed input_context; {} when none was set."""
    return _CALLER_INPUT_CONTEXT.get() or {}
