"""AgentCore Platform v1.0 — INS-C2-051 PreProcessNode (pre_process backbone)."""

# S-1/S-2 external gate for INS-C2-051 — the node that owns the caller contract.
#
# Everything a caller can influence is validated here, before the question enters
# the biomedical retrieval pipeline, and every rejection is fail-closed:
#
#   * the question itself — non-empty, size-capped, free of patient identifiers,
#     and free of instruction-override content (control tokens included);
#   * the four declared context parameters — two inert identifiers, one closed-set
#     domain selector and one finite bounded integer. An undeclared parameter is
#     rejected rather than ignored: ignoring an unknown key leaves it in state,
#     where the framework's own output gate scans it and fails the first node with
#     a traceback the caller cannot act on.
#
# The refusals are enforced in execute(), not in a security hook, so they can be
# proved by calling execute() directly with no framework wrapper in front — the
# guarantee then does not depend on a framework gate being active or configured.
#
# Rejected values are never echoed: a refusal names the field, never its content.
#
# Node contract:
#   - Extend FunctionNode; implement execute(state) -> dict (partial state update)
#   - Return ONLY the fields this node changes (never the full state)
#   - Return AgentStatus enum constants — never plain strings [A1]
#   - Read input_context via state.get("input_context", {}) — read-only [C1]
#   - Never import from mediator/, api/, or other agents

import json
from typing import Any, ClassVar, Dict, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.nodes.biobert_retrieve_node import TOP_K_MAX, TOP_K_MIN
from src.nodes.domain_route_node import OPENMED_DOMAINS
from src.services.service import (
    detect_patient_identifier,
    finite_int_in_range,
    is_inert_token,
    safe_field_label,
    screen_structure,
    screen_text,
)

# Medical assessment questions longer than this are not questions.
_MAX_QUERY_CHARS: int = 4_000

# The complete set of context parameters this agent accepts. Anything else is
# refused by name, so the accepted surface is exactly what is documented.
# Keys the platform itself puts into input_context, not the caller. The Marketplace
# runner invokes every agent as
#     agent.invoke(message, ctx=ctx, input_context={"conversation_history": history})
# (agenticstar-agentcore, shared/bootstrap/marketplace_app.py), whatever the user typed.
# Refusing it as an unknown field refused every chat request before the question was
# read. Discarded, not validated: nothing in this pipeline reads prior turns, and
# screening a transcript would let one earlier message refuse every later one. Discarding
# adds no exposure — the backbone's first node has already copied the raw input_context
# into state before this contract runs.
PLATFORM_RESERVED_KEYS = frozenset({"conversation_history"})


ACCEPTED_CONTEXT_FIELDS: Tuple[str, ...] = (
    "channel",
    "underwriter_id",
    "domain_hint",
    "max_passages",
)

# Retrieval depth bounds are published by the node that consumes them
# (BioBERTRetrieveNode) and validated here, so the contract this gate enforces
# and the depth the retriever applies cannot drift apart.


def _validate_context(input_context: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate the declared context parameters.

    Returns ``(validated, None)`` or ``(None, field_label)`` naming the first
    field that failed. The value that failed is never returned, so a refusal
    cannot carry caller content into an error log.
    """
    if input_context in (None, {}):
        return {}, None
    if not isinstance(input_context, dict):
        return None, "input_context"
    input_context = {k: v for k, v in input_context.items() if k not in PLATFORM_RESERVED_KEYS}
    if not input_context:
        return {}, None

    validated: Dict[str, Any] = {}
    for position, (key, value) in enumerate(input_context.items(), start=1):
        label = safe_field_label(key, position)
        if key not in ACCEPTED_CONTEXT_FIELDS:
            return None, label
        if key in ("channel", "underwriter_id"):
            if not is_inert_token(value):
                return None, label
            validated[key] = value
        elif key == "domain_hint":
            if not isinstance(value, str) or value not in OPENMED_DOMAINS:
                return None, label
            validated[key] = value
        else:  # max_passages
            parsed = finite_int_in_range(value, TOP_K_MIN, TOP_K_MAX)
            if parsed is None:
                return None, label
            validated[key] = parsed
    return validated, None


class PreProcessNode(FunctionNode):
    """S-1/S-2 external gate for INS-C2-051 InsuranceMedicalAssessmentAgent.

    Validates and sanitizes the incoming medical assessment question and its
    declared context parameters before either reaches the inner domain workflow.

    Assigned to the outer backbone `pre_process` slot.
    Trust level VERIFIED_EXTERNAL: this is the external-facing security gate;
    inner domain nodes declare ANONYMOUS (sub-pipeline trust).
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        input_context = state.get("input_context", {})  # read-only [C1]

        if not isinstance(user_input, str) or not user_input.strip():
            emit_trace_event(
                "ins_c2_051.pre_process.rejected",
                {"reason": "empty_input"},
                state,
            )
            return {
                "status": AgentStatus.ERROR,
                "error_log": ["PreProcessNode: field 'input' is empty or missing"],
            }

        sanitized = user_input.strip()

        # Size ceiling — an oversized body is a bulk-data submission, not a question.
        if len(sanitized) > _MAX_QUERY_CHARS:
            emit_trace_event(
                "ins_c2_051.pre_process.rejected",
                {"reason": "input_too_large", "size": len(sanitized)},
                state,
            )
            return {
                "status": AgentStatus.ERROR,
                "error_log": [
                    f"PreProcessNode: field 'input' exceeds the {_MAX_QUERY_CHARS}-character limit "
                    f"({len(sanitized)} characters) — a medical assessment question must be concise"
                ],
            }

        # Patient-identifier scan. The kind is audited; the value never is.
        identifier_kind = detect_patient_identifier(sanitized)
        if identifier_kind is not None:
            emit_trace_event(
                "ins_c2_051.pre_process.rejected",
                {"reason": "patient_identifier_detected", "kind": identifier_kind},
                state,
            )
            return {
                "status": AgentStatus.ERROR,
                "error_log": [
                    "PreProcessNode: field 'input' carries a patient identifier and was refused — "
                    "questions must be anonymised (no policy numbers, patient IDs or individual numbers)"
                ],
            }

        # Instruction-override screen: the question raw and markup-stripped, then
        # the whole parsed context structure including its keys.
        finding = screen_text(sanitized) or screen_structure(input_context)
        if finding is not None:
            emit_trace_event(
                "ins_c2_051.pre_process.rejected",
                {"reason": "instruction_override_detected", "pattern": finding},
                state,
            )
            return {
                "status": AgentStatus.ERROR,
                "error_log": ["PreProcessNode: the request carries instruction-override content and was refused"],
            }

        validated_context, bad_field = _validate_context(input_context)
        if validated_context is None:
            emit_trace_event(
                "ins_c2_051.pre_process.rejected",
                {"reason": "context_field_invalid", "field": bad_field},
                state,
            )
            return {
                "status": AgentStatus.ERROR,
                "error_log": [
                    f"PreProcessNode: context field '{bad_field}' is not accepted — "
                    "channel and underwriter_id must be 1-64 characters of [A-Za-z0-9_-], "
                    f"domain_hint must name a supported domain, and max_passages must be an "
                    f"integer in [{TOP_K_MIN}, {TOP_K_MAX}]"
                ],
            }

        channel = validated_context.get("channel", "unknown")
        underwriter_id = validated_context.get("underwriter_id", "")

        emit_trace_event(
            "ins_c2_051.pre_process.accepted",
            {
                "input_size": len(sanitized),
                "channel": channel,
                "underwriter_id": underwriter_id,
                "domain_hint": validated_context.get("domain_hint", ""),
                "max_passages": validated_context.get("max_passages", 0),
            },
            state,
        )

        return {
            "validated_input": sanitized,
            # A mapping, matching the field's declared type. It used to be built by
            # interpolating the two caller values into a JSON string, which is how
            # an inert-looking field becomes structure the next reader misparses;
            # the values are inert now and the shape is no longer hand-assembled.
            "enriched_context": {
                "source": "InsuranceMedicalAssessmentAgent",
                "channel": channel,
                "underwriter_id": underwriter_id,
            },
            # The VALIDATED parameters — serialised per the state convention — are
            # what crosses into the inner graph (src/graph/context_bridge.py). The
            # raw input_context never does, so no inner node can read a value this
            # gate has not accepted.
            "validated_context": json.dumps(validated_context, ensure_ascii=False, separators=(",", ":")),
            "status": AgentStatus.SUCCESS,
        }
