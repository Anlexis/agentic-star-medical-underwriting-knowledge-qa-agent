"""AgentCore Platform v1.0 — INS-C2-051 PostProcessNode (post_process backbone)."""

# The backbone output gate: the last code that sees the assessment before the
# caller does.
#
# Refusing is not the same as containing. The framework's envelope builder
# returns `state["formatted_output"] or state["result"]`, and it does so on an
# error status too — so a gate that returns ERROR while leaving `result` in place
# ships the un-gated text it had just refused, inside the error envelope. Every
# refusal below therefore does three things together: report the error, CLEAR the
# output-bearing state field, and publish a non-empty refusal notice. The notice
# must be non-empty: a falsy formatted_output re-opens the fallback that the
# clearing was meant to close.
#
# The credential screen is a UNION of the framework's own detector and the local
# patterns. That direction matters in both senses. Narrower than the framework
# and a value it catches raises inside the node wrapper — which returns a bare
# error partial and discards this node's clearing, i.e. the leak the clearing
# exists to prevent. Narrower than the local set and an operations note pasted
# into a knowledge base walks straight through, because the framework's patterns
# describe credential FORMATS and not assignment lines.
#
# The gate scans the assessment BEFORE it is published, so the framework's own
# scan of the returned dict never has anything to raise on.
#
# Node contract:
#   - Extend FunctionNode; implement execute(state) -> dict
#   - required_trust_level = ANONYMOUS (post_process slot — the trust gate is on pre_process)
#   - Return ONLY the fields this node changes (never the full state)

from typing import Any, ClassVar, Dict, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.services.service import detect_output_credentials

_MANDATORY_DISCLAIMER: str = "[INSURANCE ASSESSMENT DISCLAIMER]"

# Size ceiling on the released assessment. Anything larger is bulk data being
# re-emitted, not an assessment.
_MAX_OUTPUT_CHARS: int = 50_000

# Applicant-identifier cues that must never appear in a released assessment.
_FORBIDDEN_IDENTIFIER_MARKERS: Tuple[str, ...] = (
    "policy_no:",
    "patient_id:",
    "被保険者番号",
    "個人番号",
)

# Published in place of a refused assessment. Non-empty by design, and a closed
# string: it carries no detail about what was withheld, no field values, no paths.
WITHHELD_NOTICE: str = (
    "The assessment could not be released because it failed the output policy check. "
    "No content was returned. Contact the agent operator if this is unexpected."
)

# Closed set of refusal reasons. What is audited is which rule refused, never the
# text that tripped it.
_REASON_EMPTY = "empty_output"
_REASON_OVERSIZE = "output_too_large"
_REASON_NO_DISCLAIMER = "disclaimer_absent"
_REASON_IDENTIFIER = "identifier_marker_present"
_REASON_CREDENTIAL = "credential_shape_present"


def _security_gate_output(output: Any) -> Tuple[bool, Optional[str]]:
    """Decide whether *output* may be released; return ``(passed, reason)``.

    A module-level function, deliberately: the framework marks the node method of
    the same name ``@final``, and a domain check written there would be refused at
    class-definition time.

    The reason is a closed label from the set above — never the matched text and
    never the value — so a refusal can be audited without re-emitting what it
    withheld.
    """
    if not output or not isinstance(output, str):
        return False, _REASON_EMPTY
    if len(output) > _MAX_OUTPUT_CHARS:
        return False, _REASON_OVERSIZE
    if _MANDATORY_DISCLAIMER not in output:
        return False, _REASON_NO_DISCLAIMER
    lowered = output.lower()
    for marker in _FORBIDDEN_IDENTIFIER_MARKERS:
        if marker.lower() in lowered:
            return False, _REASON_IDENTIFIER
    if detect_output_credentials(output) is not None:
        return False, _REASON_CREDENTIAL
    return True, None


class PostProcessNode(FunctionNode):
    """Backbone output gate for INS-C2-051.

    Reads `result` (written by the main slot's merge_output from the inner
    graph's assessment) and decides whether it may be released. On a refusal it
    clears `result` and publishes the withheld notice, so the error envelope
    carries the notice rather than the assessment.

    Assigned to the outer backbone `post_process` slot.
    Trust level ANONYMOUS (the trust barrier is on pre_process).
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        result = state.get("result")

        gate_passed, gate_reason = _security_gate_output(result)

        emit_trace_event(
            "ins_c2_051.post_process.output_gate_verdict",
            {
                "gate_passed": gate_passed,
                "gate_reason": gate_reason or "",
                "output_size": len(result) if isinstance(result, str) else 0,
            },
            state,
        )

        if not gate_passed:
            return {
                "status": AgentStatus.ERROR,
                # Containment: the refused text is removed from the one state
                # field the envelope falls back to, and the field the envelope
                # prefers is set to a non-empty notice so the fallback is not
                # re-opened by a falsy value.
                "result": "",
                "formatted_output": WITHHELD_NOTICE,
                "error_log": [f"PostProcessNode: output policy check refused the assessment ({gate_reason})"],
            }

        return {
            "formatted_output": result,
            "status": AgentStatus.SUCCESS,
        }
