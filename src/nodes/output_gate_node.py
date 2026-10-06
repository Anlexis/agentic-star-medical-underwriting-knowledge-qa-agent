"""AgentCore Platform v1.0 — INS-C2-051 OutputGateNode (inner domain graph, step 5)."""

# Final gate of the inner pipeline: it decides whether the assessment becomes the
# subgraph's output at all.
#
# Two things happen here that are easy to get wrong:
#
#   1. An error raised earlier in the pipeline is honoured. The inner topology is
#      a straight line with no conditional edges, so every node runs even after
#      one of them fails — and a later node returning SUCCESS overwrites the
#      earlier ERROR. The last node's status is the one the subgraph reports, so
#      this node refuses to report success on a run that already failed.
#   2. Nothing output-bearing is published on a refusal. The assessment stays in
#      the inner state where the graph's own get_output() can no longer reach it
#      (see MedicalAssessmentDomainGraph.get_output), so a refused run carries no
#      un-gated text out of the subgraph.
#
# Node contract:
#   - Extend FunctionNode; implement execute(state) -> dict (partial state update)
#   - Return ONLY the fields this node changes (never the full state)
#   - required_trust_level = ANONYMOUS (inner subgraph node)

import json
from typing import Any, ClassVar, Dict

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json

# The bracketed marker the outer PostProcessNode also checks for.
_MANDATORY_DISCLAIMER: str = "[INSURANCE ASSESSMENT DISCLAIMER]"


class OutputGateNode(FunctionNode):
    """Inner pipeline output gate for INS-C2-051.

    Verifies the assessment carries the mandatory disclaimer, finalises the
    regulatory audit entry, and publishes the assessment as the subgraph's
    formatted output. On any refusal it publishes nothing.

    Assigned to the inner MedicalAssessmentDomainGraph (step 5 — final).
    Trust level ANONYMOUS: the backbone output gate is the outer PostProcessNode.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        assessment = state.get("underwriting_assessment")
        audit_log_json = state.get("hokengyoho_audit_log")
        incoming_status = state.get("status")

        # An earlier failure is not erased by reaching the end of the line.
        if incoming_status in (AgentStatus.ERROR, AgentStatus.ERROR.value):
            emit_trace_event(
                "ins_c2_051.output_gate.upstream_error",
                {"reason": "pipeline_already_failed"},
                state,
            )
            return {
                "status": AgentStatus.ERROR,
                "error_log": ["OutputGateNode: an earlier step failed; no assessment was published"],
            }

        if not assessment or not isinstance(assessment, str):
            emit_trace_event(
                "ins_c2_051.output_gate.error",
                {"reason": "no_underwriting_assessment"},
                state,
            )
            return {
                "status": AgentStatus.ERROR,
                "error_log": ["OutputGateNode: no underwriting_assessment in state"],
            }

        disclaimer_present = _MANDATORY_DISCLAIMER in assessment
        if not disclaimer_present:
            emit_trace_event(
                "ins_c2_051.output_gate.disclaimer_absent",
                {"reason": "disclaimer_absent"},
                state,
            )
            return {
                "status": AgentStatus.ERROR,
                "error_log": [
                    "OutputGateNode: the mandatory assessment disclaimer is absent; " "no assessment was published"
                ],
            }

        try:
            audit_entry: Dict[str, Any] = json.loads(audit_log_json) if audit_log_json else {}
        except (json.JSONDecodeError, TypeError):
            audit_entry = {}
        if not isinstance(audit_entry, dict):
            audit_entry = {}

        audit_entry["output_gate_passed"] = True
        audit_entry["disclaimer_verified"] = disclaimer_present

        emit_trace_event(
            "ins_c2_051.output_gate.passed",
            {
                "disclaimer_present": disclaimer_present,
                "assessment_size": len(assessment),
            },
            state,
        )

        return {
            "disclaimer_applied": disclaimer_present,
            "hokengyoho_audit_log": to_json(audit_entry),
            # Publishing formatted_output is what makes the assessment the
            # subgraph's output; a refusal above leaves it unset.
            "formatted_output": assessment,
            "status": AgentStatus.SUCCESS,
        }
