"""AgentCore Platform v1.0 — INS-C2-051 UnderwritingAssessmentFormatNode (inner graph, step 4)."""

# Renders the retrieved passages into a structured underwriting assessment and
# generates the regulatory audit entry that accompanies it.
#
# The assessment is a STRUCTURED document: markdown headings, numbered passage
# lines and italic source attributions. Two consequences follow, and both are
# enforced here rather than assumed:
#
#   * the question is quoted back so a reader can see what was asked, and the
#     quote is neutralised first — collapsed to one line with the renderer's own
#     structural characters removed — so no caller string can manufacture a line
#     that reads as a retrieved passage attributed to a named source;
#   * the extracted medical terms come from a bounded pattern and a closed
#     vocabulary, and the number rendered is capped, so a long question cannot
#     inflate the released assessment.
#
# The audit entry records the shape of the assessment, never the question text
# and never any applicant-linked data.
#
# Node contract:
#   - Extend FunctionNode; implement execute(state) -> dict (partial state update)
#   - Return ONLY the fields this node changes (never the full state)
#   - required_trust_level = ANONYMOUS (inner subgraph node)

import json
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json
from src.services.service import is_inert_token, neutralise_echo

# Mandatory disclaimer. OutputGateNode and PostProcessNode both check for the
# bracketed marker, so this string is the one place it is written.
_INSURANCE_DISCLAIMER: str = (
    "[INSURANCE ASSESSMENT DISCLAIMER] This biomedical knowledge summary is "
    "generated for insurance underwriting reference only. It does not constitute "
    "medical advice. All underwriting decisions must be reviewed by a licensed "
    "underwriter and, where applicable, a medical officer. 本情報は保険引受参考資料です。"
    "医療上のアドバイスではありません。引受判断は資格のある引受担当者が行うものとします。"
)

# How many extracted terms are rendered. The question may name more; the released
# document does not grow with the input.
_MAX_RENDERED_TERMS: int = 8


def _format_assessment(
    query_echo: str,
    intent: str,
    passages: List[Dict[str, Any]],
    domain: str,
    terms: List[str],
    underwriter_id: str,
) -> str:
    """Build the structured underwriting assessment markdown."""
    lines: List[str] = [
        f"## Medical Assessment — {domain.upper()} Domain",
        "",
        f"**Query Intent:** {intent}",
        f"**Domain Model:** {domain}",
    ]
    if underwriter_id:
        lines.append(f"**Requested By:** {underwriter_id}")
    lines += [
        "",
        "### Question",
        "",
        f"> {query_echo}" if query_echo else "> _(question not recorded)_",
        "",
    ]
    if terms:
        lines += [
            "### Identified Terms",
            "",
            ", ".join(terms[:_MAX_RENDERED_TERMS]),
            "",
        ]
    lines += [
        "### Biomedical Knowledge Summary",
        "",
    ]
    if passages:
        for i, p in enumerate(passages, 1):
            score = p.get("score", 0)
            score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "0.00"
            lines.append(f"{i}. {p.get('text', '')}  _(source: {p.get('source', '')}, score: {score_text})_")
    else:
        lines.append("_No relevant passages retrieved for this query._")

    lines += [
        "",
        "### Underwriting Considerations",
        "",
        "- Review the disease-code classification for risk banding.",
        "- Cross-reference with policy exclusion clauses for identified conditions.",
        "- Apply the loading factor table per actuarial guidelines.",
        "",
        _INSURANCE_DISCLAIMER,
    ]
    return "\n".join(lines)


class UnderwritingAssessmentFormatNode(FunctionNode):
    """Renders retrieved passages into an underwriting-ready assessment.

    Combines the domain, intent, extracted terms, retrieved passages and a
    neutralised echo of the question into a structured markdown assessment, and
    generates the regulatory audit entry that accompanies it.

    Assigned to the inner MedicalAssessmentDomainGraph (step 4).
    Trust level ANONYMOUS: inner subgraph node; the external gate is PreProcessNode.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        validated_input = state.get("validated_input", state.get("user_input", ""))
        query_intent = state.get("query_intent", "general")
        domain = state.get("domain_routed", "general")
        passages_json = state.get("retrieved_passages")
        terms_json = state.get("extracted_medical_terms")
        context = state.get("input_context", {}) or {}

        if not validated_input:
            emit_trace_event(
                "ins_c2_051.underwriting_format.error",
                {"reason": "no_validated_input"},
                state,
            )
            return {
                "status": AgentStatus.ERROR,
                "error_log": ["UnderwritingAssessmentFormatNode: no validated_input in state"],
            }

        try:
            passages: List[Dict[str, Any]] = json.loads(passages_json) if passages_json else []
        except (json.JSONDecodeError, TypeError):
            passages = []
        if not isinstance(passages, list):
            passages = []

        try:
            terms: List[str] = json.loads(terms_json) if terms_json else []
        except (json.JSONDecodeError, TypeError):
            terms = []
        if not isinstance(terms, list):
            terms = []
        terms = [t for t in terms if isinstance(t, str)]

        # Only an already-inert identifier is rendered; anything else is dropped
        # rather than shown, so the rendering cannot depend on an unchecked value
        # even if this node is called outside the validated pipeline.
        declared_id = context.get("underwriter_id") if isinstance(context, dict) else None
        underwriter_id = declared_id if is_inert_token(declared_id) else ""

        query_echo = neutralise_echo(validated_input)
        assessment = _format_assessment(query_echo, query_intent, passages, domain, terms, str(underwriter_id))

        # Regulatory audit entry (solicitation record). It records the SHAPE of
        # the assessment; the question text and any applicant-linked value are
        # deliberately absent.
        audit_entry: Dict[str, Any] = {
            "event": "underwriting_assessment_generated",
            "domain": domain,
            "intent": query_intent,
            "passage_count": len(passages),
            "term_count": len(terms),
            "disclaimer_included": _INSURANCE_DISCLAIMER in assessment,
        }

        emit_trace_event(
            "ins_c2_051.underwriting_format.complete",
            {
                "domain": domain,
                "intent": query_intent,
                "passage_count": len(passages),
                "assessment_size": len(assessment),
            },
            state,
        )

        return {
            "underwriting_assessment": assessment,
            "hokengyoho_audit_log": to_json(audit_entry),
            "status": AgentStatus.SUCCESS,
        }
