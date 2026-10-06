"""AgentCore Platform v1.0 — INS-C2-051 State definition."""

# ADR-005: State must be a flat TypedDict — never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.
#
# Complex objects (lists, dicts) are serialized to JSON strings before
# storage — use to_json() helper and json.loads() at read sites.

import json
from typing import Any, Optional

from framework.schemas.agent_state import AgentState


def to_json(obj: Any) -> Optional[str]:
    """Serialize obj to a compact JSON string for State storage.

    Returns None for None input.  Round-trips safely through msgpack
    (string primitive) and json.loads() at consumer sites.
    """
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class State(AgentState):
    """Agent state for INS-C2-051 InsuranceMedicalAssessmentAgent.

    All shared fields (user_input, validated_input, status, session_id,
    node_history, error_log, hitl_*, etc.) are inherited from AgentState.

    Domain fields added below follow the pipeline:
      QueryParseNode → DomainRouteNode → BioBERTRetrieveNode
        → UnderwritingAssessmentFormatNode → OutputGateNode
    """

    # ── PreProcessNode output ─────────────────────────────────────────────────
    # JSON string: the caller's context parameters AFTER validation — the only
    # form that crosses into the inner graph (see src/graph/context_bridge.py).
    # e.g. '{"channel":"portal","max_passages":4}'
    validated_context: Optional[str]

    # ── QueryParseNode output ─────────────────────────────────────────────────
    # Classified intent: "disease_inquiry" | "treatment_protocol" | "prognosis"
    # | "icd11_lookup" | "general"
    query_intent: Optional[str]

    # JSON string: list of extracted ICD-11 code strings and prognosis terms
    # e.g. '["ICD-11:5A11","prognosis:favorable"]'
    extracted_medical_terms: Optional[str]

    # ── DomainRouteNode output ─────────────────────────────────────────────────
    # Selected openmed domain model identifier
    # e.g. "diseases" | "oncology" | "anatomy" | "genes" | "general"
    domain_routed: Optional[str]

    # ── BioBERTRetrieveNode output ────────────────────────────────────────────
    # JSON string: list of retrieved passage dicts {text, score, source}
    retrieved_passages: Optional[str]

    # ── UnderwritingAssessmentFormatNode output ───────────────────────────────
    # Formatted underwriting assessment markdown string
    underwriting_assessment: Optional[str]

    # JSON string: Hokengyoho (保険業法) audit log entry for compliance
    hokengyoho_audit_log: Optional[str]

    # ── OutputGateNode output ──────────────────────────────────────────────────
    # True when the mandatory insurance assessment disclaimer is included in output
    disclaimer_applied: Optional[bool]
