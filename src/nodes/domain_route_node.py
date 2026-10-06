"""AgentCore Platform v1.0 — INS-C2-051 DomainRouteNode (inner domain graph, step 2)."""

# Routes the parsed medical question to the biomedical domain model whose
# passages carry the most signal for an underwriting decision.
#
# Routing is rule-based on query_intent, and a caller may override it with the
# `domain_hint` context parameter. The override is a CLOSED SET — the same
# catalogue this module publishes — so the routing decision can never be steered
# to a value the agent does not serve. The caller's value has already been
# validated by PreProcessNode; the closed-set check here is the second reading of
# the same rule and the one the inner pipeline actually depends on.
#
# Retrieval runs on device: no external network call at this step.
#
# Node contract:
#   - Extend FunctionNode; implement execute(state) -> dict (partial state update)
#   - Return ONLY the fields this node changes (never the full state)
#   - required_trust_level = ANONYMOUS (inner subgraph node)

from typing import Any, ClassVar, Dict

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

# ── Biomedical domain catalogue ───────────────────────────────────────────────
# The domain identifiers this agent serves. Published (not private) because
# PreProcessNode validates the caller's `domain_hint` against exactly this set —
# one definition, two readers, no chance of the two drifting apart.
OPENMED_DOMAINS: frozenset[str] = frozenset(
    {
        "diseases",
        "oncology",
        "anatomy",
        "genes",
        "phenotypes",
        "chemicals",
        "trials",
        "general",
    }
)

# ── Intent → domain routing table ────────────────────────────────────────────
_INTENT_TO_DOMAIN: Dict[str, str] = {
    "icd11_lookup": "diseases",
    "prognosis": "diseases",
    "treatment_protocol": "chemicals",
    "oncology": "oncology",
    "disease_inquiry": "diseases",
    "general": "general",
}

# Fallback when intent is unrecognised
_DEFAULT_DOMAIN: str = "general"


class DomainRouteNode(FunctionNode):
    """Selects the biomedical domain model for this question.

    Uses query_intent (from QueryParseNode) unless the caller declared a
    `domain_hint`, in which case that value — already validated against
    OPENMED_DOMAINS — wins. The selection is audited either way, with its source,
    so a downstream reader can tell a caller-directed retrieval from an inferred one.

    Assigned to the inner MedicalAssessmentDomainGraph (step 2).
    Trust level ANONYMOUS: inner subgraph node; S-1 gate is on the outer PreProcessNode.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        query_intent = state.get("query_intent", "general")
        validated_input = state.get("validated_input", state.get("user_input", ""))
        context = state.get("input_context", {}) or {}

        if not validated_input:
            emit_trace_event(
                "ins_c2_051.domain_route.error",
                {"reason": "no_validated_input"},
                state,
            )
            return {
                "status": AgentStatus.ERROR,
                "error_log": ["DomainRouteNode: no validated_input in state"],
            }

        hint = context.get("domain_hint") if isinstance(context, dict) else None
        if isinstance(hint, str) and hint in OPENMED_DOMAINS:
            domain = hint
            source = "caller_hint"
        else:
            domain = _INTENT_TO_DOMAIN.get(query_intent, _DEFAULT_DOMAIN)
            source = "intent"

        # Second reading of the closed set: a routing value that is not served is
        # replaced by the default rather than passed on to the retriever.
        if domain not in OPENMED_DOMAINS:
            domain = _DEFAULT_DOMAIN
            source = "default"

        emit_trace_event(
            "ins_c2_051.domain_route.selected",
            {
                "query_intent": query_intent,
                "domain_selected": domain,
                "selection_source": source,
            },
            state,
        )

        return {
            "domain_routed": domain,
            "status": AgentStatus.SUCCESS,
        }
