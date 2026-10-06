"""AgentCore Platform v1.0 — INS-C2-051 QueryParseNode (inner domain graph, step 1)."""

# Parses the validated medical assessment question and extracts structured
# medical terminology: an intent classification plus the disease codes and
# prognosis / treatment terms the routing and retrieval steps read.
#
# Everything here operates on text PreProcessNode has already accepted, so this
# node performs no security check of its own — it must not, or the guarantee
# would be split across two owners and provable at neither.
#
# Node contract:
#   - Extend FunctionNode; implement execute(state) -> dict (partial state update)
#   - Return ONLY the fields this node changes (never the full state)
#   - Return AgentStatus enum constants — never plain strings [A1]
#   - required_trust_level = ANONYMOUS (inner subgraph node; outer PreProcessNode
#     is the external-facing gate; inner nodes declare ANONYMOUS)
#   - Never import from mediator/, api/, or other agents

import re
from typing import Any, ClassVar, Dict, List, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json

# ── Intent classification keywords ────────────────────────────────────────────
# Ordered: the first intent whose keywords appear wins, so the more specific
# categories are listed before the general ones.
_INTENT_MAP: Dict[str, Tuple[str, ...]] = {
    "icd11_lookup": ("icd", "icd-11", "icd11", "disease code", "診断コード", "疾病コード"),
    "prognosis": ("prognosis", "prognose", "survival", "予後", "生存率", "余命"),
    "treatment_protocol": (
        "treatment",
        "therapy",
        "protocol",
        "medication",
        "drug",
        "治療",
        "投薬",
        "処方",
    ),
    "oncology": ("cancer", "tumor", "tumour", "oncology", "malignant", "がん", "腫瘍", "悪性"),
    "disease_inquiry": ("disease", "condition", "syndrome", "disorder", "疾患", "症候群"),
}

# ── Disease code pattern ──────────────────────────────────────────────────────
# Codes follow: letter + up to three alphanumerics + optional suffix, e.g. 5A11,
# BA80.Z, XY5Y. The length ceiling is what keeps this pattern away from the
# 10-12 digit identifier shapes PreProcessNode refuses.
_DISEASE_CODE_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,3}(?:\.[A-Z0-9]{1,2})?\b")

# ── Prognosis / treatment term extraction ────────────────────────────────────
_PROGNOSIS_TERMS: Tuple[str, ...] = (
    "favorable",
    "poor",
    "good",
    "excellent",
    "guarded",
    "予後良好",
    "予後不良",
)
_TREATMENT_TERMS: Tuple[str, ...] = (
    "surgery",
    "chemotherapy",
    "radiotherapy",
    "immunotherapy",
    "palliative",
    "手術",
    "化学療法",
    "放射線治療",
    "免疫療法",
    "緩和ケア",
)

# Upper bound on the extracted term list. A long question can name many codes;
# the assessment renders a bounded number of them, so the list carried through
# state is bounded at the point it is built rather than at the point it is shown.
_MAX_EXTRACTED_TERMS: int = 32


def _classify_intent(text: str) -> str:
    """Classify the question's intent from keyword presence."""
    lower = text.lower()
    for intent, keywords in _INTENT_MAP.items():
        if any(kw.lower() in lower for kw in keywords):
            return intent
    return "general"


def _extract_disease_codes(text: str) -> List[str]:
    """Extract candidate disease-code strings from the question."""
    candidates = _DISEASE_CODE_RE.findall(text)
    # Common English words matching the same shape are not codes.
    stopwords = {"AND", "THE", "FOR", "NOT", "BUT", "NOR", "YET", "ICD"}
    return [c for c in candidates if c not in stopwords]


def _extract_medical_terms(text: str) -> List[str]:
    """Extract prognosis and treatment-protocol terms from the question."""
    lower = text.lower()
    found: List[str] = []
    for term in _PROGNOSIS_TERMS + _TREATMENT_TERMS:
        if term.lower() in lower:
            found.append(term)
    return found


class QueryParseNode(FunctionNode):
    """Parses the validated medical assessment question for the pipeline.

    Classifies the question's intent (disease-code lookup / prognosis / treatment
    protocol / oncology / disease inquiry / general) and extracts structured
    medical terms to guide domain routing and knowledge-base retrieval.

    Assigned to the inner MedicalAssessmentDomainGraph (step 1).
    Trust level ANONYMOUS: inner subgraph node; the external gate is PreProcessNode.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        validated_input = state.get("validated_input", state.get("user_input", ""))

        if not isinstance(validated_input, str) or not validated_input.strip():
            emit_trace_event(
                "ins_c2_051.query_parse.error",
                {"reason": "no_validated_input"},
                state,
            )
            return {
                "status": AgentStatus.ERROR,
                "error_log": ["QueryParseNode: no validated_input in state"],
            }

        query = validated_input.strip()

        intent = _classify_intent(query)
        disease_codes = _extract_disease_codes(query)
        medical_terms = _extract_medical_terms(query)

        all_terms = ([f"ICD-11:{c}" for c in disease_codes] + medical_terms)[:_MAX_EXTRACTED_TERMS]

        emit_trace_event(
            "ins_c2_051.query_parse.complete",
            {
                "intent": intent,
                "disease_code_count": len(disease_codes),
                "medical_term_count": len(medical_terms),
                "terms_carried": len(all_terms),
            },
            state,
        )

        return {
            "query_intent": intent,
            "extracted_medical_terms": to_json(all_terms),
            "status": AgentStatus.SUCCESS,
        }
