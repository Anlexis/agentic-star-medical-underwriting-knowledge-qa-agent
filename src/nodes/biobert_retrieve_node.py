"""AgentCore Platform v1.0 — INS-C2-051 BioBERTRetrieveNode (inner domain graph, step 3)."""

# Retrieves biomedical knowledge-base passages for the routed domain.
#
# Retrieval runs entirely on device: no patient data leaves the execution
# boundary, which is what makes the pipeline usable on 要配慮個人情報 under APPI.
#
# The bundled knowledge base below is the shipped source of truth. A production
# deployment swaps it for a real embedding model and vector index behind
# src/services/service.py without changing this node's contract.
#
# Retrieval DEPTH is caller-controlled through the `max_passages` context
# parameter, bounded to [1, 5] by PreProcessNode. The bound is re-read here from
# the same constants, so the depth the retriever uses is the depth the contract
# published — and a caller asking for more passages measurably gets more, which
# is the point of declaring the parameter at all.
#
# Node contract:
#   - Extend FunctionNode; implement execute(state) -> dict (partial state update)
#   - Return ONLY the fields this node changes (never the full state)
#   - required_trust_level = ANONYMOUS (inner subgraph node)
#   - No network calls; on-device knowledge base only

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import to_json
from src.services.service import finite_int_in_range

# ── Bundled knowledge base, keyed by domain ──────────────────────────────────
# Passages are ordered by descending relevance score within each domain, so a
# deeper request adds lower-ranked material rather than reordering what a
# shallower request already returned.
_KB_PASSAGES: Dict[str, List[Dict[str, Any]]] = {
    "diseases": [
        {
            "text": "Disease classification codes provide a standardized vocabulary for "
            "insurance risk assessment and underwriting decisions.",
            "score": 0.91,
            "source": "kb/diseases",
        },
        {
            "text": "Chronic conditions are risk-stratified using severity bands for life "
            "and health insurance premium calculation.",
            "score": 0.87,
            "source": "kb/diseases",
        },
        {
            "text": "Comorbidity count and duration since diagnosis are the two variables "
            "that move a chronic-condition rating most in practice.",
            "score": 0.83,
            "source": "kb/diseases",
        },
        {
            "text": "Stability of a documented condition over a multi-year observation "
            "window supports a standard rating where a recent diagnosis would not.",
            "score": 0.78,
            "source": "kb/diseases",
        },
        {
            "text": "Coded diagnoses without supporting clinical detail are treated as "
            "incomplete evidence and referred for medical officer review.",
            "score": 0.71,
            "source": "kb/diseases",
        },
    ],
    "oncology": [
        {
            "text": "Oncology risk assessment incorporates tumour stage, histological grade " "and treatment response.",
            "score": 0.93,
            "source": "kb/oncology",
        },
        {
            "text": "Cancer survival statistics by site and stage inform life insurance risk "
            "classification and loading factors.",
            "score": 0.89,
            "source": "kb/oncology",
        },
        {
            "text": "Time elapsed since the end of active treatment is the dominant variable "
            "in post-treatment mortality models.",
            "score": 0.85,
            "source": "kb/oncology",
        },
        {
            "text": "Recurrence within the first observation window shifts an application "
            "from a loaded rating to a deferral in most published guidance.",
            "score": 0.80,
            "source": "kb/oncology",
        },
        {
            "text": "Adjuvant therapy completion status distinguishes an in-treatment "
            "applicant from a post-treatment one for rating purposes.",
            "score": 0.74,
            "source": "kb/oncology",
        },
    ],
    "chemicals": [
        {
            "text": "Drug interaction profiles and contraindication data inform health "
            "insurance exclusion clause assessment.",
            "score": 0.84,
            "source": "kb/chemicals",
        },
        {
            "text": "Sustained pharmacological control of a chronic condition is evidence of "
            "management, not of severity, and is rated accordingly.",
            "score": 0.79,
            "source": "kb/chemicals",
        },
        {
            "text": "Recent changes to a medication regimen indicate an unstable condition "
            "and are commonly grounds for postponement.",
            "score": 0.73,
            "source": "kb/chemicals",
        },
    ],
    "anatomy": [
        {
            "text": "Anatomical region involvement determines disability classification and "
            "benefit eligibility under policy terms.",
            "score": 0.82,
            "source": "kb/anatomy",
        },
        {
            "text": "Functional impairment measured against activities of daily living is a "
            "better rating signal than the anatomical finding alone.",
            "score": 0.76,
            "source": "kb/anatomy",
        },
    ],
    "genes": [
        {
            "text": "Predictive genetic information is subject to explicit use restrictions "
            "in underwriting; jurisdiction-specific rules apply.",
            "score": 0.81,
            "source": "kb/genes",
        },
        {
            "text": "A familial risk marker without a clinical diagnosis does not by itself "
            "support a loading in most published guidance.",
            "score": 0.75,
            "source": "kb/genes",
        },
    ],
    "phenotypes": [
        {
            "text": "Observable clinical characteristics are combined with reported history "
            "to distinguish an active condition from a resolved one.",
            "score": 0.77,
            "source": "kb/phenotypes",
        },
    ],
    "trials": [
        {
            "text": "Participation in an interventional study is recorded as an open clinical "
            "course and generally defers a final assessment.",
            "score": 0.76,
            "source": "kb/trials",
        },
    ],
    "general": [
        {
            "text": "Medical assessment for insurance underwriting requires integration of "
            "diagnosis, prognosis and treatment evidence.",
            "score": 0.79,
            "source": "kb/general",
        },
        {
            "text": "Where evidence is incomplete, the documented practice is to request "
            "further medical evidence rather than to assume a rating.",
            "score": 0.72,
            "source": "kb/general",
        },
        {
            "text": "An assessment summary is underwriting reference material and is never "
            "a substitute for a licensed medical opinion.",
            "score": 0.68,
            "source": "kb/general",
        },
    ],
}

# Retrieval depth: the default, and the bounds the caller may move it between.
# Imported by PreProcessNode's validator so contract and retriever cannot drift.
DEFAULT_TOP_K: int = 2
TOP_K_MIN: int = 1
TOP_K_MAX: int = 5


class BioBERTRetrieveNode(FunctionNode):
    """Retrieves biomedical knowledge-base passages for the underwriting pipeline.

    Reads the domain selected by DomainRouteNode and the caller's declared
    retrieval depth, and returns the top-ranked passages serialized as JSON for
    downstream formatting.

    A declared depth outside the published bounds is not silently clamped: the
    request never reaches this node, because PreProcessNode refuses it. What is
    re-checked here is the value actually held in state, so a depth that somehow
    arrived unvalidated degrades to the documented default rather than indexing
    the knowledge base with it.

    Assigned to the inner MedicalAssessmentDomainGraph (step 3).
    Trust level ANONYMOUS: inner subgraph node; S-1 gate is on the outer PreProcessNode.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        domain = state.get("domain_routed", "general")
        validated_input = state.get("validated_input", state.get("user_input", ""))
        context = state.get("input_context", {}) or {}

        if not validated_input:
            emit_trace_event(
                "ins_c2_051.biobert_retrieve.error",
                {"reason": "no_validated_input"},
                state,
            )
            return {
                "status": AgentStatus.ERROR,
                "error_log": ["BioBERTRetrieveNode: no validated_input in state"],
            }

        declared = context.get("max_passages") if isinstance(context, dict) else None
        top_k = finite_int_in_range(declared, TOP_K_MIN, TOP_K_MAX)
        if top_k is None:
            top_k = DEFAULT_TOP_K

        domain_passages = _KB_PASSAGES.get(domain, _KB_PASSAGES["general"])
        top_passages = domain_passages[:top_k]

        if not top_passages:
            emit_trace_event(
                "ins_c2_051.biobert_retrieve.no_passages",
                {"domain": domain},
                state,
            )
            return {
                "retrieved_passages": to_json([]),
                "status": AgentStatus.SUCCESS,
            }

        emit_trace_event(
            "ins_c2_051.biobert_retrieve.complete",
            {
                "domain": domain,
                "requested_depth": top_k,
                "passage_count": len(top_passages),
                "top_score": top_passages[0]["score"],
            },
            state,
        )

        return {
            "retrieved_passages": to_json(top_passages),
            "status": AgentStatus.SUCCESS,
        }
