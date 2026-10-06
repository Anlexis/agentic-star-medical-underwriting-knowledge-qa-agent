# INS-C2-051 — Unit Tests: Domain Nodes
#
# Covers:
#   PreProcessNode   (S-1 external gate — VERIFIED_EXTERNAL)
#   QueryParseNode   (intent classification + ICD-11 extraction — inner, ANONYMOUS)
#   DomainRouteNode  (openmed domain routing — inner, ANONYMOUS)
#   BioBERTRetrieveNode    (sovereign on-device KB retrieval — inner, ANONYMOUS)
#   UnderwritingAssessmentFormatNode  (assessment formatting + Hokengyoho audit — inner)
#   OutputGateNode   (inner S-3 disclaimer gate + S-5 audit finalization — inner)
#   PostProcessNode  (outer S-3 gate — backbone, ANONYMOUS)
#
# Patch rule: emit_trace_event is patched AT THE NODE MODULE (no sys.modules stub).
# The real SDK ships shared.utils.audit_logger; patching sys.modules["shared"]
# breaks the framework's own `from shared.security...` at load time.
# Module-level monkeypatching is the correct pattern.

import json
import pytest

import src.nodes.pre_process_node as _pre_mod
import src.nodes.query_parse_node as _qparse_mod
import src.nodes.domain_route_node as _route_mod
import src.nodes.biobert_retrieve_node as _retrieve_mod
import src.nodes.underwriting_assessment_format_node as _format_mod
import src.nodes.output_gate_node as _gate_mod
import src.nodes.post_process_node as _post_mod

from framework.schemas.agent_status import AgentStatus
from src.schemas.state import to_json

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_emit(monkeypatch):
    """Patch emit_trace_event in every domain node module (no sys.modules stub)."""

    def noop(*a, **k):
        return None

    for mod in (
        _pre_mod,
        _qparse_mod,
        _route_mod,
        _retrieve_mod,
        _format_mod,
        _gate_mod,
        _post_mod,
    ):
        monkeypatch.setattr(mod, "emit_trace_event", noop)


_BASE_STATE = {
    "caller_trust_level": "VERIFIED_EXTERNAL",
    "correlation_id": "test-unit-ins-c2-051",
    "node_history": [],
    "error_log": [],
}

_VALID_MEDICAL_QUERY = (
    "Insurance underwriting assessment: applicant has Type 2 diabetes mellitus "
    "with favorable prognosis. Current treatment includes oral medication. "
    "Please provide biomedical knowledge for risk classification."
)


# ---------------------------------------------------------------------------
# PreProcessNode — S-1 gate
# ---------------------------------------------------------------------------


class TestPreProcessNode:
    """S-1 external gate: validates and sanitizes medical assessment queries."""

    def setup_method(self):
        from src.nodes.pre_process_node import PreProcessNode

        self.node = PreProcessNode()

    def _state(self, user_input: str, **ctx_fields):
        return {
            **_BASE_STATE,
            "user_input": user_input,
            "input_context": ctx_fields,
        }

    def test_valid_medical_query_passes(self):
        """S-1 gate: well-formed medical assessment query yields SUCCESS."""
        result = self.node.execute(self._state(_VALID_MEDICAL_QUERY))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["validated_input"] == _VALID_MEDICAL_QUERY.strip()

    def test_empty_input_rejected(self):
        """S-1 gate: empty user_input returns ERROR."""
        result = self.node.execute(self._state(""))
        assert result["status"] == AgentStatus.ERROR
        assert any("empty" in e.lower() or "missing" in e.lower() for e in result["error_log"])

    def test_whitespace_only_rejected(self):
        """S-1 gate: whitespace-only input is treated as empty."""
        result = self.node.execute(self._state("   \n\t  "))
        assert result["status"] == AgentStatus.ERROR

    def test_oversized_input_rejected(self):
        """Input exceeding 4,000 chars is refused, and nothing is carried forward.

        Asserted behaviourally: the run is an error, the question is not promoted
        to validated_input, and the refusal names the field rather than quoting
        the value back.
        """
        huge = "X" * 4_001
        result = self.node.execute(self._state(huge))
        assert result["status"] == AgentStatus.ERROR
        assert "validated_input" not in result
        assert any("'input'" in e for e in result["error_log"])
        assert not any(huge in e for e in result["error_log"])

    def test_policy_number_pii_rejected(self):
        """S-1 gate: query containing 10-digit policy number is rejected (APPI 要配慮個人情報)."""
        query_with_policy = "Policy number 1234567890 has diabetes history."
        result = self.node.execute(self._state(query_with_policy))
        assert result["status"] == AgentStatus.ERROR
        assert any("pii" in e.lower() or "identifier" in e.lower() for e in result["error_log"])

    def test_patient_id_marker_rejected(self):
        """S-1 gate: explicit patient_id: marker triggers PII rejection."""
        query_with_marker = "patient_id: P12345 has hypertension."
        result = self.node.execute(self._state(query_with_marker))
        assert result["status"] == AgentStatus.ERROR

    def test_kojin_bango_marker_rejected(self):
        """S-1 gate: 個人番号 (My Number) marker triggers PII rejection."""
        query_with_jp = "この患者の個人番号を確認してください。"
        result = self.node.execute(self._state(query_with_jp))
        assert result["status"] == AgentStatus.ERROR

    def test_validated_input_set_on_success(self):
        """S-1 gate: SUCCESS path sets validated_input in returned dict."""
        result = self.node.execute(self._state(_VALID_MEDICAL_QUERY, channel="api"))
        assert result["status"] == AgentStatus.SUCCESS
        assert "validated_input" in result
        assert result["validated_input"] is not None

    def test_trust_level_is_verified_external(self):
        """PreProcessNode must declare VERIFIED_EXTERNAL trust (backbone S-1 gate)."""
        from src.nodes.pre_process_node import PreProcessNode
        from framework.schemas.trust_level import TrustLevel

        assert PreProcessNode.required_trust_level == TrustLevel.VERIFIED_EXTERNAL

    def test_execute_method_signature(self):
        """Node contract: node must implement execute(state) not _invoke_impl."""
        import inspect
        from src.nodes.pre_process_node import PreProcessNode

        assert hasattr(PreProcessNode, "execute")
        sig = inspect.signature(PreProcessNode.execute)
        params = list(sig.parameters.keys())
        assert "state" in params
        assert "_invoke_impl" not in PreProcessNode.__dict__


# ---------------------------------------------------------------------------
# QueryParseNode — intent classification + ICD-11 extraction
# ---------------------------------------------------------------------------


class TestQueryParseNode:
    """Inner domain node: parses medical query intent and extracts medical terms."""

    def setup_method(self):
        from src.nodes.query_parse_node import QueryParseNode

        self.node = QueryParseNode()

    def _state(self, validated_input: str):
        return {**_BASE_STATE, "caller_trust_level": "ANONYMOUS", "validated_input": validated_input}

    def test_disease_intent_classified(self):
        """Intent classification: disease-related query returns disease_inquiry intent."""
        result = self.node.execute(self._state("What disease does this condition represent?"))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["query_intent"] == "disease_inquiry"

    def test_prognosis_intent_classified(self):
        """Intent classification: prognosis query returns prognosis intent."""
        result = self.node.execute(self._state("What is the prognosis for this condition?"))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["query_intent"] == "prognosis"

    def test_oncology_intent_classified(self):
        """Intent classification: cancer query returns oncology intent."""
        # Query must not contain "treatment" or "protocol" — those fire treatment_protocol first
        result = self.node.execute(self._state("Is the cancer diagnosis malignant or benign?"))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["query_intent"] == "oncology"

    def test_general_intent_for_unknown(self):
        """Intent classification: unrecognized query returns general intent."""
        result = self.node.execute(self._state("Hello there"))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["query_intent"] == "general"

    def test_medical_terms_extracted_as_json(self):
        """Medical terms output is a valid JSON string list."""
        result = self.node.execute(self._state("favorable prognosis with surgery treatment"))
        assert result["status"] == AgentStatus.SUCCESS
        terms_json = result.get("extracted_medical_terms")
        assert terms_json is not None
        terms = json.loads(terms_json)
        assert isinstance(terms, list)

    def test_empty_input_returns_error(self):
        """QueryParseNode: missing validated_input returns ERROR."""
        result = self.node.execute({**_BASE_STATE, "caller_trust_level": "ANONYMOUS"})
        assert result["status"] == AgentStatus.ERROR

    def test_trust_level_is_anonymous(self):
        """Inner node must declare ANONYMOUS trust level."""
        from src.nodes.query_parse_node import QueryParseNode
        from framework.schemas.trust_level import TrustLevel

        assert QueryParseNode.required_trust_level == TrustLevel.ANONYMOUS


# ---------------------------------------------------------------------------
# DomainRouteNode — openmed domain routing
# ---------------------------------------------------------------------------


class TestDomainRouteNode:
    """Inner domain node: routes query to appropriate openmed biomedical domain."""

    def setup_method(self):
        from src.nodes.domain_route_node import DomainRouteNode

        self.node = DomainRouteNode()

    def _state(self, intent: str, validated_input: str = _VALID_MEDICAL_QUERY):
        return {
            **_BASE_STATE,
            "caller_trust_level": "ANONYMOUS",
            "query_intent": intent,
            "validated_input": validated_input,
        }

    def test_icd11_lookup_routes_to_diseases(self):
        """ICD-11 lookup intent routes to the 'diseases' openmed domain."""
        result = self.node.execute(self._state("icd11_lookup"))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["domain_routed"] == "diseases"

    def test_prognosis_routes_to_diseases(self):
        """Prognosis intent routes to the 'diseases' domain."""
        result = self.node.execute(self._state("prognosis"))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["domain_routed"] == "diseases"

    def test_oncology_routes_to_oncology(self):
        """Oncology intent routes to the 'oncology' domain."""
        result = self.node.execute(self._state("oncology"))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["domain_routed"] == "oncology"

    def test_treatment_routes_to_chemicals(self):
        """Treatment protocol intent routes to the 'chemicals' domain."""
        result = self.node.execute(self._state("treatment_protocol"))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["domain_routed"] == "chemicals"

    def test_general_routes_to_general(self):
        """General intent routes to the 'general' domain."""
        result = self.node.execute(self._state("general"))
        assert result["status"] == AgentStatus.SUCCESS
        assert result["domain_routed"] == "general"

    def test_missing_input_returns_error(self):
        """DomainRouteNode: missing validated_input returns ERROR."""
        result = self.node.execute(
            {
                **_BASE_STATE,
                "caller_trust_level": "ANONYMOUS",
                "query_intent": "general",
            }
        )
        assert result["status"] == AgentStatus.ERROR

    def test_trust_level_is_anonymous(self):
        """Inner node must declare ANONYMOUS trust level."""
        from src.nodes.domain_route_node import DomainRouteNode
        from framework.schemas.trust_level import TrustLevel

        assert DomainRouteNode.required_trust_level == TrustLevel.ANONYMOUS


# ---------------------------------------------------------------------------
# BioBERTRetrieveNode — KB retrieval
# ---------------------------------------------------------------------------


class TestBioBERTRetrieveNode:
    """Inner domain node: sovereign on-device biomedical KB retrieval."""

    def setup_method(self):
        from src.nodes.biobert_retrieve_node import BioBERTRetrieveNode

        self.node = BioBERTRetrieveNode()

    def _state(self, domain: str = "diseases", validated_input: str = _VALID_MEDICAL_QUERY):
        return {
            **_BASE_STATE,
            "caller_trust_level": "ANONYMOUS",
            "domain_routed": domain,
            "validated_input": validated_input,
        }

    def test_diseases_domain_returns_passages(self):
        """Diseases domain: KB stub returns relevant passages as JSON list."""
        result = self.node.execute(self._state("diseases"))
        assert result["status"] == AgentStatus.SUCCESS
        passages = json.loads(result["retrieved_passages"])
        assert isinstance(passages, list)
        assert len(passages) > 0
        assert "text" in passages[0]
        assert "score" in passages[0]

    def test_oncology_domain_returns_passages(self):
        """Oncology domain: KB stub returns oncology-relevant passages."""
        result = self.node.execute(self._state("oncology"))
        assert result["status"] == AgentStatus.SUCCESS
        passages = json.loads(result["retrieved_passages"])
        assert len(passages) > 0

    def test_general_domain_returns_passages(self):
        """General domain: KB stub returns general biomedical passages."""
        result = self.node.execute(self._state("general"))
        assert result["status"] == AgentStatus.SUCCESS
        passages = json.loads(result["retrieved_passages"])
        assert isinstance(passages, list)

    def test_missing_input_returns_error(self):
        """BioBERTRetrieveNode: missing validated_input returns ERROR."""
        result = self.node.execute(
            {
                **_BASE_STATE,
                "caller_trust_level": "ANONYMOUS",
                "domain_routed": "diseases",
            }
        )
        assert result["status"] == AgentStatus.ERROR

    def test_passages_are_json_serializable(self):
        """retrieved_passages must be a valid JSON string (ADR-005 state safety)."""
        result = self.node.execute(self._state("diseases"))
        assert result["status"] == AgentStatus.SUCCESS
        passages_str = result["retrieved_passages"]
        assert isinstance(passages_str, str)
        passages = json.loads(passages_str)
        assert isinstance(passages, list)

    def test_trust_level_is_anonymous(self):
        """Inner node must declare ANONYMOUS trust level."""
        from src.nodes.biobert_retrieve_node import BioBERTRetrieveNode
        from framework.schemas.trust_level import TrustLevel

        assert BioBERTRetrieveNode.required_trust_level == TrustLevel.ANONYMOUS


# ---------------------------------------------------------------------------
# UnderwritingAssessmentFormatNode — assessment formatting
# ---------------------------------------------------------------------------


class TestUnderwritingAssessmentFormatNode:
    """Inner domain node: formats biomedical response for underwriting context."""

    def setup_method(self):
        from src.nodes.underwriting_assessment_format_node import UnderwritingAssessmentFormatNode

        self.node = UnderwritingAssessmentFormatNode()

    def _state(self, passages: list | None = None, intent: str = "disease_inquiry", domain: str = "diseases"):
        if passages is None:
            passages = [
                {
                    "text": "ICD-11 disease classification supports risk assessment.",
                    "score": 0.91,
                    "source": "openmed/diseases",
                }
            ]
        return {
            **_BASE_STATE,
            "caller_trust_level": "ANONYMOUS",
            "validated_input": _VALID_MEDICAL_QUERY,
            "query_intent": intent,
            "domain_routed": domain,
            "retrieved_passages": to_json(passages),
        }

    def test_assessment_contains_mandatory_disclaimer(self):
        """Formatted assessment must contain mandatory insurance disclaimer (S-3 requirement)."""
        result = self.node.execute(self._state())
        assert result["status"] == AgentStatus.SUCCESS
        assessment = result["underwriting_assessment"]
        assert (
            "[INSURANCE ASSESSMENT DISCLAIMER]" in assessment
        ), "Mandatory insurance assessment disclaimer missing from formatted output"

    def test_audit_log_generated(self):
        """Hokengyoho audit log entry must be a valid JSON string."""
        result = self.node.execute(self._state())
        assert result["status"] == AgentStatus.SUCCESS
        audit_str = result["hokengyoho_audit_log"]
        assert audit_str is not None
        audit = json.loads(audit_str)
        assert isinstance(audit, dict)
        assert "event" in audit

    def test_domain_appears_in_assessment(self):
        """Assessment header references the selected openmed domain."""
        result = self.node.execute(self._state(domain="oncology"))
        assert result["status"] == AgentStatus.SUCCESS
        assert "ONCOLOGY" in result["underwriting_assessment"].upper()

    def test_empty_passages_yields_success(self):
        """Empty passage list: assessment still completes (with no-passages notice)."""
        result = self.node.execute(self._state(passages=[]))
        assert result["status"] == AgentStatus.SUCCESS
        assert "[INSURANCE ASSESSMENT DISCLAIMER]" in result["underwriting_assessment"]

    def test_missing_input_returns_error(self):
        """UnderwritingAssessmentFormatNode: missing validated_input returns ERROR."""
        result = self.node.execute(
            {
                **_BASE_STATE,
                "caller_trust_level": "ANONYMOUS",
                "query_intent": "disease_inquiry",
            }
        )
        assert result["status"] == AgentStatus.ERROR

    def test_trust_level_is_anonymous(self):
        """Inner node must declare ANONYMOUS trust level."""
        from src.nodes.underwriting_assessment_format_node import UnderwritingAssessmentFormatNode
        from framework.schemas.trust_level import TrustLevel

        assert UnderwritingAssessmentFormatNode.required_trust_level == TrustLevel.ANONYMOUS


# ---------------------------------------------------------------------------
# OutputGateNode — inner S-3 gate
# ---------------------------------------------------------------------------


class TestOutputGateNode:
    """Inner S-3 gate: verifies mandatory disclaimer presence and finalizes audit log."""

    def setup_method(self):
        from src.nodes.output_gate_node import OutputGateNode

        self.node = OutputGateNode()

    def _state(self, assessment: str | None = None, audit_log: dict | None = None):
        default_assessment = (
            "## Medical Assessment — DISEASES Domain\n\n"
            "**Query Intent:** disease_inquiry\n\n"
            "[INSURANCE ASSESSMENT DISCLAIMER] This biomedical knowledge summary is "
            "generated for insurance underwriting reference only."
        )
        return {
            **_BASE_STATE,
            "caller_trust_level": "ANONYMOUS",
            "underwriting_assessment": assessment if assessment is not None else default_assessment,
            "hokengyoho_audit_log": to_json(audit_log or {"event": "test"}),
        }

    def test_valid_assessment_passes(self):
        """S-3 gate: assessment with disclaimer yields SUCCESS and sets disclaimer_applied."""
        result = self.node.execute(self._state())
        assert result["status"] == AgentStatus.SUCCESS
        assert result["disclaimer_applied"] is True
        assert result["formatted_output"] is not None

    def test_missing_disclaimer_triggers_error(self):
        """S-3 gate: assessment without disclaimer returns ERROR."""
        result = self.node.execute(
            self._state(assessment="## Assessment\n\nSome content without the required disclaimer.")
        )
        assert result["status"] == AgentStatus.ERROR
        assert any("disclaimer" in e.lower() or "s-3" in e.lower() for e in result["error_log"])

    def test_missing_assessment_returns_error(self):
        """S-3 gate: no underwriting_assessment in state returns ERROR."""
        result = self.node.execute({**_BASE_STATE, "caller_trust_level": "ANONYMOUS"})
        assert result["status"] == AgentStatus.ERROR

    def test_audit_log_finalized(self):
        """S-5: Hokengyoho audit log is updated with gate verdict."""
        result = self.node.execute(self._state())
        assert result["status"] == AgentStatus.SUCCESS
        audit = json.loads(result["hokengyoho_audit_log"])
        assert audit.get("output_gate_passed") is True
        assert audit.get("disclaimer_verified") is True

    def test_trust_level_is_anonymous(self):
        """Inner node must declare ANONYMOUS trust level."""
        from src.nodes.output_gate_node import OutputGateNode
        from framework.schemas.trust_level import TrustLevel

        assert OutputGateNode.required_trust_level == TrustLevel.ANONYMOUS


# ---------------------------------------------------------------------------
# PostProcessNode — outer S-3 backbone gate
# ---------------------------------------------------------------------------


class TestPostProcessNode:
    """Outer backbone S-3 gate: final output security verification before emission."""

    def setup_method(self):
        from src.nodes.post_process_node import PostProcessNode

        self.node = PostProcessNode()

    def _valid_result(self) -> str:
        return (
            "## Medical Assessment — DISEASES Domain\n\n"
            "1. ICD-11 disease classification provides standardized codes for insurance. "
            "_(source: openmed/diseases, score: 0.91)_\n\n"
            "[INSURANCE ASSESSMENT DISCLAIMER] This biomedical knowledge summary is "
            "generated for insurance underwriting reference only. It does not constitute "
            "medical advice. All underwriting decisions must be reviewed by a licensed underwriter."
        )

    def _state(self, result: str | None = None):
        return {
            **_BASE_STATE,
            "caller_trust_level": "ANONYMOUS",
            "result": result if result is not None else self._valid_result(),
        }

    def test_valid_assessment_passes_gate(self):
        """S-3 gate: assessment with disclaimer and within size limit passes."""
        result = self.node.execute(self._state())
        assert result["status"] == AgentStatus.SUCCESS
        assert result["formatted_output"] is not None
        assert "[INSURANCE ASSESSMENT DISCLAIMER]" in result["formatted_output"]

    def test_missing_result_returns_error(self):
        """S-3 gate: missing result in state returns ERROR."""
        # Pass state with no "result" key at all (not via _state() helper which defaults to valid)
        state = {**_BASE_STATE, "caller_trust_level": "ANONYMOUS"}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.ERROR

    def test_disclaimer_absent_rejected(self):
        """S-3 gate: output without mandatory disclaimer is rejected."""
        result = self.node.execute(self._state("Some biomedical content without the required disclaimer."))
        assert result["status"] == AgentStatus.ERROR
        assert any("disclaimer" in e.lower() or "s-3" in e.lower() for e in result["error_log"])

    def test_oversized_output_rejected(self):
        """S-3 gate: output > 50KB is rejected (bulk data leakage guard)."""
        huge = "[INSURANCE ASSESSMENT DISCLAIMER]\n" + "data " * 12_000
        result = self.node.execute(self._state(huge))
        assert result["status"] == AgentStatus.ERROR

    def test_trust_level_is_anonymous(self):
        """PostProcessNode must declare ANONYMOUS trust (gate trust lives on pre_process)."""
        from src.nodes.post_process_node import PostProcessNode
        from framework.schemas.trust_level import TrustLevel

        assert PostProcessNode.required_trust_level == TrustLevel.ANONYMOUS

    def test_execute_method_signature(self):
        """Node contract: node must implement execute(state) not _invoke_impl."""
        import inspect
        from src.nodes.post_process_node import PostProcessNode

        assert hasattr(PostProcessNode, "execute")
        sig = inspect.signature(PostProcessNode.execute)
        params = list(sig.parameters.keys())
        assert "state" in params
        assert "_invoke_impl" not in PostProcessNode.__dict__
