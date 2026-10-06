# PB: the output boundary — refusing is not the same as containing.
#
# The framework's envelope builder returns
#   {"output": state["formatted_output"] or state["result"], ...}
# and it does so on an error status too. So a gate that returns ERROR while
# leaving `result` in place ships the un-gated text it had just refused, inside
# the error envelope — and a gate that clears `formatted_output` to "" re-opens
# the very fallback the clearing was meant to close, because "" is falsy.
#
# These tests measure EVERY path that can return non-success, not only the one a
# static sweep would name, and they measure it at both levels: the node's own
# return value, and the envelope a caller actually receives.

import json

import pytest

import src.nodes.post_process_node as _post_mod
from framework.schemas.agent_status import AgentStatus
from src.graph.domain_workflow_graph import MedicalAssessmentDomainGraph
from src.nodes.output_gate_node import OutputGateNode
from src.nodes.post_process_node import WITHHELD_NOTICE, PostProcessNode

_DISCLAIMER = "[INSURANCE ASSESSMENT DISCLAIMER]"

_BASE_STATE = {
    "caller_trust_level": "ANONYMOUS",
    "correlation_id": "test-containment",
    "node_history": [],
    "error_log": [],
}

# A well-formed assessment, and the same document carrying something that must
# never be released. The marker text is what a leak would look like.
_CLEAN = (
    "## Medical Assessment — DISEASES Domain\n\n"
    "1. Chronic conditions are risk-stratified using severity bands.  "
    "_(source: kb/diseases, score: 0.87)_\n\n" + _DISCLAIMER
)
_LEAK_MARKER = "RELEASED-TEXT-CANARY"


@pytest.fixture(autouse=True)
def _noop_emit(monkeypatch):
    monkeypatch.setattr(_post_mod, "emit_trace_event", lambda *a, **k: None)


def _envelope(node_result: dict, prior_state: dict) -> dict:
    """The envelope a caller would receive, given a node result over prior state.

    Reproduces the framework's own projection — `formatted_output or result` —
    against the state as it stands AFTER the node's partial update is merged.
    That merge is what makes "the gate returned ERROR" and "the gate contained
    the text" two different claims.
    """
    merged = {**prior_state, **node_result}
    return {
        "output": merged.get("formatted_output") or merged.get("result"),
        "status": merged.get("status"),
    }


class TestEveryRefusalPathContains:
    """Each non-success path of the backbone gate, measured separately."""

    def setup_method(self):
        self.node = PostProcessNode()

    @pytest.mark.parametrize(
        "case, result_value",
        [
            ("disclaimer_absent", f"## Assessment\n\n{_LEAK_MARKER} released without the notice."),
            ("identifier_marker", f"{_DISCLAIMER}\npolicy_no: 99887766 {_LEAK_MARKER}"),
            ("credential_shape", f"{_DISCLAIMER}\nAKIAIOSFODNN7EXAMPLE {_LEAK_MARKER}"),
            ("credential_assignment", f"{_DISCLAIMER}\npassword=hunter2hunter2 {_LEAK_MARKER}"),
            ("oversize", _DISCLAIMER + "\n" + (_LEAK_MARKER + " ") * 3000),
        ],
    )
    def test_refused_text_is_cleared_from_the_envelope(self, case, result_value):
        state = {**_BASE_STATE, "result": result_value}
        node_result = self.node.execute(state)

        assert node_result["status"] == AgentStatus.ERROR, case
        # The one field the envelope falls back to is emptied ...
        assert node_result["result"] == "", case
        # ... and the field it prefers is non-empty, so the fallback stays closed.
        assert node_result["formatted_output"] == WITHHELD_NOTICE, case

        envelope = _envelope(node_result, state)
        assert envelope["output"] == WITHHELD_NOTICE, case
        assert _LEAK_MARKER not in envelope["output"], case
        assert "AKIA" not in envelope["output"], case

    def test_absent_result_is_also_a_refusal(self):
        state = dict(_BASE_STATE)
        node_result = self.node.execute(state)
        assert node_result["status"] == AgentStatus.ERROR
        assert _envelope(node_result, state)["output"] == WITHHELD_NOTICE

    def test_refusal_reason_is_a_closed_label(self):
        """The audit line says which rule refused, never the text that tripped it."""
        state = {**_BASE_STATE, "result": f"{_DISCLAIMER}\nAKIAIOSFODNN7EXAMPLE {_LEAK_MARKER}"}
        node_result = self.node.execute(state)
        joined = " ".join(node_result["error_log"])
        assert "credential_shape_present" in joined
        assert _LEAK_MARKER not in joined
        assert "AKIAIOSFODNN7EXAMPLE" not in joined

    def test_clean_assessment_is_released_unchanged(self):
        state = {**_BASE_STATE, "result": _CLEAN}
        node_result = self.node.execute(state)
        assert node_result["status"] == AgentStatus.SUCCESS
        assert node_result["formatted_output"] == _CLEAN
        assert _envelope(node_result, state)["output"] == _CLEAN

    def test_the_withheld_notice_is_truthy(self):
        """A falsy replacement re-opens the fallback; this is the property that
        makes the clearing work at all."""
        assert bool(WITHHELD_NOTICE)
        assert _DISCLAIMER not in WITHHELD_NOTICE


class TestInnerGateWithholds:
    """The inner graph publishes nothing on a run its own gate refused.

    Unit-level by construction: with `error_strategy = "propagate"` the outer
    GraphNode raises on an inner error and never calls merge_output, so the inner
    projection is not reachable from /invoke today. It is asserted here because
    the projection is what a later change — a different error strategy, or a
    second consumer of this subgraph — would rely on.
    """

    def setup_method(self):
        self.node = OutputGateNode()
        self.graph = MedicalAssessmentDomainGraph()

    def test_missing_disclaimer_publishes_nothing(self):
        state = {
            **_BASE_STATE,
            "underwriting_assessment": f"## Assessment\n\n{_LEAK_MARKER}",
            "hokengyoho_audit_log": json.dumps({"event": "test"}),
        }
        node_result = self.node.execute(state)
        assert node_result["status"] == AgentStatus.ERROR
        assert "formatted_output" not in node_result

        merged = {**state, **node_result}
        assert self.graph.get_output(merged)["output"] is None

    def test_upstream_failure_is_not_erased_by_reaching_the_end(self):
        """The inner topology has no conditional edge, so every node runs even
        after one fails and a later SUCCESS would overwrite the earlier ERROR.
        The last node's status is the one the subgraph reports."""
        state = {
            **_BASE_STATE,
            "status": AgentStatus.ERROR.value,
            "underwriting_assessment": f"{_DISCLAIMER}\n{_LEAK_MARKER}",
            "hokengyoho_audit_log": json.dumps({"event": "test"}),
        }
        node_result = self.node.execute(state)
        assert node_result["status"] == AgentStatus.ERROR
        assert "formatted_output" not in node_result
        assert self.graph.get_output({**state, **node_result})["output"] is None

    def test_accepted_assessment_is_published(self):
        state = {
            **_BASE_STATE,
            "underwriting_assessment": _CLEAN,
            "hokengyoho_audit_log": json.dumps({"event": "test"}),
        }
        node_result = self.node.execute(state)
        assert node_result["status"] == AgentStatus.SUCCESS
        assert self.graph.get_output({**state, **node_result})["output"] == _CLEAN
