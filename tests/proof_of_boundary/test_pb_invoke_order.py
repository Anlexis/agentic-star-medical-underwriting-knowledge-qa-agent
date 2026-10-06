# PB-6: Invoke Execution Order Verification — INS-C2-051
#
# Two complementary tests:
#
# TestInvokeOrder — per-node __call__() gate order (S-1 → node_start → S-2 → execute →
#     S-3 → node_complete) for every concrete BaseNode subclass under src/nodes/.
#
# TestBackboneInvokeOrder — full Graph().invoke() backbone order assertion:
#     Verifies the five-slot backbone (initialize → pre_process → main →
#     post_process → finalize) executes in correct order and returns SUCCESS
#     for a well-formed medical assessment query.
#
# ── Template-specific constants (fill per template) ───────────────────────────
# _MAIN_SLOT_NODE : GraphNode subclass registered in the `main` backbone slot.
# _VALID_PAYLOAD  : user_input string that exercises the full domain pipeline
#                   and returns AgentStatus.SUCCESS (S-1 + S-3 gates must both pass).
#
# Patch rule: emit_trace_event is patched AT THE NODE MODULE — never via
#   sys.modules["shared"] (that would shadow the framework's own shared imports
#   and cause ModuleNotFoundError: 'shared' is not a package).

import importlib
import inspect
import pkgutil

import pytest

# ── Template-specific fills ───────────────────────────────────────────────────

from src.graph.graph import MedicalAssessmentWorkflowGraphNode

_MAIN_SLOT_NODE = MedicalAssessmentWorkflowGraphNode
"""GraphNode subclass assigned to the `main` backbone slot in InsuranceMedicalAssessmentAgent."""

_VALID_PAYLOAD = (
    "Insurance underwriting assessment request: "
    "The applicant has a documented history of Type 2 diabetes mellitus with code BA80. "
    "Current treatment protocol includes oral antidiabetic medication and dietary management. "
    "Prognosis is favorable based on stable glycemic control over the past three years. "
    "Please provide biomedical knowledge summary for risk classification and underwriting decision."
)
"""SUCCESS-yielding medical assessment query.

Passes PreProcessNode S-1 gate:
  - No PII patient identifiers (no policy numbers, My Number, patient IDs)
  - Well within 4,000 char limit
  - No injection patterns

Passes PostProcessNode S-3 gate:
  - UnderwritingAssessmentFormatNode always injects [INSURANCE ASSESSMENT DISCLAIMER]
  - Output size is well within 50KB limit
"""

# ─────────────────────────────────────────────────────────────────────────────
# PB-6a: per-node __call__() invoke gate order
# ─────────────────────────────────────────────────────────────────────────────


def _discover_node_classes() -> list[type]:
    """Import every module under src/nodes/ and collect concrete BaseNode subclasses."""
    from framework.nodes.base_node import BaseNode

    try:
        pkg = importlib.import_module("src.nodes")
    except ImportError:
        return []

    discovered = []
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, prefix="src.nodes."):
        module = importlib.import_module(modname)
        for attr in vars(module).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseNode)
                and attr is not BaseNode
                and attr.__module__ == modname
                and not inspect.isabstract(attr)
            ):
                discovered.append(attr)
    return discovered


class TestInvokeOrder:
    """PB-6a: __call__ must run S-1 → node_start → S-2 → execute() → S-3 → node_complete."""

    def test_call_order_for_every_node(self, monkeypatch):
        node_classes = _discover_node_classes()
        if not node_classes:
            pytest.skip("no concrete BaseNode subclasses found under src/nodes/")

        import framework.nodes.base_node as base_node_module

        failures: list[str] = []
        for node_cls in node_classes:
            order: list[str] = []
            monkeypatch.setattr(
                base_node_module,
                "emit_trace_event",
                lambda event_type, _payload, _state, _o=order: _o.append(f"event:{event_type}"),
            )

            for method_name, label in (
                ("_security_gate_input", "security_gate_input"),
                ("execute", "execute"),
                ("_security_gate_output", "security_gate_output"),
            ):
                original = getattr(node_cls, method_name)

                def spy(self, arg, _o=order, _label=label, _orig=original):
                    _o.append(_label)
                    return _orig(self, arg)

                monkeypatch.setattr(node_cls, method_name, spy)

            instance = node_cls()
            state = {
                "caller_trust_level": node_cls.required_trust_level.value,
                "correlation_id": "pb6-invoke-order-test",
            }
            instance(state)

            expected = [
                "event:node_start",
                "security_gate_input",
                "execute",
                "security_gate_output",
                "event:node_complete",
            ]
            if order != expected:
                failures.append(
                    f"{node_cls.__name__}: invoke order violation.\n" f"expected: {expected}\nactual:   {order}"
                )

        assert not failures, "\n\n".join(failures)


# ─────────────────────────────────────────────────────────────────────────────
# PB-6b: full backbone invoke order (Graph().invoke() end-to-end)
# ─────────────────────────────────────────────────────────────────────────────


class TestBackboneInvokeOrder:
    """PB-6b: Graph().invoke() with _VALID_PAYLOAD must traverse all 5 backbone slots
    in order (initialize → pre_process → main → post_process → finalize) and
    return AgentStatus.SUCCESS with output set.

    _MAIN_SLOT_NODE identifies the GraphNode subclass in the `main` slot.
    The invoke is called with VERIFIED_EXTERNAL trust to satisfy the
    PreProcessNode S-1 gate (required_trust_level=VERIFIED_EXTERNAL).
    """

    @pytest.fixture(autouse=True)
    def _noop_domain_emit(self, monkeypatch):
        """Patch emit_trace_event at the framework module AND each domain node module.

        The framework's BaseNode.__call__() calls emit_trace_event from
        framework.nodes.base_node (node_start / node_complete / s1_denied events).
        Domain nodes call emit_trace_event from their own module binding.
        Both must be patched; neither via sys.modules["shared"] (that shadows
        the real shared package and breaks framework imports).
        """
        import framework.nodes.base_node as base_node_module
        import src.nodes.pre_process_node as m1
        import src.nodes.query_parse_node as m2
        import src.nodes.domain_route_node as m3
        import src.nodes.biobert_retrieve_node as m4
        import src.nodes.underwriting_assessment_format_node as m5
        import src.nodes.output_gate_node as m6
        import src.nodes.post_process_node as m7

        def noop(*a, **k):
            return None

        monkeypatch.setattr(base_node_module, "emit_trace_event", noop)
        for mod in (m1, m2, m3, m4, m5, m6, m7):
            monkeypatch.setattr(mod, "emit_trace_event", noop)

    def test_backbone_order_on_success(self):
        """Full invoke with _VALID_PAYLOAD completes SUCCESS through all 5 backbone slots.

        The outer agent is invoked with VERIFIED_EXTERNAL caller trust so that the
        PreProcessNode (pre_process, required_trust_level=VERIFIED_EXTERNAL) passes
        its S-1 gate. Inner domain nodes are ANONYMOUS and pass regardless.

        node_history contains CLASS NAMES (not slot names), as recorded by
        BaseNode.__call__() appending self.__class__.__name__.
        """
        from src.graph.graph import InsuranceMedicalAssessmentAgent
        from framework.schemas.agent_status import AgentStatus
        from framework.schemas.invocation_context import InvocationContext
        from framework.schemas.trust_level import TrustLevel

        agent = InsuranceMedicalAssessmentAgent()
        agent.compile()
        # VERIFIED_EXTERNAL trust satisfies PreProcessNode's S-1 gate
        ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
        result = agent.invoke(_VALID_PAYLOAD, ctx=ctx)

        # Must complete with SUCCESS (not ERROR / partial)
        assert result.get("status") == AgentStatus.SUCCESS.value, (
            f"Expected SUCCESS, got '{result.get('status')}'. " f"error_log: {result.get('error_log')}"
        )

        # AgentBaseGraph.get_output() maps state["formatted_output"] or state["result"] → "output".
        # PostProcessNode sets state["formatted_output"]; confirming "output" is non-None proves
        # the full S-3 gate path ran to completion.
        assert result.get("output") is not None, (
            "output not set — PostProcessNode (post_process) may have failed or been skipped. "
            "AgentBaseGraph.get_output() maps formatted_output → output."
        )

        # Verify the mandatory insurance disclaimer is present in the output
        assert "[INSURANCE ASSESSMENT DISCLAIMER]" in result.get("output", ""), (
            "Mandatory insurance assessment disclaimer missing from output — "
            "OutputGateNode or UnderwritingAssessmentFormatNode S-3 gate may have failed."
        )

        # Backbone node_history carries CLASS NAMES (BaseNode.__call__ appends
        # self.__class__.__name__), not LangGraph slot labels.
        node_history: list = result.get("node_history", [])
        expected_classes = [
            "InitializeNode",  # initialize slot
            "PreProcessNode",  # pre_process slot (S-1 gate)
            "MedicalAssessmentWorkflowGraphNode",  # main slot (GraphNode)
            "PostProcessNode",  # post_process slot (S-3 gate)
            "FinalizeNode",  # finalize slot
        ]
        assert len(node_history) >= 5, f"node_history should have ≥5 backbone entries, got: {node_history}"
        for cls_name in expected_classes:
            assert cls_name in node_history, f"Backbone class '{cls_name}' missing from node_history: {node_history}"
        # Strict order: each class appears after the previous one
        positions = {cls_name: node_history.index(cls_name) for cls_name in expected_classes}
        ordered_classes = sorted(positions, key=lambda s: positions[s])
        assert ordered_classes == expected_classes, (
            f"Backbone class order violated.\n"
            f"expected order: {expected_classes}\n"
            f"actual order:   {ordered_classes}\n"
            f"positions:      {positions}"
        )

    def test_main_slot_node_is_graph_node(self):
        """_MAIN_SLOT_NODE is a GraphNode subclass (Cat-2 nested pattern contract)."""
        from framework.nodes.graph_node import GraphNode

        assert issubclass(
            _MAIN_SLOT_NODE, GraphNode
        ), f"{_MAIN_SLOT_NODE.__name__} must extend GraphNode for Cat-2 main slot"

    def test_main_slot_registered_in_graph(self):
        """_MAIN_SLOT_NODE class is registered in the `main` slot of the outer graph."""
        from src.graph.graph import InsuranceMedicalAssessmentAgent

        agent = InsuranceMedicalAssessmentAgent()
        agent.compile()
        main_node = agent._nodes.get("main")
        assert main_node is not None, "No node registered in 'main' slot"
        assert isinstance(main_node, _MAIN_SLOT_NODE), (
            f"main slot holds {type(main_node).__name__}, " f"expected {_MAIN_SLOT_NODE.__name__}"
        )

    def test_verified_external_caller_required(self):
        """Verify that an ANONYMOUS caller is rejected by the PreProcessNode S-1 gate.

        This confirms the backbone trust gate is correctly enforced: only
        VERIFIED_EXTERNAL (or higher) callers reach the inner domain pipeline.
        The outer backbone sets the trust barrier; inner nodes are ANONYMOUS.
        """
        from src.graph.graph import InsuranceMedicalAssessmentAgent
        from framework.schemas.agent_status import AgentStatus
        from framework.schemas.invocation_context import InvocationContext
        from framework.schemas.trust_level import TrustLevel

        agent = InsuranceMedicalAssessmentAgent()
        agent.compile()
        # ANONYMOUS caller should be denied by PreProcessNode (VERIFIED_EXTERNAL required)
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS)
        result = agent.invoke(_VALID_PAYLOAD, ctx=ctx)
        # S-1 gate denies → ERROR or post_process/finalize skipped
        assert result.get("status") != AgentStatus.SUCCESS.value, (
            "ANONYMOUS caller should NOT reach SUCCESS — "
            "PreProcessNode S-1 gate (VERIFIED_EXTERNAL required) must deny it"
        )
