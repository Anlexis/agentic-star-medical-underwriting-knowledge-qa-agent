"""AgentCore Platform v1.0 — INS-C2-051 inner domain workflow graph (Cat 2)."""

# Inner domain workflow graph for INS-C2-051.
# Instantiated by MedicalAssessmentWorkflowGraphNode.get_subgraph() in graph.py.
#
# Pipeline (linear):
#   START → query_parse → domain_route → biobert_retrieve
#         → underwriting_format → output_gate → END
#
# Rules:
#   ✅ Inherits BaseGraph (fully custom topology, no pre_process/main/post_process slots)
#   ✅ Implements all 7 BaseGraph abstract methods
#   ✅ register_nodes() does NOT call super() (abstract in BaseGraph)
#   ✅ Inner domain nodes are instantiated with NO ctor args
#   ✅ required_trust_level = ANONYMOUS on all inner domain nodes
#   ✅ get_output() is designed together with merge_output() in graph.py
#   ❌ Do NOT register initialize / finalize (outer backbone concern)
#   ❌ Do NOT call super() in register_nodes()

from typing import Any, Dict

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_input_context
from src.nodes.biobert_retrieve_node import BioBERTRetrieveNode
from src.nodes.domain_route_node import DomainRouteNode
from src.nodes.output_gate_node import OutputGateNode
from src.nodes.query_parse_node import QueryParseNode
from src.nodes.underwriting_assessment_format_node import UnderwritingAssessmentFormatNode
from src.schemas.state import State


class MedicalAssessmentDomainGraph(BaseGraph):
    """Inner domain workflow graph for INS-C2-051.

    Runs the assessment pipeline from parsed question to formatted underwriting
    response. Called by MedicalAssessmentWorkflowGraphNode.get_subgraph().

    Pipeline:
      START
        → query_parse         (intent classification + disease-code extraction)
        → domain_route        (biomedical domain selection)
        → biobert_retrieve    (on-device knowledge-base retrieval)
        → underwriting_format (assessment rendering + regulatory audit entry)
        → output_gate         (disclaimer check + audit finalisation)
      → END
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "ins_c2_051_medical_assessment_domain"

    @property
    def state_schema(self) -> type:
        return State

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """No mandatory configuration: every declared value has a documented default."""
        return None

    # ── Caller context ────────────────────────────────────────────────────────

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the validated caller parameters into the inner initial state.

        The framework's GraphNode invokes this graph without forwarding
        input_context, so without this hook every inner read of
        state["input_context"] would see {} and every declared parameter would be
        silently ignored. The value comes from the outer node's extract_input()
        via src/graph/context_bridge.py, and it is the VALIDATED form — the raw
        request never reaches here.
        """
        return {"input_context": get_caller_input_context()}

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register all domain nodes with NO constructor arguments.

        Inner graph nodes are no-arg; configuration travels on the graph, not on
        the node. A ctor argument raises TypeError at graph compile time.
        """
        self._nodes["query_parse"] = QueryParseNode()
        self._nodes["domain_route"] = DomainRouteNode()
        self._nodes["biobert_retrieve"] = BioBERTRetrieveNode()
        self._nodes["underwriting_format"] = UnderwritingAssessmentFormatNode()
        self._nodes["output_gate"] = OutputGateNode()

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear domain pipeline.

        All five nodes always execute, in order. Because there is no conditional
        edge, a failure part-way through does not stop the line — OutputGateNode
        is what refuses to report success on a run that already failed.
        """
        self._sg.add_edge(START, "query_parse")
        self._sg.add_edge("query_parse", "domain_route")
        self._sg.add_edge("domain_route", "biobert_retrieve")
        self._sg.add_edge("biobert_retrieve", "underwriting_format")
        self._sg.add_edge("underwriting_format", "output_gate")
        self._sg.add_edge("output_gate", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    # route() exists only to satisfy the BaseGraph ABC, which declares it
    # abstract. This graph's topology is a fixed linear pipeline — add_edges()
    # registers no conditional edge — so the runtime never dispatches through it.
    # It is annotated with this graph's own State rather than the framework base
    # state because a path callable's annotation IS the schema the runtime
    # projects onto: annotating a narrower type would silently drop this graph's
    # own fields if a conditional edge were ever added here.
    def route(self, state: State) -> str:
        """Required by the BaseGraph ABC; never called on a linear topology."""
        if state.get("status") in (AgentStatus.ERROR, AgentStatus.ERROR.value):
            return END
        return "output_gate"

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: State) -> Dict[str, Any]:
        """Shape the sub_result returned to MedicalAssessmentWorkflowGraphNode.

        `formatted_output` is published only by OutputGateNode, and only on a run
        it accepted. There is deliberately NO fallback to the raw assessment: the
        one state that would exercise such a fallback is the one where the gate
        refused, so the fallback would hand the outer graph exactly the text the
        gate withheld.
        """
        return {
            "output": state.get("formatted_output"),
            "status": state.get("status"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
