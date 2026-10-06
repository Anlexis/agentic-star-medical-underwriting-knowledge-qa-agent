"""AgentCore Platform v1.0 — INS-C2-051 outer graph (Cat 2)."""

# Cat 2 outer graph — AgentBaseGraph with a GraphNode in the `main` slot.
#
# Architecture:
#   Outer backbone (fixed, standard 5-node):
#     START → initialize → pre_process → main → {route} → post_process → finalize → END
#
#   Slot mapping for INS-C2-051:
#     pre_process  ← PreProcessNode   (external gate; owns the caller contract)
#     main         ← MedicalAssessmentWorkflowGraphNode (wraps the inner graph)
#     post_process ← PostProcessNode  (backbone output gate)
#
#   Inner graph (src/graph/domain_workflow_graph.py):
#     START → query_parse → domain_route → biobert_retrieve
#           → underwriting_format → output_gate → END
#
# Two pieces of plumbing live here because there is nowhere else they can live:
#
#   * runtime configuration. The platform registry constructs the graph as
#     Graph(config=...) from config/config.yaml; a standalone process constructs
#     it directly and would otherwise run on built-in defaults, so every declared
#     value would be inert on exactly the deployment shape an operator tests
#     against. _runtime_config() reads the same file, and _parent_config() hands
#     it to the inner graph, which the framework does not do either.
#   * the caller's context parameters. The framework's GraphNode invokes the
#     subgraph without forwarding input_context, so an inner node reading it
#     would always see {}. extract_input() is the last hook that sees the outer
#     state before the inner invoke; it stashes the VALIDATED parameters, and the
#     inner graph seeds them (see src/graph/context_bridge.py).
#
# Class name matches:
#   config/agent.yaml  `class: src.graph.graph.InsuranceMedicalAssessmentAgent`
#   src/api/server.py  `from src.graph.graph import InsuranceMedicalAssessmentAgent`
#
# Rules:
#   ✅ Outer graph inherits AgentBaseGraph
#   ✅ Call super().register_nodes() in the outer graph (injects initialize + finalize)
#   ✅ Assign a GraphNode subclass to the `main` slot
#   ❌ Do NOT override add_edges() on the outer graph

import json
import math
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, cast

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from src.graph.context_bridge import set_caller_input_context
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State

# Runtime parameters: src/graph/graph.py -> parents[2] is the repository root.
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

# Bounds for the declared runtime settings. A value outside its bound is not
# forwarded — the consumer keeps its built-in default — rather than crashing
# graph construction on a malformed configuration file.
_MAX_RETRY_MIN, _MAX_RETRY_MAX = 0, 9
_TIMEOUT_MIN, _TIMEOUT_MAX = 1, 600


def _config_int(value: Any, lo: int, hi: int) -> Optional[int]:
    """Validate a declared integer setting: a real int, finite, within ``[lo, hi]``.

    Bools, strings, non-integral floats, NaN / Infinity and out-of-range values
    return None so the consumer keeps its documented default. The non-finite case
    is the dangerous one: NaN compares False against every bound, so an unchecked
    NaN retry budget would make every later bound check silently pass.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        return None
    parsed = int(number)
    return parsed if lo <= parsed <= hi else None


def runtime_config() -> Dict[str, Any]:
    """Read the runtime parameters from config/config.yaml.

    This is the same file the platform registry loads and passes as
    Graph(config=...); the standalone entry point and _parent_config() read it
    here so every deployment shape sees identical configuration. Returns an empty
    mapping — never raises — when the file is absent, unreadable, not valid YAML
    or not a mapping, in which case the graph runs on its built-in defaults.

    Only the keys with a validated bound are returned, so a malformed declaration
    cannot reach a consumer as an unchecked value.
    """
    try:
        import yaml

        loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}

    resolved: Dict[str, Any] = {}
    max_retry = _config_int(loaded.get("max_retry"), _MAX_RETRY_MIN, _MAX_RETRY_MAX)
    if max_retry is not None:
        resolved["max_retry"] = max_retry
    timeout_s = _config_int(loaded.get("timeout_s"), _TIMEOUT_MIN, _TIMEOUT_MAX)
    if timeout_s is not None:
        resolved["timeout_s"] = timeout_s
    return resolved


class MedicalAssessmentWorkflowGraphNode(GraphNode):
    """Wraps the inner MedicalAssessmentDomainGraph; assigned to the `main` slot.

    Orchestrates the domain pipeline:
      QueryParse → DomainRoute → BioBERTRetrieve
        → UnderwritingAssessmentFormat → OutputGate

    No constructor arguments — nodes are no-arg. Configuration reaches the
    subgraph through _parent_config().
    """

    error_strategy: ClassVar[str] = "propagate"
    propagate_hitl: ClassVar[bool] = False

    def _parent_config(self) -> Dict[str, Any]:
        """Runtime parameters for the inner graph, from the same file as the outer."""
        return runtime_config()

    def get_subgraph(self) -> Any:
        """Instantiate and return the inner domain workflow graph.

        Called on every execute(). The inner graph holds no external connections,
        so a fresh instance per call is correct and keeps concurrent invocations
        independent.
        """
        from src.graph.domain_workflow_graph import MedicalAssessmentDomainGraph

        return MedicalAssessmentDomainGraph(config=self._parent_config())

    def extract_input(self, state: AgentState) -> str:
        """Return the question handed to the inner graph's invoke().

        This is also where the validated context parameters cross the boundary:
        the framework's GraphNode does not forward input_context to the subgraph,
        and this is the last hook in this repository's code that sees the outer
        state before the inner invoke. Only the VALIDATED parameters are stashed —
        the raw request never crosses.
        """
        stashed: Dict[str, Any] = {}
        raw = state.get("validated_context")
        if isinstance(raw, str) and raw:
            try:
                decoded = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                decoded = None
            if isinstance(decoded, dict):
                stashed = decoded
        set_caller_input_context(stashed)
        return cast(str, state.get("validated_input") or state.get("user_input", ""))

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner graph's get_output() fields back into the outer state.

        `output` becomes `result`, which PostProcessNode gates before release.
        """
        return {
            "result": sub_result.get("output"),
            "status": sub_result.get("status"),
        }


class InsuranceMedicalAssessmentAgent(AgentBaseGraph):
    """Cat 2 outer graph for INS-C2-051.

    Backbone: initialize → pre_process → main → post_process → finalize (fixed).
    Domain complexity is encapsulated in MedicalAssessmentWorkflowGraphNode.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Construct with the declared runtime parameters when none are supplied.

        The platform registry passes config explicitly; a standalone process does
        not, and without this the declared max_retry would be ignored on exactly
        the deployment an operator tests by hand.
        """
        super().__init__(config if config is not None else runtime_config())

    @property
    def name(self) -> str:
        return "InsuranceMedicalAssessmentAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        super().register_nodes()  # injects InitializeNode + FinalizeNode (required)
        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = MedicalAssessmentWorkflowGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring is the framework's concern.


# Alias kept for the entry point's import.
Graph = InsuranceMedicalAssessmentAgent
