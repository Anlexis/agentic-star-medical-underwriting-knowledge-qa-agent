# Design Specification — INS-C2-051 InsuranceMedicalAssessmentAgent

## Position in the framework

| Aspect | Value |
|---|---|
| Agent class | `InsuranceMedicalAssessmentAgent` (`src/graph/graph.py`) |
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |
| Inner graph base | `BaseGraph` (`MedicalAssessmentDomainGraph`) |
| Pattern | Cat 2 nested — outer backbone with a `GraphNode` wrapping an inner domain pipeline |
| Category / industry | Cat 2 / INS |

Three-layer separation:

- **State** — a flat `TypedDict` (`src/schemas/state.py`). No Pydantic models and no arbitrary
  objects: checkpoints are serialised, and a non-primitive value corrupts silently rather than
  failing. Structured values travel as JSON strings via `to_json()`.
- **Node** — each node extends `FunctionNode` and implements
  `execute(self, state: AgentState) -> Dict[str, Any]`, returning only the fields it changes.
  The framework's node wrapper calls `execute(state)` with one argument; a node that declared a
  second `config` parameter would simply never receive one, which is why configuration reaches
  nodes through state and through the graph, never through the call.
- **Graph** — composition. `register_nodes()` selects the nodes; the inner pipeline is supplied
  by `get_subgraph()`.

## Architecture

```
Outer backbone (AgentBaseGraph — fixed):
  START → initialize → pre_process → main → {route} → post_process → finalize → END
                                              ↓ (retry, budget from config/config.yaml)
                                           pre_process

Slot mapping:
  pre_process  ← PreProcessNode                     (caller contract; VERIFIED_EXTERNAL)
  main         ← MedicalAssessmentWorkflowGraphNode (GraphNode — wraps the inner graph)
  post_process ← PostProcessNode                    (output gate; ANONYMOUS)

Inner domain graph (BaseGraph — fully custom topology):
  START → query_parse → domain_route → biobert_retrieve
        → underwriting_format → output_gate → END
```

### Nodes

| Layer | Node | Responsibility | Trust level | Reads | Writes |
|---|---|---|---|---|---|
| Backbone | `InitializeNode` | session setup, schema version | framework default | `user_input` | `session_id`, `trace_id`, `input_context` |
| Backbone | `PreProcessNode` | the caller contract: size, identifiers, instruction-override screen, context-parameter validation | VERIFIED_EXTERNAL | `user_input`, `input_context` | `validated_input`, `enriched_context`, `validated_context` |
| Backbone | `MedicalAssessmentWorkflowGraphNode` | runs the inner pipeline; bridges the validated parameters into it | inherited | `validated_input`, `validated_context` | `result`, `status` |
| Backbone | `PostProcessNode` | output gate: disclaimer, size, identifier markers, credential shapes | ANONYMOUS | `result` | `formatted_output` (or the withheld notice) |
| Backbone | `FinalizeNode` | response metadata | framework default | `formatted_output` | `node_history` |
| Inner | `QueryParseNode` | intent classification, disease-code and term extraction | ANONYMOUS | `validated_input` | `query_intent`, `extracted_medical_terms` |
| Inner | `DomainRouteNode` | biomedical domain selection, honouring `domain_hint` | ANONYMOUS | `query_intent`, `input_context` | `domain_routed` |
| Inner | `BioBERTRetrieveNode` | on-device knowledge-base retrieval at the requested depth | ANONYMOUS | `domain_routed`, `input_context` | `retrieved_passages` |
| Inner | `UnderwritingAssessmentFormatNode` | renders the assessment and the regulatory audit entry | ANONYMOUS | `retrieved_passages`, `query_intent`, `extracted_medical_terms` | `underwriting_assessment`, `hokengyoho_audit_log` |
| Inner | `OutputGateNode` | publishes the assessment only on a run it accepts | ANONYMOUS | `underwriting_assessment`, `status` | `formatted_output`, `disclaimer_applied` |

### Data flow

```
user_input + input_context
  → [PreProcessNode]  contract checks → validated_input + validated_context
  → [MedicalAssessmentWorkflowGraphNode]
       extract_input() stashes validated_context  (src/graph/context_bridge.py)
       → inner graph.invoke(validated_input)
            _extra_initial_state() seeds input_context from the stash
            → [QueryParseNode]                 → query_intent, extracted_medical_terms
            → [DomainRouteNode]                → domain_routed
            → [BioBERTRetrieveNode]            → retrieved_passages
            → [UnderwritingAssessmentFormatNode] → underwriting_assessment, audit entry
            → [OutputGateNode]                 → formatted_output (or nothing)
       → merge_output(): result = the inner graph's published output
  → [PostProcessNode]  output gate → formatted_output, or the withheld notice
  → [FinalizeNode] → response
```

### The caller-data contract

`POST /invoke` accepts the question as `input` and an optional `input_context` object. Exactly
four parameters are declared; anything else is refused by name rather than ignored.

| Parameter | Accepted values | Effect |
|---|---|---|
| `channel` | 1–64 characters of `[A-Za-z0-9_-]` | recorded in the enrichment and audit records |
| `underwriter_id` | 1–64 characters of `[A-Za-z0-9_-]` | recorded, and rendered in the assessment header |
| `domain_hint` | one of `diseases`, `oncology`, `anatomy`, `genes`, `phenotypes`, `chemicals`, `trials`, `general` | overrides the domain the question's intent would have selected |
| `max_passages` | a whole number in `[1, 5]` | how many knowledge-base passages the assessment renders |

Three properties hold by construction:

1. **Undeclared keys are refused, not dropped.** A dropped key would leave the caller with a
   success and a silently ignored parameter. A key that is merely ignored is worse: it stays in
   state, the first node returns the context object verbatim in its own result, and the
   framework's output scan then fails that node with an error the caller cannot act on.
2. **Every value is inert, closed-set or finite-bounded.** The two identifiers are restricted to
   a closed alphabet, the domain hint to the catalogue the retriever serves, and the depth to a
   range parsed with an explicit finite check — `NaN` parses through `float()` and compares
   `False` against every bound, so an unchecked number disables the bound it feeds.
3. **The parameters actually reach the inner pipeline.** The framework's `GraphNode` invokes a
   subgraph without forwarding `input_context`, so an inner read would otherwise always see an
   empty mapping. `src/graph/context_bridge.py` carries the *validated* parameters across on a
   `ContextVar`, which keeps concurrent invocations independent.

### State

| Field | Type | Purpose | Written by |
|---|---|---|---|
| `validated_context` | `Optional[str]` | JSON: the caller parameters after validation — the only form that crosses into the inner graph | `PreProcessNode` |
| `query_intent` | `Optional[str]` | `disease_inquiry` / `icd11_lookup` / `prognosis` / `treatment_protocol` / `oncology` / `general` | `QueryParseNode` |
| `extracted_medical_terms` | `Optional[str]` | JSON: disease codes and clinical terms found in the question | `QueryParseNode` |
| `domain_routed` | `Optional[str]` | the selected biomedical domain | `DomainRouteNode` |
| `retrieved_passages` | `Optional[str]` | JSON: `{text, score, source}` passages | `BioBERTRetrieveNode` |
| `underwriting_assessment` | `Optional[str]` | the rendered assessment | `UnderwritingAssessmentFormatNode` |
| `hokengyoho_audit_log` | `Optional[str]` | JSON: the regulatory audit entry | `UnderwritingAssessmentFormatNode`, `OutputGateNode` |
| `disclaimer_applied` | `Optional[bool]` | the mandatory disclaimer was verified present | `OutputGateNode` |

State constraints: primitives and JSON-serialisable values only; no credentials; no Pydantic
models or arbitrary objects; structured values serialised through `to_json()`.

## Security design

### S-1 — trust gate
`PreProcessNode` declares `VERIFIED_EXTERNAL`, so an unauthenticated caller cannot reach the
pipeline. The standalone entry point authenticates a bearer credential and only then presents a
`VERIFIED_EXTERNAL` caller; with no credential configured it answers `503` rather than accepting
requests it can only refuse four nodes later.

### S-2 — input gate
The framework masks personal-data shapes and screens actionable fields before `execute()` runs.
The template does not rely on that alone: the question is screened again inside `execute()`, for
chat-template control tokens (`<|…|>`, `[INST]`, `<<SYS>>`) and explicit instruction-override
directives, raw **and** with inline markup removed. Screening only after a strip would convert a
detectable token attack into undetectable plain text; screening only before it would miss a
directive spliced across tags. Patient identifiers are refused rather than masked, with digit
guards instead of a word boundary so a value written against Kanji is caught exactly as the same
value written between spaces.

The screen is deliberately narrow where clinical language overlaps with attack vocabulary: a
question about treatment guidelines or policy rules passes, while a directive scoped to the
agent's own instructions does not.

### S-3 — output gate
`OutputGateNode` (inner) publishes the assessment only on a run it accepts, and the inner graph's
output projection has no fallback to the un-gated text — the one state that would exercise such a
fallback is the state where the gate refused.

`PostProcessNode` (backbone) checks the released document for the mandatory disclaimer, a size
ceiling, applicant-identifier markers and credential shapes. The credential check is the union of
the framework's own detector and the local patterns: narrower than the framework and the framework
raises inside the node wrapper, which discards this node's containment; narrower than the local set
and an assignment line walks through, because the framework's patterns describe credential formats.

**Refusing is not containing.** The framework's envelope returns
`formatted_output or result`, on an error status too, so a gate that returns an error while leaving
`result` in place still ships the text it just refused. Every refusal here clears `result` and
publishes a non-empty withheld notice — non-empty because a falsy value re-opens the fallback.

**Numeric precision grid: not applicable.** This template renders no monetary aggregate. Its
output invariant is instead: the mandatory disclaimer is present, no applicant identifier or
credential shape appears, the document is within the size ceiling, and any quoted caller text is
collapsed to a single line with the renderer's structural characters removed — so a question
cannot manufacture a line that reads as a retrieved passage attributed to a named source.

### S-4 — audit
Every node emits domain events positionally through `emit_trace_event(event, payload, state)`:

- `ins_c2_051.pre_process.{accepted|rejected}`
- `ins_c2_051.query_parse.complete`
- `ins_c2_051.domain_route.selected`
- `ins_c2_051.biobert_retrieve.{complete|no_passages|error}`
- `ins_c2_051.underwriting_format.{complete|error}`
- `ins_c2_051.output_gate.{passed|disclaimer_absent|upstream_error|error}`
- `ins_c2_051.post_process.output_gate_verdict`

Payloads carry counts, closed-set labels and sizes. They never carry the question text, an
applicant-linked value, or the text that tripped a refusal.

### S-5 — credentials
No credential is held in state or in node code. Retrieval is on device, so no outbound call and no
credential is needed for it. The bearer credential the entry point compares comes from the
environment and is compared in constant time.

## Runtime configuration

`config/config.yaml` holds the runtime parameters. The registry loads it and passes it as
`Graph(config=...)`; the standalone entry point reads the same file, so a declared value has the
same effect however the agent is started. Values are range-checked on read and a value outside its
range is not forwarded, leaving the consumer on its documented default.

`max_retry` is read by the backbone's routing decision. `timeout_s` is declared for parity with
the platform's parameter set and has no reader in the framework or in this template.

## Key design decisions

| Decision | Alternative | Chosen | Rationale |
|---|---|---|---|
| Backbone base class | `AutonomousBaseGraph` | `AgentBaseGraph` | a fixed pipeline, not an autonomous loop |
| Inner graph base | `AgentBaseGraph` | `BaseGraph` | a fully custom topology with no backbone slots to fill |
| Inner topology | conditional branching | linear | all five steps always apply; the end-of-line gate refuses to report success on a run that already failed, which is what a branch would otherwise have prevented |
| Knowledge base | external service | on device | keeps applicant-linked data inside the execution boundary |
| Where the caller contract lives | the entry point only | `PreProcessNode`, with the entry point covering what only it can | the platform calls `invoke()` directly and never runs the entry point, so a rule owned only there would be absent on that path |
| Retrieval depth | fixed | caller-declared, bounded | a declared parameter that changes nothing is indistinguishable from one that is ignored |

## Import isolation

- [x] No platform-SDK import (Level 0) anywhere in `src/`
- [x] Imports are limited to `framework/`, `shared/` and the standard library
- [x] Graph execution is always through `.invoke()`
