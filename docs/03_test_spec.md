# Test Specification — INS-C2-051 InsuranceMedicalAssessmentAgent

## Scope

The suite tests the Cat 2 nested pipeline at three levels, and the level is chosen by what the
test has to prove:

```
Outer backbone: initialize → pre_process → main (GraphNode) → post_process → finalize
Inner domain:   query_parse → domain_route → biobert_retrieve → underwriting_format → output_gate
```

- **The caller contract is proved by calling `execute()` directly**, with no framework wrapper in
  front. A refusal asserted only through the framework's own input gate passes where that gate is
  active and fails open where it is absent or configured off, so the guarantee has to be the
  template's own.
- **Behaviour that the caller sees is proved through the real ASGI entry point** — authentication,
  the parameter bridge into the inner graph, and the released document.
- **Assertions are behavioural.** A test asserts an error status, a field that was not carried
  forward, a value that is not echoed. It never asserts a framework message's wording, which
  changes between releases and tells you nothing about containment.

## Test files

| File | What it covers |
|---|---|
| `tests/unit/test_agent.py` | each node in isolation: classification, routing, retrieval, rendering, both gates |
| `tests/unit/test_caller_contract.py` | the caller contract at the node that owns it, both directions |
| `tests/unit/test_framework_compliance_tc06_tc07.py` | framework compliance checks |
| `tests/proof_of_boundary/test_invoke_e2e.py` | end-to-end through the real ASGI `/invoke` |
| `tests/proof_of_boundary/test_output_containment.py` | every refusal path of the output boundary |
| `tests/proof_of_boundary/test_pb_invoke_order.py` | per-node gate order and full backbone order |
| `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | interrupt propagation (skipped: no review step configured) |
| `tests/proof_of_boundary/test_import_isolation.py` | no platform-SDK import in `src/` |
| `tests/proof_of_boundary/test_state_safety.py` | state carries no credentials and no non-serialisable objects |

## Unit — nodes (`tests/unit/test_agent.py`)

| TC | Node | What it asserts |
|---|---|---|
| UT-01 | `PreProcessNode` | a well-formed question is accepted and promoted to `validated_input` |
| UT-02 | `PreProcessNode` | an empty question is refused |
| UT-03 | `PreProcessNode` | a whitespace-only question is refused |
| UT-04 | `PreProcessNode` | a question over the 4,000-character ceiling is refused, nothing is carried forward, and the value is not echoed |
| UT-05 | `PreProcessNode` | a bare policy-number run is refused |
| UT-06 | `PreProcessNode` | an explicit `patient_id:` marker is refused |
| UT-07 | `PreProcessNode` | the Japanese individual-number marker is refused |
| UT-08 | `PreProcessNode` | the accepted question is what reaches `validated_input` |
| UT-09 | `PreProcessNode` | the node declares `VERIFIED_EXTERNAL` |
| UT-10..15 | `QueryParseNode` | intent classification for disease / prognosis / oncology / unknown; terms extracted as JSON; a missing question is an error |
| UT-16..20 | `DomainRouteNode` | each intent routes to its documented domain |
| UT-21..23 | `BioBERTRetrieveNode` | each domain returns passages, serialised as valid JSON |
| UT-24..27 | `UnderwritingAssessmentFormatNode` | the disclaimer is always present; the audit entry is valid JSON; the domain appears in the header; an empty passage list still renders |
| UT-28..30 | `OutputGateNode` | an accepted assessment is published with the audit entry finalised; a missing disclaimer is refused |
| UT-31..34 | `PostProcessNode` | an accepted document is released; a missing document, a missing disclaimer and an oversized document are refused |

## Unit — the caller contract (`tests/unit/test_caller_contract.py`)

| TC | What it asserts |
|---|---|
| CC-01 | every attack form is refused: `<\|im_start\|>`, `[INST]`, `<<SYS>>`, English and Japanese instruction-override directives, and a directive spliced across inline markup |
| CC-02 | a corpus of real underwriting and clinical questions is **not** refused — including ones containing "guidelines", "rules", "override", "excluded" and "you are now". A screen that refuses real work is the failure mode that costs more |
| CC-03 | a refusal names the field and never quotes the value back |
| CC-04 | patient identifiers are detected with digit guards rather than a word boundary, so the Japanese-adjacent form (`番号1234567890を`) is caught exactly as the ASCII-spaced form is |
| CC-05 | disease codes, dates, monetary caps, ratios and article numbers are **not** identifiers and pass unchanged |
| CC-06 | the accepted parameter set is exactly the four documented ones |
| CC-07 | a valid parameter object is carried forward in validated form; an absent one is accepted |
| CC-08 | an undeclared key is refused, not ignored |
| CC-09 | a hostile field NAME is reported positionally, never echoed |
| CC-10 | a non-inert `channel` is refused, for every wrong shape and type |
| CC-11 | `domain_hint` accepts only the domains the retriever serves |
| CC-12 | `max_passages` rejects `NaN`, `±Infinity`, booleans, strings, non-integral floats and out-of-range values, and accepts the declared range |
| CC-13 | the parameter object is screened post-parse, keys included, so a `\u`-escaped directive is still seen |
| CC-14 | the enrichment record is an assembled mapping of inert values, not an interpolated string |
| CC-15 | a quoted question cannot manufacture the assessment's own structure: newlines collapse, structural characters are removed, the quote is length-capped, and the redaction sentinel stays legible |
| CC-16 | the credential screen catches the framework's shapes **and** the local assignment shape the framework does not carry |
| CC-17 | ordinary clinical text is not read as a credential |

## Boundary — the released document (`tests/proof_of_boundary/test_output_containment.py`)

| PB | What it asserts |
|---|---|
| PB-C1 | for each of the five refusal reasons — missing disclaimer, identifier marker, credential shape, credential assignment, oversize — the refused text is cleared from `result`, the withheld notice is published, and the envelope a caller would receive contains neither the text nor the credential |
| PB-C2 | an absent document is also a refusal |
| PB-C3 | the audited reason is a closed label; the text that tripped it is not in the log |
| PB-C4 | an accepted document is released byte-identical |
| PB-C5 | the withheld notice is non-empty — a falsy replacement re-opens the fallback the clearing exists to close |
| PB-C6 | the inner graph publishes nothing on a run its own gate refused, and its output projection has no fallback to the un-gated assessment |
| PB-C7 | an earlier failure is not erased by a later node succeeding at the end of a linear pipeline |

## Boundary — end to end (`tests/proof_of_boundary/test_invoke_e2e.py`)

| PB | What it asserts |
|---|---|
| PB-E1 | a deployment with no credential configured answers `503` at the door rather than accepting requests it can only refuse |
| PB-E2 | a wrong or missing bearer is `401`, and the response does not say which |
| PB-E3 | the separate runner credential is accepted as well as the ordinary one |
| PB-E4 | a valid question produces a gated assessment carrying the mandatory disclaimer |
| PB-E5 | `max_passages` reaches the inner graph and changes the number of rendered passages — the bridge is load-bearing |
| PB-E6 | `domain_hint` reaches the inner graph and changes which knowledge base is read |
| PB-E7 | `underwriter_id` is rendered from caller data |
| PB-E8 | two different questions produce different documents |
| PB-E9 | an undeclared parameter is refused with `400` before `invoke()`, naming the field |
| PB-E10 | a credential-shaped parameter value is refused with `400` naming the field, while ordinary text on the same field passes |
| PB-E11 | an oversized question, a credential-shaped question and a non-inert session identifier are each refused with `400` |
| PB-E12 | instruction-override content, a patient identifier and an out-of-range or non-finite depth are refused by the agent |
| PB-E13 | on every refusal the envelope carries no assessment, no traceback and no source paths, and the error log is not projected into it |

## Boundary — framework contract

| PB | File | What it asserts |
|---|---|---|
| PB-6a | `test_pb_invoke_order.py` | every node's `__call__()` runs S-1 → node_start → S-2 → `execute()` → S-3 → node_complete in order |
| PB-6b | `test_pb_invoke_order.py` | a full `invoke()` traverses all five backbone slots in order and returns a document with the disclaimer; an ANONYMOUS caller is denied |
| PB-7 | `test_pb7_hitl_interrupt_propagation.py` | skipped — this template configures no human-review step |
| PB-4 | `test_import_isolation.py` | `src/` imports no platform SDK |
| PB-2/5 | `test_state_safety.py` | state holds no credential field, no invocation context and no non-serialisable object |

## Security coverage

| Layer | Where it is proved |
|---|---|
| S-1 trust | PB-6b (ANONYMOUS denied) and PB-E1..E3 (the entry point makes VERIFIED_EXTERNAL reachable at all) |
| S-2 input | CC-01..CC-05, CC-10..CC-13 by direct `execute()`; PB-E12 end to end |
| S-3 output | PB-C1..C7 at the node and envelope level; PB-E13 end to end |
| S-4 audit | CC-03, PB-C3 — the refusal reason is a closed label and the payload carries no caller text |
| S-5 credentials | CC-16, PB-C1 — the screen is the union of the framework's detector and the local patterns |

## Running the suite

```bash
pytest tests/ -v            # everything
pytest tests/unit -v        # nodes and the caller contract
pytest tests/proof_of_boundary -v
```

The suite runs without a platform connection. Running the agent itself does not.

## Regulatory notes

- The identifier refusals (UT-05..07, CC-04) keep applicant-linked values out of the retrieval
  path entirely, rather than answering a masked version of the question.
- The audit entry (UT-25, UT-30) records the shape of the assessment — domain, intent, counts and
  gate verdict — and no question text or applicant-linked value. CC-03 and PB-C3 assert that
  property directly rather than leaving it to inspection.
