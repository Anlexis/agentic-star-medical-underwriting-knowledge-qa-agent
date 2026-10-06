# Medical Underwriting Knowledge Q&A Agent

AI agent for answering medical assessment questions for insurance underwriting, built with Agentic Star.

> **Category**: Cat 2 (Domain-specific pipeline)
> **Industry**: Insurance
> **Template ID**: INS-C2-051

## Overview

An insurance underwriter assessing a life or health application often needs a short, sourced
summary of what is known about a medical condition — how it is classified, what drives its risk
rating, what the evidence says about treatment and outcome — before deciding whether to accept,
load, postpone or refer the case.

This agent answers that question. It takes a medical assessment question in English or Japanese,
works out what is being asked, selects the biomedical domain that best matches it, retrieves the
relevant passages from an on-device knowledge base, and renders an underwriting-ready summary:
the question as asked, the terms it recognised, the retrieved evidence with its sources and
relevance scores, the considerations an underwriter should apply, and a mandatory disclaimer
stating that the summary is reference material and not medical advice.

Two properties are deliberate and are enforced by the code rather than by convention. First,
retrieval runs entirely in process — nothing about an applicant leaves the execution boundary, and
a question carrying an applicant identifier is refused outright rather than answered in masked
form. Second, the assessment that reaches the caller is the one the output boundary accepted: if
it fails the boundary's checks the agent returns a short notice and nothing else, never the
document it just refused.

The bundled knowledge base is small and is meant to be replaced. Fork this template, point the
retrieval node at your own corpus or vector index, and the rest of the pipeline — the caller
contract, the parameter validation, the rendering and the output boundary — applies unchanged.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >= 3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails at graph
compile / start-up preflight rather than starting in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Calling the agent

`POST /invoke` takes the question and, optionally, four declared parameters. Anything not on this
list is refused by name rather than silently ignored.

```json
{
  "input": "The applicant has a documented history of Type 2 diabetes mellitus with code BA80. Prognosis is favorable based on stable control over three years.",
  "session_id": "case-40218",
  "input_context": {
    "channel": "portal",
    "underwriter_id": "uw_204",
    "domain_hint": "diseases",
    "max_passages": 4
  }
}
```

| Parameter | Accepted values | Effect |
|---|---|---|
| `channel` | 1–64 characters of `[A-Za-z0-9_-]` | recorded with the request |
| `underwriter_id` | 1–64 characters of `[A-Za-z0-9_-]` | recorded, and shown in the assessment header |
| `domain_hint` | `diseases`, `oncology`, `anatomy`, `genes`, `phenotypes`, `chemicals`, `trials`, `general` | overrides the domain the question would have selected |
| `max_passages` | a whole number in `[1, 5]` | how many passages the assessment renders |

The endpoint expects a bearer credential (`INVOKE_AUTH_TOKEN`, or `STG_INTERNAL_RUNNER_TOKEN`).
With neither configured it answers `503` rather than accepting requests it can only refuse.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent manifest and runtime parameters
docs/         design and operational documentation
```

See `docs/02_design.md` for the architecture and `docs/03_test_spec.md` for the test matrix.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the bundled knowledge base in `src/nodes/biobert_retrieve_node.py` with your own
   corpus, or wire a retriever in behind `src/services/service.py`.
3. Extend the domain catalogue in `src/nodes/domain_route_node.py` — `PreProcessNode` validates
   the caller's `domain_hint` against that same set, so the two cannot drift apart.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
