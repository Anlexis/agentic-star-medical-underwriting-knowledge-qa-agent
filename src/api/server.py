"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
#
# Entry points are adapters only — no business logic here. For platform-level
# routing the gateway calls agent.invoke() directly and this module is not used;
# that is why the caller contract is enforced in the pre_process node as well as
# here. The two are not duplicates: they cover two different entry paths, and a
# rule owned only by this file would be absent on the platform path.
#
# Four things this adapter owns, and each exists because of what happens when it
# does not:
#
#   1. Caller authentication. The agent's trust boundary admits only a
#      VERIFIED_EXTERNAL caller, and an unauthenticated caller is ANONYMOUS. With
#      no token configured, every request was therefore refused deep inside the
#      graph and answered 200 with an error body — a deployment that looks
#      healthy and serves nothing. An unconfigured deployment now refuses at the
#      door with 503 and says why.
#
#   2. The shape of the context object, before invoke(). The framework's first
#      node returns the caller's context verbatim into its own result, and the
#      framework's output gate scans every value of every result — so an
#      undeclared key carrying a credential-shaped string fails node one with a
#      traceback the caller cannot act on. Validators ignore undeclared keys, and
#      ignoring is not stripping: the key stays in state and reaches that scan.
#      This adapter therefore refuses undeclared keys rather than passing them
#      on, and screens the declared ones with the framework's own detector, so
#      the refusal set is exactly the block set enforced one layer later.
#
#   3. Bounding what enters the graph — the question's length and the number of
#      context keys — so an oversized body is refused before it becomes state.
#
#   4. Bounding the session identifier. It travels into correlation and audit
#      records, so it is restricted to a closed alphabet rather than accepted as
#      free text.
#
# 400 is used rather than 422: pydantic owns 422 and returns a list of error
# objects there, so reusing it would make client handling ambiguous.

import os
import secrets
from typing import Any, Dict, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets import factory as secrets_factory
from src.graph.graph import InsuranceMedicalAssessmentAgent
from src.nodes.pre_process_node import ACCEPTED_CONTEXT_FIELDS
from src.services.service import detect_output_credentials, is_inert_token, safe_field_label

app = FastAPI(title="Agent")

agent = InsuranceMedicalAssessmentAgent()
agent.compile()
# Namespace and agent name match config/agent.yaml.
agent.provision_secrets(secrets_factory(namespace="ins", agent_name="InsuranceMedicalAssessmentAgent"))

# Upper bound on the question, mirroring the caller boundary's own limit so an
# oversized body is refused before it is parsed into state.
MAX_INPUT_CHARS = 4000

# The context object declares at most four parameters; a body carrying more than
# that is refused on shape alone, before any value is read.
MAX_CONTEXT_KEYS = len(ACCEPTED_CONTEXT_FIELDS)


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""
    input_context: Dict[str, Any] = Field(default_factory=dict)


def _reject_context(context: Dict[str, Any]) -> None:
    """Refuse a context object this agent cannot accept; return None if it can.

    Undeclared keys are refused rather than dropped: a dropped key would leave
    the caller with a success and a silently ignored parameter, and a passed-on
    key would reach the framework's own scan and fail the first node opaquely.
    """
    if len(context) > MAX_CONTEXT_KEYS:
        raise HTTPException(
            status_code=400,
            detail=f"Field 'input_context' accepts at most {MAX_CONTEXT_KEYS} parameters.",
        )
    for position, (key, value) in enumerate(context.items(), start=1):
        label = safe_field_label(key, position)
        if key not in ACCEPTED_CONTEXT_FIELDS:
            raise HTTPException(
                status_code=400,
                detail=f"Field 'input_context.{label}' is not a parameter this agent accepts.",
            )
        if detect_output_credentials(value):
            # Name the field, never the value or the pattern's matched text.
            raise HTTPException(
                status_code=400,
                detail=f"Field 'input_context.{label}' carries a credential-shaped value and was refused.",
            )


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Dict[str, Any]:
    # The deployment may present either the ordinary external bearer or the
    # separate runner credential; both authenticate the same VERIFIED_EXTERNAL
    # caller this agent admits.
    configured = [
        token
        for token in (
            os.environ.get("INVOKE_AUTH_TOKEN"),
            os.environ.get("STG_INTERNAL_RUNNER_TOKEN"),
        )
        if token
    ]
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)

    if trust is TrustLevel.ANONYMOUS:
        if not configured:
            # Nothing can authenticate a caller, so nothing this endpoint returns
            # could be an answer. Say so once, at the door.
            raise HTTPException(
                status_code=503,
                detail="Caller authentication is not configured on this deployment.",
            )
        supplied = request.headers.get("authorization", "").encode()
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of the generic 401.
        if not any(secrets.compare_digest(supplied, f"Bearer {token}".encode()) for token in configured):
            # Generic body on purpose — do not leak whether the token was absent,
            # malformed, or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL

    if len(req.input) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Field 'input' exceeds the {MAX_INPUT_CHARS}-character limit.",
        )

    for field, value in (("input", req.input), ("session_id", req.session_id)):
        if detect_output_credentials(value):
            raise HTTPException(
                status_code=400,
                detail=f"Field '{field}' carries a credential-shaped value and was refused.",
            )

    if req.session_id and not is_inert_token(req.session_id):
        raise HTTPException(
            status_code=400,
            detail="Field 'session_id' must be 1-64 characters of [A-Za-z0-9_-].",
        )

    _reject_context(req.input_context)

    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        # The framework wheel ships no type information, so invoke() is Any.
        return cast(
            Dict[str, Any],
            agent.invoke(req.input, ctx=ctx, input_context=dict(req.input_context)),
        )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "agent": "InsuranceMedicalAssessmentAgent"}
