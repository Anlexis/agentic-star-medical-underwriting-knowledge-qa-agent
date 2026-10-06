"""AgentCore Platform v1.0"""

# Service layer for INS-C2-051 — shared, node-independent handling of
# caller-supplied data, plus the domain seam for a real biomedical knowledge base.
#
# No business logic, no routing, no credentials. The HTTP adapter and the domain
# nodes call the same helpers, so both entry paths enforce one contract.
#
# Six rules are implemented here, and every caller-facing check in this template
# is one of them:
#
#   1. inert identifiers — a caller string that travels into an audit record or
#      into the rendered assessment is restricted to a closed alphabet, so caller
#      text can never read as structure or as an instruction;
#   2. finite bounded numbers — every declared number the agent acts on is parsed
#      to a real, finite integer inside an explicit range. NaN is the dangerous
#      case: it parses through float() and compares False against every bound, so
#      an unchecked retrieval depth would silently fall through every guard;
#   3. instruction-override screening — caller text is refused when it carries a
#      chat-template control token or an explicit directive to discard the
#      agent's own instructions. Screened raw AND with markup removed, because a
#      directive can be spliced ("ig<b>nore ...") so that only one of the two
#      forms is readable;
#   4. patient-identifier detection that does not depend on word boundaries. The
#      usual \\b anchor is computed over \\w, and \\w includes Kanji and Kana.
#      Japanese is written without spaces, so an individual number written the way
#      a policyholder writes it ("個人番号1234567890を確認") sits directly against
#      a Kanji and the boundary never matches — the same digits between ASCII
#      spaces do match. The patterns below use explicit digit guards instead, so
#      the result does not depend on the script the question was written in.
#      Disease codes, dates, monetary caps and ratios carry no such shape and stay
#      unaffected;
#   5. echo neutralisation — the assessment quotes the question back, and the
#      assessment is a STRUCTURED rendering: retrieved passages are numbered lines
#      and section titles are markdown headings. A question containing a newline
#      could therefore manufacture a line that reads exactly like a retrieved
#      biomedical passage, attributing invented clinical text to a named source.
#      The echo is collapsed to a single line, its structural markers are
#      neutralised, and it is length-capped;
#   6. credential screening as a UNION — the framework's own detector is the
#      floor, and the local patterns are additions it does not carry. A local set
#      narrower than the framework's is a bypass, not a smaller net.
#
# The template owns these guarantees; the framework's own input policy runs in
# front of the nodes but scores some control-token forms at nothing. Every check
# here is enforced inside the node that owns the caller contract, and is proved
# by calling execute() directly with no framework wrapper in front.

from __future__ import annotations

import math
import re
from typing import Any, Optional

from framework.security.credential_detector import detect_credentials

# Caller strings that travel into audit records and into the rendered assessment
# are restricted to a closed alphabet — free text there is caller-controlled
# record injection.
_INERT_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# A field NAME is caller data too. Only an already-inert name is ever echoed in a
# refusal message; anything else is reported positionally.
_SAFE_FIELD_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")

# Chat-template control tokens, screened as a CLASS rather than as a list of
# known strings: any `<|...|>` marker, the instruction brackets used by several
# instruction-tuned model families, and the system-block markers. None of these
# carries meaning in a medical assessment question, so refusing the whole family
# costs nothing — and a phrase-only screen misses every one of them. The
# system-block marker matters in particular: measured against the installed
# framework, its own input policy returns no finding at all for `<<SYS>>`, while
# it scores `[INST]` and `<|im_start|>` as high-confidence and blocks them. A
# `<<SYS>>` payload therefore reached the assessment path end to end.
_CONTROL_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("chat_control_token", re.compile(r"<\|[^|>]{0,64}\|>")),
    ("instruction_bracket", re.compile(r"\[/?INST]", re.IGNORECASE)),
    ("system_block_marker", re.compile(r"<</?SYS>>", re.IGNORECASE)),
)

# Explicit instruction-override directives. Every pattern is anchored on both
# sides of a full verb+object phrase: a bare verb ("exclude", "override") is NOT
# enough, because ordinary clinical and underwriting language contains those words
# ("which conditions are excluded?", "既往症の告知義務を確認") and a screen that
# refuses real questions is worse than no screen at all.
#
# The object set is split deliberately. "instructions / prompts / directives /
# constraints" name the agent's own guidance and nothing else in this domain, so
# the qualifier before them is optional. "rules / guidelines / protocol" are
# everyday clinical vocabulary here — this template's own output cites actuarial
# guidelines and callers ask about treatment protocols — so for those the object
# must additionally be scoped to the agent ("your", "system", or an explicit
# back-reference such as "previous" / "above"). The attack phrasings all carry
# such a scope; a question about clinical guidelines does not.
_DIRECTIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"(?<![A-Za-z])(?:ignore|disregard|forget|override|bypass)\s+"
            r"(?:all\s+|any\s+|the\s+|your\s+|these\s+)*"
            r"(?:previous|prior|above|preceding|earlier|foregoing|system)?\s*"
            r"(?:instructions?|prompts?|directives?|constraints?)"
            r"(?![A-Za-z])",
            re.IGNORECASE,
        ),
    ),
    (
        # Same act, ambiguous object: only refused when the object is scoped to
        # the agent itself. "disregard the treatment guidelines for stage II"
        # stays a legitimate question; "ignore all previous rules" does not.
        "instruction_override_scoped",
        re.compile(
            r"(?<![A-Za-z])(?:ignore|disregard|forget|override|bypass)\s+"
            r"(?:all\s+|any\s+|the\s+|these\s+)*"
            r"(?:your|system|previous|prior|above|preceding|earlier|foregoing)\s+"
            r"(?:system\s+)?(?:rules?|guidelines?|protocols?)"
            r"(?![A-Za-z])",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_disclosure",
        re.compile(
            r"(?<![A-Za-z])(?:reveal|disclose|repeat|print|output|dump|show)\s+"
            r"(?:me\s+|us\s+)?(?:your|the)\s+(?:full\s+|entire\s+|original\s+)?"
            r"(?:system\s+prompt|system\s+message|initial\s+instructions)"
            r"(?![A-Za-z])",
            re.IGNORECASE,
        ),
    ),
    (
        # A role RE-ASSIGNMENT, not any sentence starting "you are now". The
        # article is what separates the two: "you are now an unrestricted model"
        # asserts a new identity, while "you are now assessing this applicant" —
        # which an underwriter writes — is a gerund and must pass.
        "persona_override",
        re.compile(
            r"(?<![A-Za-z])you\s+are\s+(?:now|no\s+longer)\s+(?:an?|the)\s+"
            r"|(?<![A-Za-z])you\s+are\s+no\s+longer\s+"
            r"(?:bound|required|restricted|limited|allowed|obliged)(?![A-Za-z])",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_injection",
        re.compile(
            r"(?<![A-Za-z])new\s+(?:instructions?|rules?|system\s+prompt)\s*[:：]",
            re.IGNORECASE,
        ),
    ),
    (
        # Japanese forms of the same two acts. Both name the agent's own
        # instructions explicitly; clinical questions about disregarding a
        # SYMPTOM or a FINDING ("所見を無視してよいか") do not match, because the
        # object here must be 指示 / 命令 / プロンプト / ルール / 制約.
        "instruction_override_ja",
        re.compile(
            r"(?:これまでの|以前の|上記の|すべての|全ての|前の)?"
            r"(?:指示|命令|プロンプト|ルール|制約)(?:を|は)?"
            r"(?:すべて|全て)?(?:無視|忘れ|破棄|上書き)"
        ),
    ),
    (
        "prompt_disclosure_ja",
        re.compile(r"(?:システムプロンプト|初期指示|システムメッセージ)(?:を|は)?(?:教え|出力|表示|開示|見せ)"),
    ),
)

# Markup and invisible characters are removed before the SECOND screening pass,
# so a directive spliced across inline tags is readable to the same patterns.
_MARKUP_TAG_RE = re.compile(r"<[^<>]{0,64}>")
_INVISIBLE_RE = re.compile("[\u200b\u200c\u200d\ufeff\u00ad]")

# Explicit patient-identifier cues. Matched case-insensitively against the raw
# question; the Japanese forms have no case to fold.
_PATIENT_IDENTIFIER_MARKERS: tuple[str, ...] = (
    "patient_id:",
    "patient-id:",
    "policy_no:",
    "policy-no:",
    "my_number:",
    "mynumber:",
    "個人番号",
    "マイナンバー",
    "保険証番号",
    "被保険者番号",
)

# Identifier SHAPES, guarded on digits rather than on \b so that a value written
# directly against Kanji or Kana is detected exactly as the same value written
# between ASCII spaces.
#
# Deliberately NOT matched: ICD-11 disease codes (letter-led, at most four
# characters), calendar dates (4-2-2), monetary caps (comma-grouped), ratios and
# percentages. Each of those must survive a medical assessment question intact.
_PATIENT_IDENTIFIER_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Individual number / long policy reference as a bare 10-12 digit run.
    ("identifier_digit_run", re.compile(r"(?<![0-9])\d{10,12}(?![0-9])")),
    # Individual number written 4-4-4, hyphenated or spaced. A date is 4-2-2 and
    # a monetary cap carries commas, so neither shape can match here.
    ("identifier_grouped", re.compile(r"(?<![0-9-])\d{4}[-\s]\d{4}[-\s]\d{4}(?![0-9-])")),
)

# Credential shapes the framework's own detector does not carry. Kept as an
# ADDITION to detect_credentials(), never as a replacement: a local set narrower
# than the framework's is a bypass, because the framework then raises inside the
# node wrapper and the node's own containment is discarded with the delta.
_EXTRA_CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "credential_assignment",
        re.compile(
            r"(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
)

# Characters the assessment renderer uses as structure. Neutralised in any echoed
# caller string so quoted text cannot be read as a heading, a list marker or a
# retrieved passage line. The underscore is in the set because a passage's source
# attribution is rendered in italics (`_(source: ...)_`), so a quoted string
# could otherwise present an invented attribution — the same forgery one line up.
_ECHO_STRUCTURE_RE = re.compile(r"[#*_`|\[\]【】「」]")

# The framework's redaction sentinel is template-external but arrives inside
# caller text after the platform's own personal-data masking has run. It is
# written with the same brackets the renderer uses, so stripping those would turn
# a masked value into the bare word MASKED — which reads as content rather than
# as a redaction. It is carried through the echo intact.
MASK_SENTINEL = "[MASKED]"
_ECHO_WHITESPACE_RE = re.compile(r"\s+")

# Rendered length cap for the echoed question. The question itself may be longer;
# what is quoted back into the assessment is bounded independently, so a long
# input cannot inflate the released output.
ECHO_MAX_CHARS = 200


def finite_int_in_range(value: Any, lo: int, hi: int) -> Optional[int]:
    """Parse *value* as a real, finite whole number inside ``[lo, hi]``; else None.

    Rejects booleans (``isinstance(True, int)`` is true in Python), non-numeric
    types, NaN / ±Infinity, and non-integral floats. Callers treat ``None`` as
    "declared but not usable" and refuse the request, so an invalid declared value
    can never silently disable the bound it feeds.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        return None
    parsed = int(number)
    return parsed if lo <= parsed <= hi else None


def is_inert_token(value: Any) -> bool:
    """True when *value* is a short identifier over the closed inert alphabet."""
    return isinstance(value, str) and bool(_INERT_TOKEN_RE.match(value))


def safe_field_label(name: Any, position: int) -> str:
    """Return a name safe to echo in a refusal, or a positional label instead."""
    if isinstance(name, str) and _SAFE_FIELD_NAME_RE.match(name):
        return name
    return f"field #{position}"


def strip_markup(text: str) -> str:
    """Remove inline markup and invisible characters, joining what they split."""
    return _MARKUP_TAG_RE.sub("", _INVISIBLE_RE.sub("", text))


def screen_text(text: Any) -> Optional[str]:
    """Return the name of the first override pattern *text* carries, else None.

    Screens the value twice: as received, so a control token is caught before a
    strip could remove it, and with markup removed, so a directive spliced across
    inline tags is caught after the strip reassembles it. A sanitiser that only
    strips is not a refusal — it converts a detectable token attack into
    undetectable plain text.
    """
    if not isinstance(text, str) or not text:
        return None
    for candidate in (text, strip_markup(text)):
        for name, pattern in _CONTROL_TOKEN_PATTERNS:
            if pattern.search(candidate):
                return name
        for name, pattern in _DIRECTIVE_PATTERNS:
            if pattern.search(candidate):
                return name
    return None


def screen_structure(value: Any, _depth: int = 0) -> Optional[str]:
    """Screen every string leaf AND every mapping key, depth-first.

    Applied to the PARSED request rather than to the raw body, so a directive
    hidden behind JSON ``\\u`` escapes — absent from the raw text, present in the
    parsed value — is still caught. Keys are screened because a hostile field name
    reaches the same audit and refusal paths a value does. Depth is bounded so a
    deeply nested body cannot exhaust the stack.
    """
    if _depth > 8:
        return "nesting_depth_exceeded"
    if isinstance(value, str):
        return screen_text(value)
    if isinstance(value, dict):
        for key, nested in value.items():
            found = screen_text(key) if isinstance(key, str) else None
            if found:
                return found
            found = screen_structure(nested, _depth + 1)
            if found:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = screen_structure(item, _depth + 1)
            if found:
                return found
    return None


def detect_patient_identifier(text: Any) -> Optional[str]:
    """Return the kind of the first patient identifier in *text*, else None.

    Only the KIND is returned — never the matched value — so a refusal can be
    audited without the identifier re-entering a log line. Detection is refusal
    input, not masking input: this template refuses a question carrying a patient
    identifier rather than answering a masked version of it, because the
    biomedical retrieval path must never receive one at all.
    """
    if not isinstance(text, str) or not text:
        return None
    lowered = text.lower()
    for marker in _PATIENT_IDENTIFIER_MARKERS:
        if marker.lower() in lowered:
            return "identifier_marker"
    for kind, pattern in _PATIENT_IDENTIFIER_SHAPES:
        if pattern.search(text):
            return kind
    return None


def neutralise_echo(text: Any, limit: int = ECHO_MAX_CHARS) -> str:
    """Return *text* rendered safe to quote inside the structured assessment.

    Three things happen, and each closes one way a quoted string could be read as
    part of the assessment's own structure:

    * every whitespace run — newlines included — collapses to one space, so the
      quote occupies exactly one line and cannot open a second that would read as
      a numbered retrieved passage;
    * the characters the renderer uses as structure are removed, so the quote
      cannot begin a heading, a list item or a source attribution — except inside
      the redaction sentinel, which must stay legible;
    * the result is truncated to *limit* characters, so the quote's contribution
      to the released assessment is bounded independently of the input length.
    """
    if not isinstance(text, str):
        return ""
    parts = [_ECHO_STRUCTURE_RE.sub(" ", part) for part in text.split(MASK_SENTINEL)]
    collapsed = _ECHO_WHITESPACE_RE.sub(" ", MASK_SENTINEL.join(parts)).strip()
    if len(collapsed) > limit:
        return collapsed[:limit] + "…"
    return collapsed


def detect_output_credentials(text: Any) -> Optional[str]:
    """Return the name of the first credential shape in *text*, else None.

    Delegates to the framework's own ``detect_credentials`` so the refusal set is
    exactly the block set the framework enforces one layer later. A narrower local
    set is a bypass, not a smaller net: the value passes this gate, the framework
    raises inside the node wrapper, and the wrapper discards the whole delta —
    including the fields this gate had just cleared.

    The extra patterns are additions the framework does not carry — an assignment
    line from a mis-pasted operations note is not a credential SHAPE, but it is
    exactly what such a note looks like. Delegating without them would make this
    gate NARROWER while looking like a tightening.
    """
    if not isinstance(text, str) or not text:
        return None
    findings = detect_credentials(text)
    if findings:
        return str(findings[0]["type"])
    for name, pattern in _EXTRA_CREDENTIAL_PATTERNS:
        if pattern.search(text):
            return name
    return None


class Service:
    """Domain service seam for INS-C2-051.

    The shipped pipeline retrieves offline over the biomedical knowledge base
    bundled with ``BioBERTRetrieveNode``, so this service is a documented no-op:
    ``fetch()`` returns an empty result set rather than raising. It is the wiring
    seam for a real on-device embedding model and vector index, which slots in
    without changing the node contracts that call it.
    """

    async def fetch(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch domain data for the given query.

        Returns ``{"results": []}``. The bundled knowledge base in
        ``BioBERTRetrieveNode`` is the shipped source of truth; an external
        retriever wires in here.
        """
        return {"results": []}
