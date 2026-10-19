"""MCP: pure unit tests (no HTTP). Redaction, role mapping, tool definitions, start-up safety checks."""
import pytest
from pydantic import SecretStr

from app.config import settings
from app.mcp_server import rbac, routes as mcp_routes
from app.mcp_server.sanitize import TRUNCATED, clean_text, mask_email_addresses, sanitize_json
from app.mcp_server.tools import TOOLS
from tests.mcp_support import ALL_TOOLS, TOOLS_BY_WHO


# ---------------------------------------------------------------- redaction
@pytest.mark.parametrize("raw,expected", [
    ("pay with 4111 1111 1111 1111 please", "pay with [REDACTED-CARD ****1111] please"),
    ("card 4111-1111-1111-1111.", "card [REDACTED-CARD ****1111]."),
    ("IBAN DE89 3704 0044 0532 0130 00 ok", "IBAN [REDACTED-IBAN] ok"),
    ("DE89370400440532013000", "[REDACTED-IBAN]"),
    ("ssn 123-45-6789", "ssn [REDACTED-SSN]"),
    ("order 1234567890123", "order 1234567890123"),            # long number that fails the Luhn check: left alone
    ("Ref AB12CDEFGHIJKLMNOP", "Ref AB12CDEFGHIJKLMNOP"),      # looks like an IBAN but fails mod-97
])
def test_sensitive_numbers_are_redacted_and_ordinary_text_is_not(raw, expected):
    assert clean_text(raw) == expected


def test_a_card_hidden_behind_other_digits_is_still_found_and_two_cards_are_both_masked():
    assert "4111 1111" not in clean_text("2026 10 17 4111 1111 1111 1111")
    assert clean_text("a 4111111111111111 b 5500 0000 0000 0004").count("REDACTED-CARD") == 2


def test_identifiers_can_be_kept_verbatim_and_emails_masked_on_request():
    assert clean_text("4111111111111111", redact=False) == "4111111111111111"
    assert clean_text("Billing <billing@acme.example>", mask_emails=True) == "Billing <b***@acme.example>"
    assert mask_email_addresses("no address here") == "no address here"


def test_hidden_characters_are_removed_and_long_text_is_truncated_and_none_stays_none():
    assert clean_text("a\u200bb\u202ec\x00d\x1be") == "abcde"             # zero-width, bidi override, NUL, ESC
    assert clean_text("x\n\n\n\n\ny") == "x\n\ny"
    assert clean_text("a" * 1000, 50) == "a" * 50 + TRUNCATED
    assert clean_text(None) is None


def test_sanitize_json_cleans_strings_drops_nan_and_bounds_size_and_depth():
    out = sanitize_json({"a": float("nan"), None: 1, "n": {"d": [{"x": "4111 1111 1111 1111"}]}, "l": list(range(500))})
    assert out["a"] is None and out["unknown"] == 1 and "REDACTED-CARD" in out["n"]["d"][0]["x"] and len(out["l"]) == 100
    deep = cur = {}
    for _ in range(20):
        cur["k"] = {}
        cur = cur["k"]
    assert "[truncated]" in str(sanitize_json(deep))


# ---------------------------------------------------------------- roles
def test_role_labels_and_deny_by_default():
    assert [rbac.mcp_role(r, "bearer") for r in ("viewer", "reviewer", "admin")] == ["user", "user", "super_admin"]
    assert rbac.mcp_role("service", rbac.SYSTEM_VIA) == "system"
    assert rbac.mcp_role("service", "api_key") is None                   # the ingest key is not an MCP credential
    assert rbac.mcp_role("somebody-new", "bearer") is None
    assert not rbac.is_allowed("service", "api_key", {"service", "admin"})
    assert not rbac.is_allowed("viewer", "bearer", {"admin"})


def test_tool_roles_produce_exactly_the_documented_matrix():
    assert {t.name for t in TOOLS} == ALL_TOOLS                          # adding a tool forces updating the matrix and docs
    principals = {"view1": ("viewer", "bearer"), "rev1": ("reviewer", "bearer"), "admin1": ("admin", "bearer"),
                  "system": ("service", rbac.SYSTEM_VIA)}
    for who, (role, via) in principals.items():
        allowed = {t.name for t in TOOLS if rbac.is_allowed(role, via, t.roles)}
        assert allowed == TOOLS_BY_WHO[who], who


# ---------------------------------------------------------------- tool definitions
def _has_title(node) -> bool:
    if isinstance(node, dict):
        return any((k == "title" and isinstance(v, str)) or _has_title(v) for k, v in node.items())
    if isinstance(node, list):
        return any(_has_title(v) for v in node)
    return False


def test_every_tool_is_read_only_strictly_typed_and_documented():
    names = [t.name for t in TOOLS]
    assert len(names) == len(set(names))
    for t in TOOLS:
        d = t.descriptor()
        assert d["description"] and d["title"]
        assert d["annotations"]["readOnlyHint"] is True and d["annotations"]["destructiveHint"] is False
        schema = d["inputSchema"]
        assert schema["type"] == "object" and schema.get("additionalProperties") is False, t.name
        assert not _has_title(schema), t.name
        assert t.roles <= {"viewer", "reviewer", "admin", "service"} and "admin" in t.roles
        assert not t.name.startswith(("approve", "reject", "create", "update", "delete", "review", "ingest", "set_"))


# ---------------------------------------------------------------- start-up safety
def _configure(monkeypatch, **changes):
    base = {"mcp_enabled": True, "auth_enabled": True, "demo_mode": False,
            "mcp_system_key": SecretStr(""), "api_key": SecretStr("")}
    for k, v in {**base, **changes}.items():
        monkeypatch.setattr(settings, k, v)


@pytest.mark.parametrize("changes,fragment", [
    ({"auth_enabled": False}, "AUTH_ENABLED"),
    ({"demo_mode": True}, "DEMO_MODE"),
    ({"mcp_system_key": SecretStr("too-short")}, "at least 32"),
    ({"mcp_system_key": SecretStr("k" * 40), "api_key": SecretStr("k" * 40)}, "must differ"),
])
def test_mcp_refuses_to_start_in_an_unsafe_configuration(monkeypatch, changes, fragment):
    _configure(monkeypatch, **changes)
    with pytest.raises(RuntimeError, match=fragment):
        mcp_routes.startup_checks()


def test_safe_or_disabled_configurations_start(monkeypatch):
    _configure(monkeypatch, mcp_system_key=SecretStr("k" * 40))
    mcp_routes.startup_checks()
    _configure(monkeypatch, mcp_enabled=False, auth_enabled=False)       # disabled: nothing to check
    mcp_routes.startup_checks()
