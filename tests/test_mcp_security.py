"""Small unit tests for MCP role policy; these tests do not require a live MySQL DB."""

import os
import unittest
from unittest.mock import patch

from mcp_layer.security import (
    ADMIN_TOOLS,
    ALL_TOOLS,
    ROLE_ADMIN,
    ROLE_SUPER_ADMIN,
    ROLE_SYSTEM,
    ROLE_USER,
    SYSTEM_TOOLS,
    USER_TOOLS,
    can_access_tool,
    role_for_bearer_token,
    normalize_session_role,
)
from mcp_layer.tools import TOOL_CATALOG, invoke_tool, list_tools_for_role
from mcp_layer.usage import _price, pricing_rates


class MCPRoleTests(unittest.TestCase):
    def test_expected_tool_groups_and_total(self):
        self.assertEqual(len(USER_TOOLS), 5)
        self.assertEqual(len(ADMIN_TOOLS), 3)
        self.assertEqual(len(SYSTEM_TOOLS), 2)
        self.assertEqual(len(ALL_TOOLS), 10)
        self.assertEqual(len(TOOL_CATALOG), 10)

    def test_user_cannot_call_admin_or_system_tools(self):
        self.assertTrue(can_access_tool(ROLE_USER, "user_search_doctors"))
        self.assertFalse(can_access_tool(ROLE_USER, "admin_approve_doctor"))
        self.assertFalse(can_access_tool(ROLE_USER, "system_database_health"))

    def test_admin_inherits_user_tools_but_not_system_tools(self):
        self.assertTrue(can_access_tool(ROLE_ADMIN, "user_search_medicines"))
        self.assertTrue(can_access_tool(ROLE_ADMIN, "admin_clinic_metrics"))
        self.assertFalse(can_access_tool(ROLE_ADMIN, "system_database_health"))

    def test_system_is_restricted_to_system_tools(self):
        self.assertTrue(can_access_tool(ROLE_SYSTEM, "system_mcp_status"))
        self.assertFalse(can_access_tool(ROLE_SYSTEM, "user_search_doctors"))
        self.assertFalse(can_access_tool(ROLE_SYSTEM, "admin_approve_doctor"))

    def test_super_admin_can_access_every_tool(self):
        for tool in ALL_TOOLS:
            self.assertTrue(can_access_tool(ROLE_SUPER_ADMIN, tool))

    def test_session_role_mapping(self):
        self.assertEqual(normalize_session_role("PATIENT"), ROLE_USER)
        self.assertEqual(normalize_session_role("DOCTOR"), ROLE_USER)
        self.assertEqual(normalize_session_role("ADMIN"), ROLE_ADMIN)
        self.assertEqual(normalize_session_role("SUPER_ADMIN"), ROLE_SUPER_ADMIN)
        self.assertIsNone(normalize_session_role("GUEST"))

    def test_token_lookup_fails_closed_on_duplicate_tokens(self):
        with patch.dict(os.environ, {
            "MCP_USER_TOKEN": "duplicate",
            "MCP_ADMIN_TOKEN": "duplicate",
            "MCP_SYSTEM_TOKEN": "system-secret",
            "MCP_SUPER_ADMIN_TOKEN": "super-secret",
        }, clear=False):
            self.assertIsNone(role_for_bearer_token("duplicate"))
            self.assertEqual(role_for_bearer_token("system-secret"), ROLE_SYSTEM)
            self.assertIsNone(role_for_bearer_token("unknown"))

    def test_tool_catalog_hides_restricted_tools(self):
        self.assertEqual(len(list_tools_for_role(ROLE_USER)), 5)
        self.assertEqual(len(list_tools_for_role(ROLE_ADMIN)), 8)
        self.assertEqual(len(list_tools_for_role(ROLE_SYSTEM)), 2)
        self.assertEqual(len(list_tools_for_role(ROLE_SUPER_ADMIN)), 10)

    def test_database_outage_blocks_data_tool_before_handler_runs(self):
        with patch("mcp_layer.tools.check_database", return_value={
            "available": False,
            "message": "Clinic database is unavailable.",
        }):
            result = invoke_tool("user_list_specializations", {}, role=ROLE_USER)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "database_unavailable")
        self.assertEqual(result["source"], "database_guard")

    def test_pricing_uses_selected_model_rates_and_cached_tokens(self):
        env = {
            "GROQ_CHAT_MODEL": "openai/gpt-oss-120b",
            "GROQ_VISION_MODEL": "qwen/qwen3.8-27b",
            "LLM_INPUT_PRICE_PER_1M_USD": "0.15",
            "LLM_CACHED_INPUT_PRICE_PER_1M_USD": "0.075",
            "LLM_OUTPUT_PRICE_PER_1M_USD": "0.60",
            "GROQ_VISION_INPUT_PRICE_PER_1M_USD": "",
            "GROQ_VISION_CACHED_INPUT_PRICE_PER_1M_USD": "",
            "GROQ_VISION_OUTPUT_PRICE_PER_1M_USD": "",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(pricing_rates()["model"], "openai/gpt-oss-120b")
            cost = _price({
                "model": "openai/gpt-oss-120b",
                "prompt_tokens": 100,
                "cached_input_tokens": 10,
                "completion_tokens": 20,
            })
        self.assertAlmostEqual(cost, 0.00002625, places=10)

    def test_unknown_or_unconfigured_model_is_not_silently_priced(self):
        env = {
            "GROQ_CHAT_MODEL": "openai/gpt-oss-120b",
            "GROQ_VISION_MODEL": "qwen/qwen3.8-27b",
            "GROQ_VISION_INPUT_PRICE_PER_1M_USD": "",
            "GROQ_VISION_CACHED_INPUT_PRICE_PER_1M_USD": "",
            "GROQ_VISION_OUTPUT_PRICE_PER_1M_USD": "",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertIsNone(_price({
                "model": "unconfigured/alternate-model",
                "prompt_tokens": 100,
                "completion_tokens": 20,
            }))


if __name__ == "__main__":
    unittest.main()
