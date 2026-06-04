"""LM Studio MCP integration specs for live tests."""

import json
import os

# Hugging Face MCP (remote). Override with LMSTUDIO_MCP_INTEGRATION_JSON if using mcp.json plugin IDs.
MCP_HF = {
    "type": "ephemeral_mcp",
    "server_label": "huggingface",
    "server_url": "https://huggingface.co/mcp",
}

# Alternative when HF is pre-configured in ~/.lmstudio/mcp.json:
# MCP_HF = {"type": "plugin", "id": "mcp/huggingface"}
# MCP_HF = "mcp/huggingface"


def load_mcp_hf_integration():
    """Return MCP HF integration from env JSON or the default ephemeral spec."""
    raw = os.getenv("LMSTUDIO_MCP_INTEGRATION_JSON")
    if not raw:
        return MCP_HF
    return json.loads(raw)
