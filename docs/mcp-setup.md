# MCP Server Setup

Expose Sovereign Edge experts as tools in Claude Desktop / Claude Code.

## Tools available

| Tool | Description |
|------|-------------|
| `ask_expert` | Dispatch a query to: spiritual, career, intelligence, creative, goals |
| `get_memory` | Search recent episodic memory entries |
| `get_skills` | Top skill patterns for an intent class |
| `get_stats` | Today's usage and cost stats |

## Claude Desktop config

Add to `~/.config/claude/claude_desktop_config.json` (Linux/Mac) or
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "sovereign-edge": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/home/omnipotence/sovereign-edge",
      "env": {
        "PYTHONPATH": "/home/omnipotence/sovereign-edge/.venv/lib/python3.11/site-packages"
      }
    }
  }
}
```

Or using the venv directly:

```json
{
  "mcpServers": {
    "sovereign-edge": {
      "command": "/home/omnipotence/sovereign-edge/.venv/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/home/omnipotence/sovereign-edge"
    }
  }
}
```

## Claude Code usage

```bash
# From the Sovereign Edge repo root
python -m mcp_server.server

# Or install and run as CLI
uv pip install -e services/mcp
sovereign-edge-mcp
```

## Docker (MCP profile)

```bash
docker compose --profile mcp up -d mcp
docker compose logs -f mcp
```

The container exposes port 3000 for SSE transport (daemon mode).

## Verify

After configuring Claude Desktop, restart it and check the MCP tools panel.
You should see `sovereign-edge` with 4 tools listed.

Test from Claude Desktop:
```
ask_expert("career", "What ML engineer roles are available in DFW?")
get_stats()
```
