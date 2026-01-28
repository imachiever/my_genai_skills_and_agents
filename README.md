# Claude Configuration

Personal agents and skills for Claude Code.

## Structure

```
claude-config/
├── agents/         # Custom agents (always-loaded reasoning)
├── skills/         # Custom skills (on-demand knowledge)
└── [future]/       # Hooks, settings, etc.
```

## Setup

Symlink into Claude's config directory:

```bash
# Backup existing (if any)
mv ~/.claude/agents ~/.claude/agents.bak
mv ~/.claude/skills ~/.claude/skills.bak

# Symlink
ln -s ~/claude-config/agents ~/.claude/agents
ln -s ~/claude-config/skills ~/.claude/skills
```

## Agents

| Agent | Purpose |
|-------|---------|
| `planner` | Architecture-first planning |
| `implementer` | Executes approved plans |
| `prompt-engineer` | Designs agent instructions |
| `standards` | Enforces best practices |
| `orchestration` | Multi-agent workflows |
| `api-integration` | External API work |
| `production` | Observability, security |
| `tester` | Test strategy and implementation |
| `documenter` | Technical writing |

## Skills

| Skill | Purpose |
|-------|---------|
| `engineering-practices` | Prompt templates, tooling reference |
| `agent-harness-patterns` | Ralph-loop, long-running agents |
| `google-adk-enterprise` | Google ADK patterns |
| `mcp-builder` | MCP server development |
| `skill-creator` | Creating new skills |
| `pdf`, `docx`, `xlsx`, `pptx` | Document manipulation |
| `frontend-design` | UI/web development |
| `webapp-testing` | Playwright E2E testing |
