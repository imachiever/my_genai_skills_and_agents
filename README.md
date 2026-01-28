# Claude Code Configuration

Personal agents and skills for Claude Code, following the **"Smart Agents, Dumb Framework"** philosophy.

## Philosophy

```
Agents = Judgment (how to think)
Skills = Knowledge (what to reference)
```

- **Agents** are always-loaded reasoning specialists. They make decisions, plan approaches, and execute with judgment.
- **Skills** are on-demand knowledge packs. They load when relevant and provide templates, patterns, and domain expertise.

This separation is MECE (Mutually Exclusive, Collectively Exhaustive) - no duplication between agents and skills.

---

## Structure

```
~/.claude/
├── agents/                    # Specialist agents (9 total)
│   ├── planner.md            # Architecture-first planning
│   ├── implementer.md        # Executes approved plans
│   ├── prompt-engineer.md    # Designs agent instructions
│   ├── standards.md          # Enforces best practices
│   ├── orchestration.md      # Multi-agent workflows
│   ├── api-integration.md    # External API work
│   ├── production.md         # Observability, security, DevOps
│   ├── tester.md             # Test strategy and implementation
│   └── documenter.md         # Technical writing
│
├── skills/                    # On-demand knowledge (14 owned + external)
│   ├── engineering-practices/ # Prompt templates, tooling reference
│   ├── agent-harness-patterns/# Ralph-loop, autonomous agents
│   ├── google-adk-enterprise/ # Google ADK patterns
│   ├── mcp-builder/          # MCP server development
│   ├── skill-creator/        # Meta: creating new skills
│   ├── webapp-testing/       # Playwright E2E testing
│   ├── frontend-design/      # UI/web development
│   ├── pdf/                  # PDF manipulation
│   ├── docx/                 # Word document handling
│   ├── xlsx/                 # Spreadsheet operations
│   ├── pptx/                 # Presentation creation
│   ├── explain-code/         # Code explanations with diagrams
│   ├── brand-guidelines/     # Anthropic brand standards
│   └── internal-comms/       # Internal communication templates
│
├── .gitignore                # Ignores Claude Code local data
└── README.md                 # This file
```

---

## Agents

| Agent | Model | Description |
|-------|-------|-------------|
| **planner** | opus | Architecture-first planning. Creates implementation plans before any code is written. Never executes - only designs. |
| **implementer** | opus | Executes approved plans. Writes production code following established patterns. Works incrementally with verification. |
| **prompt-engineer** | opus | Expert in prompt design for agentic systems. Optimizes LLM instructions, decides when to use prompts vs code. |
| **standards** | sonnet | Enforces best practices and conventions. Knows modern tooling (uv, ruff, etc.) and anti-patterns to avoid. |
| **orchestration** | opus | Expert in multi-agent coordination, workflow design, state management, and task routing. |
| **api-integration** | sonnet | Expert in API clients, HTTP integrations, authentication, retry logic, and circuit breakers. |
| **production** | sonnet | Expert in observability, security, performance, and deployment. DevOps and production hardening. |
| **tester** | sonnet | Expert in test strategy, TDD, unit/integration/E2E testing, and quality assurance. |
| **documenter** | sonnet | Expert in technical writing, API documentation, architecture docs, and runbooks. |

---

## Skills

| Skill | Description |
|-------|-------------|
| **engineering-practices** | Reference material for prompt engineering and development standards. Complements prompt-engineer and standards agents. |
| **agent-harness-patterns** | Ralph-loop patterns for long-running autonomous agents with progress.txt and features.json. |
| **google-adk-enterprise** | Google Agent Development Kit patterns - SequentialAgent, ParallelAgent, LoopAgent. |
| **mcp-builder** | Guide for creating MCP servers that enable LLMs to interact with external services. |
| **skill-creator** | Meta-skill for creating new skills that extend Claude's capabilities. |
| **webapp-testing** | Playwright toolkit for E2E testing, screenshots, and browser automation. |
| **frontend-design** | Production-grade frontend interfaces with high design quality. |
| **pdf** | PDF manipulation - extract text/tables, create, merge, split, fill forms. |
| **docx** | Word documents - create, edit, tracked changes, comments, formatting. |
| **xlsx** | Spreadsheets - formulas, formatting, data analysis, visualization. |
| **pptx** | Presentations - create, edit, layouts, speaker notes. |
| **explain-code** | Code explanations with visual diagrams and analogies. |
| **brand-guidelines** | Anthropic's official brand colors and typography. |
| **internal-comms** | Templates for status reports, leadership updates, newsletters. |

---

## When to Use What

### Use ONLY Agent When:
- Task requires judgment or planning
- Implementation work with decisions to make
- No specialized domain knowledge needed
- Agent already has patterns inline

### Use ONLY Skill When:
- Task is procedural (follow steps)
- Specialized file format (PDF, XLSX, PPTX, DOCX)
- Need reference material, not reasoning
- "How do I..." questions with known answers

### Use Both When:
- Agent needs domain knowledge from skill
- Complex task with specialized + reasoning needs
- Building something that matches a skill's domain
- Multi-step workflow crossing domains

---

## Scenario Examples

| Scenario | Skill | Agent | Why |
|----------|-------|-------|-----|
| "Set up a new Python project" | - | standards | Need opinionated guidance |
| "Create an MCP server for GitHub" | mcp-builder | api-integration | Skill has MCP patterns, agent handles HTTP/auth |
| "Design agent instructions for a support bot" | engineering-practices | prompt-engineer | Agent designs, skill provides templates |
| "Add retry logic to this API call" | - | api-integration | Pure implementation |
| "Generate a PDF report" | pdf | implementer | Skill has PDF manipulation, agent writes code |
| "Build a multi-agent workflow" | google-adk-enterprise | orchestration | Skill has ADK patterns, agent designs workflow |
| "Run E2E tests and fix failures" | webapp-testing | tester | Skill has Playwright patterns, agent runs tests |
| "Explain how this code works" | explain-code | - | Teaching task, no implementation |

---

## Common Workflows

### For Long-Running Autonomous Development
```
1. /agent-harness-patterns  → Sets up features.json, progress.txt
2. planner agent            → Creates implementation plan
3. implementer agent        → Executes plan, commits
4. tester agent             → Adds tests
5. (loop until features complete)
```

### For Building Agent Systems (ADK, MCP)
```
1. /google-adk-enterprise   → Load ADK patterns
2. prompt-engineer agent    → Design agent instructions
3. orchestration agent      → Design workflow structure
4. /mcp-builder             → If building MCP tools
5. api-integration agent    → Implement external API clients
6. standards agent          → Validate tooling choices
```

### For New Project Setup
```
1. standards agent          → Recommend tooling (uv, ruff, etc.)
2. planner agent            → Design architecture
3. implementer agent        → Scaffold project
4. /skill-creator           → If creating project-specific skills
```

---

## Setup on New Machine

```bash
# Clone directly into ~/.claude
cd ~
git clone git@github.com-personal:imachiever/my_genai_skills_and_agents.git temp-clone
mv temp-clone/agents ~/.claude/
mv temp-clone/skills ~/.claude/
mv temp-clone/.git ~/.claude/
cp temp-clone/.gitignore ~/.claude/
cp temp-clone/README.md ~/.claude/
rm -rf temp-clone

# Verify
cd ~/.claude && git status
```

Or if ~/.claude doesn't exist yet:
```bash
git clone git@github.com-personal:imachiever/my_genai_skills_and_agents.git ~/.claude
```

---

## Updating

```bash
cd ~/.claude

# Pull latest
git pull

# After making changes
git add agents/ skills/ README.md
git commit -m "Update: description of changes"
git push
```

---

## How Claude Selects

```
User Query
    │
    ├─► Skill auto-loads if description matches
    │   (pdf, mcp-builder, google-adk-enterprise, etc.)
    │
    └─► Claude decides if agent needed:
        │
        ├─► Planning?   → planner
        ├─► Prompts?    → prompt-engineer
        ├─► Standards?  → standards
        ├─► APIs?       → api-integration
        ├─► Workflow?   → orchestration
        ├─► Security?   → production
        ├─► Tests?      → tester
        ├─► Docs?       → documenter
        └─► Code?       → implementer
```

**Key insight:** Skills load passively when relevant; agents are spawned actively when reasoning is needed.
