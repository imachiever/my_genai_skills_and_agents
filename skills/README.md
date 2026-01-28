# My Claude Skills

Personal collection of Claude Code skills for enhanced development workflows.

## Skills (14 Total)

### 🚀 Agent Development

#### 1. google-adk-enterprise
Enterprise-grade Google ADK agent development with agentic best practices.

**Use when:** Building multi-agent systems, migrating from LangGraph to ADK, implementing quality gates, performance optimization, knowledge graphs, or requiring SOLID principles for production agents.

**Features:**
- ✅ Correct ADK 1.20.0 API (fixed all common mistakes)
- ✅ SOLID principles for agent architecture
- ✅ Session & context management (PostgreSQL, Memory Bank)
- ✅ Performance optimization (parallel execution, caching, cost-aware routing)
- ✅ Knowledge graphs & structured memory (temporal graphs, entity extraction)
- ✅ UX design patterns (Ritz-Carlton concierge experience)
- ✅ Multi-agent orchestration (8 essential patterns)
- ✅ Quality gates & testing strategies
- ✅ Production deployment checklists
- ✅ LangGraph migration guide

**Version:** 2.0.0 | **Source:** Custom

---

#### 2. agent-harness-patterns
Effective harness patterns for long-running autonomous agents.

**Use when:** Building Ralph-Loop style continuous development, agent evaluation frameworks, or multi-session agent workflows.

**Features:**
- ✅ Initializer + Coding Agent architecture
- ✅ Feature list management (200+ granular features)
- ✅ Progress tracking & session management
- ✅ Testing strategies (E2E with Playwright)
- ✅ Ralph-Loop autonomous development patterns
- ✅ Harbor/Bloom evaluation frameworks

**Version:** 1.0.0 | **Source:** Custom (based on Anthropic research)

---

#### 3. mcp-builder
Guide for creating high-quality MCP servers.

**Use when:** Building MCP servers to integrate external APIs or services (Python/FastMCP or Node/TypeScript).

**Version:** Official | **Source:** Anthropic

---

### 📄 Document Skills (Official Anthropic)

#### 4. docx
Comprehensive Word document creation, editing, and analysis.

**Features:**
- Create/edit .docx files with tracked changes
- Comments and formatting preservation
- Text extraction and analysis

**Use when:** Working with professional documents, client deliverables, reports.

**Version:** Official | **Source:** Anthropic

---

#### 5. xlsx
Excel spreadsheet creation, editing, and analysis.

**Features:**
- Formulas, formatting, data analysis
- Chart generation and visualization
- Formula recalculation

**Use when:** Creating financial models, data analysis, client spreadsheets.

**Version:** Official | **Source:** Anthropic

---

#### 6. pptx
PowerPoint presentation creation and editing.

**Features:**
- Layouts, templates, automated slides
- Charts and visual elements
- HTML to PowerPoint conversion

**Use when:** Building client presentations, pitch decks, reports.

**Version:** Official | **Source:** Anthropic

---

#### 7. pdf
Comprehensive PDF manipulation toolkit.

**Features:**
- Extract text and tables
- Merge/split documents
- Fill forms and handle annotations

**Use when:** Processing PDFs, extracting data, filling forms.

**Version:** Official | **Source:** Anthropic

---

### 💻 Development Skills

#### 8. frontend-design
Create distinctive, production-grade frontend interfaces.

**Use when:** Building web components, pages, or applications. Generates creative, polished UI that avoids generic AI aesthetics.

**Features:**
- React + Tailwind CSS optimization
- Bold design decisions
- Production-grade code quality

**Version:** Official | **Source:** Anthropic

---

#### 9. webapp-testing
Test local web applications using Playwright.

**Use when:** Verifying frontend functionality, debugging UI behavior, automated testing.

**Features:**
- Browser automation testing
- Screenshot capture
- Console log viewing

**Version:** Official | **Source:** Anthropic

---

### 📝 Communication & Branding

#### 10. internal-comms
Write professional internal communications.

**Use when:** Creating status reports, newsletters, FAQs, incident reports, project updates.

**Features:**
- Company-standard formats
- Multiple communication types
- Professional tone

**Version:** Official | **Source:** Anthropic

---

#### 11. brand-guidelines
Apply brand colors and typography to artifacts.

**Use when:** Creating branded documents, presentations, or web content.

**Features:**
- Anthropic brand colors
- Typography standards
- Visual consistency

**Version:** Official | **Source:** Anthropic

---

### 🛠️ Utility Skills

#### 12. explain-code
Explains code with visual diagrams and analogies.

**Use when:** Teaching, onboarding, or when someone asks "how does this work?"

**Version:** 1.0.0 | **Source:** Anthropic

---

#### 13. skill-creator
Interactive guide for creating new skills.

**Use when:** Building custom skills with specialized knowledge or workflows.

**Version:** 1.0.0 | **Source:** Anthropic

## Installation

1. Clone this repo to your Claude skills directory:
```bash
cd ~/.claude
mv skills skills.backup  # backup existing if any
git clone git@github.com:imachiever/my-claude-skills.git skills
```

2. Restart Claude Code or run `/reload` to load the new skills.

## Usage

Invoke skills with the `/` command:

**Agent Development:**
- `/google-adk-enterprise` - ADK development with SOLID principles
- `/agent-harness-patterns` - Long-running autonomous agents
- `/mcp-builder` - Build MCP servers

**Documents:**
- `/docx` - Create/edit Word documents
- `/xlsx` - Create/edit Excel spreadsheets
- `/pptx` - Create/edit PowerPoint presentations
- `/pdf` - Manipulate PDF files

**Development:**
- `/frontend-design` - Production-grade UI design
- `/webapp-testing` - Playwright browser testing

**Communication:**
- `/internal-comms` - Professional communications
- `/brand-guidelines` - Apply brand standards

**Utilities:**
- `/explain-code` - Code explanations with diagrams
- `/skill-creator` - Create new skills

**Quick Reference:**
```bash
/skills list           # List all available skills
/google-adk-enterprise # Most comprehensive - ADK development
/agent-harness-patterns # Autonomous agent patterns
/docx                  # Word documents
/xlsx                  # Excel spreadsheets
```

## Development

To add a new skill:
1. Create a directory: `mkdir my-new-skill`
2. Add `SKILL.md` with frontmatter and content
3. Test the skill with `/my-new-skill`
4. Commit and push

## Contributions

These are personal skills maintained by [@imachiever](https://github.com/imachiever). Feel free to fork and customize for your own use!

## License

MIT License - Feel free to use and modify for your own purposes.

---

**Maintained by:** Rajat Bhatia  
**Last Updated:** January 25, 2026  
**Claude Code Version:** Latest
