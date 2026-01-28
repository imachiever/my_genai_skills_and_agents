---
name: engineering-practices
description: Reference material for prompt engineering and development standards. Use when designing agent prompts, choosing tooling, or needing specific templates. Complements prompt-engineer and standards agents.
---

# Engineering Practices

On-demand references for the `prompt-engineer` and `standards` agents.

## When to Load References

**Load `references/prompt-templates.md` when:**
- Designing new agent instructions
- Need proven prompt structures to adapt

**Load `references/tooling.md` when:**
- Setting up a new project
- Need current tooling recommendations with install commands

## Quick Reference (No Need to Load Files)

### Prompt Design Checklist
1. Identity → Context → Task → Tools → Constraints → Output Format
2. Include examples for ambiguous cases
3. Add explicit "don't know" instructions
4. Test with 10+ diverse inputs before deploying

### Tooling Quick Picks
- **Python**: `uv` + `ruff` + `pytest`
- **TypeScript**: `bun` or `pnpm` + `biome` + `vitest`
- **Containers**: Alpine or distroless base images
