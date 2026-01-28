# Tooling Reference

Current recommendations with setup commands. Update as ecosystem evolves.

---

## Python (2025-2026)

### Package Management
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# New project
uv init myproject && cd myproject

# Add dependencies
uv add fastapi httpx pydantic

# Add dev dependencies
uv add --dev pytest ruff mypy

# Run commands
uv run python main.py
uv run pytest
```

### Linting & Formatting
```bash
# Single tool for lint + format
uv add --dev ruff

# pyproject.toml
[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W", "UP"]

# Run
uv run ruff check .
uv run ruff format .
```

### Type Checking
```bash
# mypy (mature) or pyright (faster)
uv add --dev mypy

# pyproject.toml
[tool.mypy]
strict = true
```

### Testing
```bash
uv add --dev pytest pytest-asyncio pytest-cov

# Run
uv run pytest -v
uv run pytest --cov=src
```

---

## TypeScript/JavaScript (2025-2026)

### Runtime & Package Manager
```bash
# Bun (fastest, all-in-one)
curl -fsSL https://bun.sh/install | bash
bun init
bun add hono
bun run index.ts

# Or pnpm (if Node required)
npm install -g pnpm
pnpm init
pnpm add express
```

### Linting & Formatting
```bash
# Biome (replaces eslint + prettier)
bun add -d @biomejs/biome
bunx biome init

# biome.json created, then:
bunx biome check .
bunx biome format . --write
```

### Testing
```bash
# Vitest (fast, vite-native)
bun add -d vitest
bunx vitest run
```

---

## Containers

### Base Images (smallest to largest)
```dockerfile
# Distroless (most secure, ~2MB)
FROM gcr.io/distroless/python3

# Alpine (small, has shell, ~5MB)
FROM python:3.12-alpine

# Slim (larger but compatible, ~150MB)
FROM python:3.12-slim
```

### Multi-Stage Build
```dockerfile
FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml .
RUN uv pip install --system --no-cache .

FROM python:3.12-slim
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY . .
USER nobody
CMD ["python", "-m", "myapp"]
```

---

## CI/CD

### GitHub Actions (Python)
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run pytest
```

### GitHub Actions (TypeScript)
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: oven-sh/setup-bun@v2
      - run: bun install
      - run: bun run biome check .
      - run: bun test
```

---

## Quick Decision Matrix

| Need | Tool | Why |
|------|------|-----|
| Python packages | uv | 10-100x faster than pip |
| Python lint+format | ruff | Single tool, fast |
| Python types | mypy | Mature, strict mode |
| JS/TS runtime | bun | Fastest, batteries included |
| JS/TS lint+format | biome | Replaces eslint+prettier |
| Testing (Python) | pytest | Best ecosystem |
| Testing (JS/TS) | vitest | Fast, modern |
| HTTP client (Python) | httpx | Async-native |
| HTTP client (JS/TS) | built-in fetch | No dependency needed |
| Web framework (Python) | FastAPI | Modern, typed |
| Web framework (JS/TS) | Hono | Fast, works everywhere |
| Database | PostgreSQL | Handles everything |
| Cache | Redis | Simple, proven |
| Container base | Alpine | Small, has shell |
