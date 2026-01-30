#!/bin/bash
# Format staged Python files before commit - low cost (only on commit)
TOOL_INPUT=$(cat)
COMMAND=$(echo "$TOOL_INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('command',''))" 2>/dev/null)

if [[ "$COMMAND" == *"git commit"* ]]; then
    # Get staged Python files and format them
    staged_py=$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null | grep '\.py$' || true)
    if [[ -n "$staged_py" ]]; then
        echo "$staged_py" | xargs -I{} ruff format {} 2>/dev/null || true
        echo "$staged_py" | xargs -I{} ruff check --fix {} 2>/dev/null || true
        # Re-stage formatted files
        echo "$staged_py" | xargs git add 2>/dev/null || true
    fi
fi
exit 0
