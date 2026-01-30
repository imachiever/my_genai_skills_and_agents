#!/usr/bin/env python3
"""
Safety gate - blocks dangerous commands before execution.
Exit code 2 = block and feed error to Claude
"""
import sys
import json
import re

# Read tool input from stdin
tool_input = json.loads(sys.stdin.read())
command = tool_input.get("command", "")

# Dangerous patterns to block
dangerous = [
    (r'rm\s+(-[rf]+\s+)*/', "Refusing to rm from root"),
    (r'rm\s+-[rf]*\s+~', "Refusing to rm home directory"),
    (r'chmod\s+777', "chmod 777 is insecure"),
    (r'>\s*/etc/', "Refusing to write to /etc"),
    (r'sudo\s+rm', "Refusing sudo rm"),
]

for pattern, msg in dangerous:
    if re.search(pattern, command, re.IGNORECASE):
        print(msg, file=sys.stderr)
        sys.exit(2)

# Protected files - block edits
protected = ['.env', 'credentials', '.secrets', 'id_rsa', '.pem']
for p in protected:
    if p in command and ('cat' in command or 'rm' in command or '>' in command):
        print(f"Protected file pattern detected: {p}", file=sys.stderr)
        sys.exit(2)

sys.exit(0)
