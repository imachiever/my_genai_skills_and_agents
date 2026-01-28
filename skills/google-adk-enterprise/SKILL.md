---
name: google-adk-enterprise
description: |
  Enterprise-grade Google ADK agent development with agentic best practices.
  Use when: building multi-agent systems, migrating from LangGraph to ADK,
  implementing quality gates, performance optimization, knowledge graphs, or requiring
  SOLID principles for production agents. Includes UX patterns, session management,
  and conversational design.
---

# Google ADK Enterprise Agent Development

## Overview
This skill provides enterprise-grade guidance for building production AI agents using Google's Agent Development Kit (ADK). It emphasizes **SOLID principles**, **agentic best practices**, **performance optimization**, **UX design patterns**, and **quality-first development** suitable for consulting and enterprise environments.

**Version:** 2.1.0 (Updated January 2026 - with real-world production learnings)
**ADK Version:** google-adk==1.20.0
**Python:** 3.11+

### 🚨 CRITICAL: Follow ADK Patterns

**GOLDEN RULE**: Always check ADK examples and documentation BEFORE implementing custom solutions.

- ✅ **DO**: Work with ADK's architecture
- ✅ **DO**: Put logic in LLM instructions (smart agents, dumb framework)
- ✅ **DO**: Use standard `SequentialAgent` and `ParallelAgent`
- ✅ **DO**: Manage state via conversation history and `output_key`
- ❌ **DON'T**: Create custom agents that override `_run_async_impl()`
- ❌ **DON'T**: Try to access `.state` in agent implementations (only available in callbacks!)
- ❌ **DON'T**: Fight the framework with custom routing logic

**This skill now includes real-world production learnings** from building enterprise agents, including common mistakes and why they don't work with ADK's architecture.

---

## Table of Contents

### I. Foundation
1. [Quick Start & API Reference](#quick-start--api-reference)
2. [SOLID Principles for Agents](#solid-principles-for-agents)
3. [Core Agent Types & Patterns](#core-agent-types--patterns)

### II. Advanced Features
4. [Session & Context Management](#session--context-management)
5. [Knowledge Graphs & Structured Memory](#knowledge-graphs--structured-memory)
6. [Performance & Budget Optimization](#performance--budget-optimization)

### III. User Experience
7. [UX Design Patterns (Concierge Experience)](#ux-design-patterns-concierge-experience)
8. [Multi-Agent Orchestration Patterns](#multi-agent-orchestration-patterns)

### IV. Production Deployment
9. [Quality Gates & Approval Workflows](#quality-gates--approval-workflows)
10. [Testing Strategies](#testing-strategies)
11. [Observability & Monitoring](#observability--monitoring)
12. [LangGraph → ADK Migration](#langgraph--adk-migration)
13. [Enterprise Deployment Checklist](#enterprise-deployment-checklist)

### V. Reference
14. [Real-World Examples](#real-world-examples)
15. [Troubleshooting & Common Pitfalls](#troubleshooting--common-pitfalls)

---

## Quick Start & API Reference

### Installation
```bash
pip install google-adk==1.20.0
# For production deployments
pip install google-cloud-aiplatform[adk,agent_engines]
```

### Minimal Working Agent
```python
from google.adk.agents import Agent
from google.adk.tools import google_search

agent = Agent(
    name="researcher",
    model="gemini-2.0-flash",
    instruction="You are a research assistant. Be concise and cite sources.",
    tools=[google_search],
    description="Researches topics using web search"
)

result = agent.run("What is SOLID?")
print(result)
```

### Core API Patterns

#### 1. Agent Types (Correct Imports)
```python
# ✅ CORRECT - Always use these imports
from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.tools import google_search
from google.adk.sessions import InMemorySessionService, Session
from google.adk.models.lite_llm import LiteLlm

# ❌ WRONG - DO NOT USE
# from google import adk  # This does not exist
```

#### 2. Agent Constructor Parameters
```python
agent = Agent(
    name="agent_name",              # Required: unique identifier
    model="gemini-2.0-flash",       # Model name or LiteLlm instance
    instruction="...",              # ✅ CORRECT (not system_instruction)
    description="...",              # Agent purpose for routing
    tools=[func1, func2],           # ✅ List of functions/tools (not ToolRegistry)
    output_key="result_key",        # Key to save output in session.state
    output_schema=PydanticModel,    # Pydantic model for structured output
    input_schema=PydanticModel,     # Pydantic model for input validation
    before_agent_callback=func,     # Callback before agent runs
    after_agent_callback=func,      # Callback after agent runs
    before_tool_callback=func,      # Callback before tool execution
    after_tool_callback=func,       # Callback after tool execution
)
```

#### 3. State Management with output_key
```python
# Agent writes to state
research_agent = Agent(
    name="researcher",
    instruction="Research the topic and save findings",
    output_key="research_findings"  # Saves output to state["research_findings"]
)

# Agent reads from state using template interpolation
writer_agent = Agent(
    name="writer",
    instruction="""
    Write a report based on research findings:
    {research_findings}

    Use this format for the report...
    """,
    # Automatically reads {research_findings} from state
    output_key="final_report"
)

# Pipeline
pipeline = SequentialAgent(
    name="research_pipeline",
    sub_agents=[research_agent, writer_agent]  # ✅ sub_agents (not agents)
)
```

#### 4. Structured Input/Output with Pydantic
```python
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent

class Analysis(BaseModel):
    category: str = Field(description="Classification category")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    summary: str = Field(description="Brief summary")
    recommendations: list[str] = Field(default_factory=list)

analyzer = LlmAgent(
    name="analyzer",
    model="gemini-2.0-flash",
    instruction="Analyze the input and return structured analysis",
    output_schema=Analysis,        # Agent MUST return this structure
    output_key="structured_analysis"
)

result = analyzer.run("Analyze this product review: The camera is great but battery life is poor")
# result.structured_analysis is a validated Analysis object
print(result.structured_analysis.category)
print(result.structured_analysis.confidence)
```

#### 5. Multi-Model Support with LiteLlm
```python
from google.adk.models.lite_llm import LiteLlm
import os

# Different models for different tasks
cheap_model = LiteLlm(model="gemini-2.0-flash")      # Fast, cheap
smart_model = LiteLlm(model="gemini-2.0-pro")        # Powerful, expensive
openai_model = LiteLlm(model=os.environ.get("OPENAI_MODEL"))  # Via LiteLLM proxy
claude_model = LiteLlm(model=os.environ.get("CLAUDE_MODEL"))  # Via LiteLLM proxy

# Use different models for different agents
classifier = Agent(model=cheap_model, ...)    # Simple classification
analyzer = Agent(model=smart_model, ...)      # Complex reasoning
linkedin_writer = Agent(model=openai_model, ...)  # OpenAI for LinkedIn
instagram_writer = Agent(model=claude_model, ...)  # Claude for Instagram
```

#### 6. Workflow Agents (Orchestration)
```python
# Sequential: Execute in order
sequential = SequentialAgent(
    name="pipeline",
    sub_agents=[step1, step2, step3]  # ✅ sub_agents parameter
)

# Parallel: Execute concurrently
parallel = ParallelAgent(
    name="concurrent_tasks",
    sub_agents=[task1, task2, task3]  # All run at same time
)

# Loop: Iterate until condition met
loop = LoopAgent(
    name="quality_loop",
    sub_agents=[generator, reviewer, improver],
    max_iterations=5,
    # Exit when any sub-agent returns Event with escalate=True
)
```

#### 7. Session Management
```python
from google.adk.sessions import InMemorySessionService
from google.adk.runner import Runner

# Create session service
session_service = InMemorySessionService()

# Create runner with session
runner = Runner(
    agent=agent,
    session_service=session_service
)

# Run with session (maintains context)
result1 = runner.run(
    user_input="My name is Rajat",
    session_id="user-123",
    user_id="rajat@example.com"
)

result2 = runner.run(
    user_input="What's my name?",  # Agent remembers from session
    session_id="user-123",
    user_id="rajat@example.com"
)
```

#### 8. Callbacks for Observability
```python
from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def before_agent_callback(context: CallbackContext) -> Optional[types.Content]:
    """Called before agent execution"""
    logger.info(f"[AGENT START] {context.agent.name}")
    context.state["start_time"] = time.time()
    return None

def after_agent_callback(context: CallbackContext) -> Optional[types.Content]:
    """Called after agent execution"""
    duration = time.time() - context.state.get("start_time", 0)
    logger.info(f"[AGENT COMPLETE] {context.agent.name} in {duration:.2f}s")
    return None

agent = Agent(
    name="monitored_agent",
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    ...
)
```

### Common Patterns Quick Reference

| Pattern | Use Case | Code |
|---------|----------|------|
| Simple agent | Single task | `Agent(name="...", instruction="...", model="...")` |
| Tool usage | Agent needs functions | `Agent(tools=[func1, func2], ...)` |
| State passing | Multi-agent pipeline | `Agent(output_key="result", ...)` |
| Structured output | Parse to Pydantic | `Agent(output_schema=Model, ...)` |
| Sequential flow | Ordered steps | `SequentialAgent(sub_agents=[...])` |
| Parallel execution | Concurrent tasks | `ParallelAgent(sub_agents=[...])` |
| Iterative refinement | Quality loops | `LoopAgent(sub_agents=[...], max_iterations=N)` |
| Multi-model | Different LLMs | `Agent(model=LiteLlm(model="..."), ...)` |
| Session memory | Conversation context | `runner.run(session_id="...", ...)` |
| Observability | Logging/tracing | `Agent(before_agent_callback=func, ...)` |

---

## SOLID Principles for Agents

### 1. Single Responsibility Principle (SRP)
**Guideline**: Each agent should have ONE clear purpose. Avoid monolithic agents.

```python
from google.adk.agents import Agent, SequentialAgent

# ❌ BAD: Monolithic agent doing everything
universal_agent = Agent(
    name="universal",
    instruction="Research, analyze, report, and execute all tasks"
    # Too many responsibilities in one agent
)

# ✅ GOOD: Specialized agents with single responsibilities
research_agent = Agent(
    name="researcher",
    model="gemini-2.0-flash",
    instruction="Gather and validate information only. Do NOT analyze or make recommendations.",
    tools=[google_search],
    output_key="research_data"
)

analysis_agent = Agent(
    name="analyzer",
    model="gemini-2.0-flash",
    instruction="""Analyze research data only. Do NOT gather new information.
    Research data: {research_data}""",
    output_key="analysis"
)

report_agent = Agent(
    name="reporter",
    model="gemini-2.0-flash",
    instruction="""Format and present findings only.
    Analysis: {analysis}""",
    output_key="final_report"
)

# Pipeline with clear separation of concerns
pipeline = SequentialAgent(
    name="research_pipeline",
    sub_agents=[research_agent, analysis_agent, report_agent]
)
```

**Benefits**: Each agent is easier to test, debug, and modify independently.

---

### 2. Open/Closed Principle (OCP)
**Guideline**: Agents should be open for extension but closed for modification.

```python
from typing import Protocol
from pydantic import BaseModel

# Abstract base for validation (enables extension without modification)
class Validator(Protocol):
    """Interface for all validators - enables extension without modification"""
    def validate(self, data: dict) -> bool: ...

# Concrete implementations (Open for Extension)
class SchemaValidator:
    """Validates data structure"""
    def validate(self, data: dict) -> bool:
        return all(k in data for k in ['required_field_1', 'required_field_2'])

class BusinessRulesValidator:
    """Validates business logic"""
    def validate(self, data: dict) -> bool:
        return data.get('amount', 0) > 0

class ComplianceValidator:
    """Validates regulatory requirements"""
    def validate(self, data: dict) -> bool:
        return data.get('region') in ['GDPR_COMPLIANT', 'CCPA_COMPLIANT']

# Tool that uses validators (Closed for Modification)
def validation_tool(validators: list[Validator], data: dict) -> dict:
    """Runs all validators without needing to know their specifics"""
    errors = []
    for validator in validators:
        if not validator.validate(data):
            errors.append(f"{validator.__class__.__name__} failed")
    return {"valid": len(errors) == 0, "errors": errors}

# Agent uses tool - can add new validators without modifying agent
validator_agent = Agent(
    name="validator",
    model="gemini-2.0-flash",
    tools=[lambda data: validation_tool([
        SchemaValidator(),
        BusinessRulesValidator(),
        ComplianceValidator()
    ], data)],
    instruction="Validate data using the validation tool. Report any errors."
)
```

**Benefits**: New validators can be added without modifying existing agent code.

---

### 3. Liskov Substitution Principle (LSP)
**Guideline**: Derived agent types must be substitutable for their base types.

```python
from typing import Protocol, runtime_checkable

# Base protocol that all agents must satisfy
@runtime_checkable
class ExecutableAgent(Protocol):
    """Contract that all agents in the system must fulfill"""
    async def execute(self, context: dict) -> dict:
        """All agents must implement execute with consistent signature"""
        ...

# Concrete implementations that satisfy the protocol
class LangGraphLegacyAgent:
    """Legacy agent from LangGraph migration"""
    async def execute(self, context: dict) -> dict:
        # LangGraph-specific implementation
        return {"result": "langraph_output", "status": "success"}

class ADKNativeAgent:
    """New ADK agent"""
    async def execute(self, context: dict) -> dict:
        # ADK-specific implementation
        return {"result": "adk_output", "status": "success"}

# Orchestrator works with ANY ExecutableAgent
class AgentOrchestrator:
    def __init__(self, agents: list[ExecutableAgent]):
        self.agents = agents

    async def run_pipeline(self, initial_context: dict) -> dict:
        """Works with ANY ExecutableAgent implementation"""
        context = initial_context
        for agent in self.agents:
            # LSP ensures all agents work the same way
            context = await agent.execute(context)
        return context

# ✅ Both agent types are interchangeable
orchestrator = AgentOrchestrator([
    LangGraphLegacyAgent(),  # Legacy
    ADKNativeAgent(),        # New
])
```

**Benefits**: Enables gradual migration from LangGraph to ADK without breaking existing workflows.

---

### 4. Interface Segregation Principle (ISP)
**Guideline**: Agents should not depend on tools they don't use.

```python
# ❌ BAD: Fat interface forces unnecessary dependencies
class UniversalToolkit:
    def search_web(self): pass
    def query_database(self): pass
    def send_email(self): pass
    def generate_image(self): pass
    def execute_code(self): pass
    # Research agent forced to have email/image capabilities it never uses

# ✅ GOOD: Segregated tool sets per agent need
from google.adk.tools import google_search

def database_query(query: str) -> dict:
    """Database query tool"""
    # Implementation...
    return {"results": []}

def send_email(recipient: str, content: str) -> bool:
    """Email sending tool"""
    # Implementation...
    return True

# Agents only get tools they need
research_agent = Agent(
    name="researcher",
    tools=[google_search],  # Only needs search
    instruction="Research using web search"
)

report_agent = Agent(
    name="reporter",
    tools=[send_email],  # Only needs email
    instruction="Send report via email"
)

data_agent = Agent(
    name="data_analyzer",
    tools=[database_query],  # Only needs database
    instruction="Query and analyze database"
)
```

**Benefits**: Agents have minimal dependencies, reducing complexity and potential for misuse.

---

### 5. Dependency Inversion Principle (DIP)
**Guideline**: High-level orchestration should not depend on low-level agent implementations.

```python
from abc import ABC, abstractmethod
from google.adk.agents import Agent

# High-level abstraction
class AgentInterface(ABC):
    """Abstract interface for all agents"""
    @abstractmethod
    async def process(self, input_data: dict) -> dict:
        """All agents must implement process"""
        pass

# Low-level implementations depend on abstraction
class GeminiAgent(AgentInterface):
    """Concrete Gemini implementation"""
    def __init__(self):
        self.agent = Agent(model="gemini-2.0-flash", ...)

    async def process(self, input_data: dict) -> dict:
        result = self.agent.run(input_data)
        return {"model": "gemini", "result": result}

class ClaudeAgent(AgentInterface):
    """Concrete Claude implementation"""
    def __init__(self):
        from google.adk.models.lite_llm import LiteLlm
        self.agent = Agent(model=LiteLlm(model="claude-3-5-sonnet-20241022"), ...)

    async def process(self, input_data: dict) -> dict:
        result = self.agent.run(input_data)
        return {"model": "claude", "result": result}

# High-level orchestrator depends only on abstraction
class WorkflowOrchestrator:
    """High-level module doesn't know about specific implementations"""
    def __init__(self, agents: list[AgentInterface]):
        self.agents = agents  # Depends on abstraction, not concrete classes

    async def execute_workflow(self, data: dict) -> dict:
        """Works with ANY AgentInterface implementation"""
        results = []
        for agent in self.agents:
            result = await agent.process(data)
            results.append(result)
        return {"results": results}

# Dependency injection at runtime
workflow = WorkflowOrchestrator([
    GeminiAgent(),  # Low-level module
    ClaudeAgent(),  # Low-level module
])
```

**Benefits**: Easy to swap implementations, test with mocks, and support multiple LLM providers.

---


## Session & Context Management

### Understanding ADK's Context Architecture

ADK separates context into distinct layers for production deployments:

| Layer | Scope | Durability | Use Case |
|-------|-------|------------|----------|
| **Working Context** | Single model call | Ephemeral | Immediate prompt |
| **Session State** | Conversation thread | Durable | Multi-turn conversations |
| **Memory** | Cross-session knowledge | Long-term | Historical recall |
| **Artifacts** | Large data (files, CSVs) | External storage | Binary/text data |

**Key Principle**: Scope by default - agents receive minimum necessary context; additional information requires explicit tool calls.

---

### Session Services

#### Option 1: InMemorySessionService (Development)
```python
from google.adk.sessions import InMemorySessionService
from google.adk.runner import Runner

# Development/testing - data lost on restart
session_service = InMemorySessionService()

runner = Runner(
    agent=your_agent,
    session_service=session_service
)

result = runner.run(
    user_input="Remember my name is Rajat",
    session_id="user-123",
    user_id="rajat@example.com"
)
```

**When to use**: Local development, unit tests, prototypes

---

#### Option 2: PostgreSQL Session Service (Production)
From your askEngage-Bot implementation:

```python
import asyncio
import json
import time
from google.adk.sessions import BaseSessionService, Session
from psycopg2.pool import ThreadedConnectionPool

class PostgresSessionService(BaseSessionService):
    """
    Production-grade PostgreSQL session persistence.
    
    Features:
    - JSONB storage for state (queryable, indexed)
    - Separate conversation_history table for audit
    - Fallback to in-memory if DB unavailable
    - Thread-safe connection pooling
    """
    
    def __init__(self):
        self._pool = ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            **get_db_connection_params()
        )
    
    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict | None = None,
        session_id: str | None = None,
    ) -> Session:
        sid = session_id or f"session-{int(time.time() * 1000)}"
        now = time.time()
        
        def _op(conn):
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO adk_sessions
                        (app_name, user_id, session_id, state, events, last_update_time)
                    VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s)
                    ON CONFLICT (app_name, user_id, session_id)
                    DO UPDATE SET
                        state = EXCLUDED.state,
                        last_update_time = EXCLUDED.last_update_time
                """, (app_name, user_id, sid, json.dumps(state or {}), json.dumps([]), now))
            conn.commit()
        
        await asyncio.to_thread(lambda: self._with_conn(_op))
        
        return Session(
            id=sid,
            appName=app_name,
            userId=user_id,
            state=state or {},
            events=[],
            lastUpdateTime=now,
        )
    
    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config=None,
    ) -> Session | None:
        def _op(conn):
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT state, events, last_update_time
                    FROM adk_sessions
                    WHERE app_name = %s AND user_id = %s AND session_id = %s
                """, (app_name, user_id, session_id))
                return cur.fetchone()
        
        row = await asyncio.to_thread(lambda: self._with_conn(_op))
        if not row:
            return None
        
        return Session(
            id=session_id,
            appName=app_name,
            userId=user_id,
            state=row[0],  # JSONB automatically parsed
            events=self._events_from_json(row[1]),
            lastUpdateTime=float(row[2] or 0.0),
        )

# Usage
session_service = PostgresSessionService()
runner = Runner(agent=agent, session_service=session_service)
```

**When to use**: Production deployments, multi-instance scaling, audit requirements

**Database Schema**:
```sql
CREATE TABLE adk_sessions (
    id BIGSERIAL PRIMARY KEY,
    app_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    state JSONB NOT NULL DEFAULT '{}'::jsonb,
    events JSONB NOT NULL DEFAULT '[]'::jsonb,
    last_update_time DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(app_name, user_id, session_id)
);

CREATE INDEX idx_adk_sessions_lookup 
ON adk_sessions(app_name, user_id, session_id);

-- Optional: Conversation history for analytics
CREATE TABLE conversation_history (
    id BIGSERIAL PRIMARY KEY,
    app_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

#### Option 3: Vertex AI Memory Bank Service (Enterprise)
```python
from google.adk.memory import VertexAiMemoryBankService
from google.adk.tools import load_memory

# Enterprise cloud-native memory with semantic search
memory_service = VertexAiMemoryBankService(
    project="your-gcp-project",
    location="us-central1",
    agent_engine_id="your-agent-engine-id"
)

# Agent with memory recall
agent = Agent(
    name="memory_enabled_agent",
    tools=[load_memory],  # Agent can search past conversations
    instruction="""You have access to load_memory tool.
    Use it when the answer might be in past conversations."""
)

runner = Runner(
    agent=agent,
    session_service=session_service,
    memory_service=memory_service  # Long-term memory
)

# After conversation completes, add to memory
completed_session = await session_service.get_session(
    app_name="app", user_id="user", session_id="session"
)
await memory_service.add_session_to_memory(completed_session)
```

**When to use**: Enterprise deployments, cross-session learning, semantic search needs

---

### Context Compaction (Token Optimization)

As conversations grow, context windows fill up. ADK provides automatic compaction:

```python
from google.adk.runner import Runner, RunConfig
from google.adk.context import LlmEventSummarizer
from google.genai import GenerativeModel

# Configure context compaction
compaction_config = {
    "compaction_interval": 5,  # Compress every 5 events
    "overlap_size": 1,          # Include 1 previous event for continuity
    "summarizer": LlmEventSummarizer(
        model=GenerativeModel("gemini-2.0-flash")  # Cheap model for summarization
    )
}

run_config = RunConfig(
    context_window_compression=compaction_config
)

runner = Runner(agent=agent, run_config=run_config)
```

**How it works**:
- Event 1-5: Full detail
- Event 6: Compress events 1-5 into summary, include event 5 for overlap
- Event 6-10: Full detail
- Event 11: Compress events 6-10 (with event 10 overlap)
- Result: Sliding window that prevents unbounded context growth

**Performance Impact**:
- Reduces context size by 60-80%
- Minimal accuracy loss for older events
- Enables conversations with 100+ turns

---

### State Management Patterns

#### Pattern 1: Shared State in Sequential Pipelines
```python
# Each agent reads from and writes to session.state
extractor = Agent(
    name="extractor",
    instruction="Extract entities from user input",
    output_key="entities"  # Writes to state["entities"]
)

validator = Agent(
    name="validator",
    instruction="""Validate extracted entities:
    {entities}
    
    Return validation result.""",
    output_key="validation"  # Reads {entities}, writes state["validation"]
)

enricher = Agent(
    name="enricher",
    instruction="""Enrich validated entities with external data:
    Entities: {entities}
    Validation: {validation}""",
    output_key="enriched_data"
)

pipeline = SequentialAgent(
    sub_agents=[extractor, validator, enricher]
)
# All agents share the same session.state
```

#### Pattern 2: Caching to Avoid Redundant API Calls
From your askEngage-Bot optimization:

```python
person_lookup_agent = Agent(
    name="PersonLookup",
    instruction="""
    **STEP 1: CHECK CACHE FIRST**
    Look at conversation history for previous person_info results.
    If person_info shows:
    - "found": true with an "fmno" value → DATA ALREADY EXISTS
    - Return immediately: {"cached": true, "found": true, "message": "Using cached data"}
    
    **STEP 2: CALL API (only if not cached)**
    If partner name exists AND no cached result:
    - Call get_persons API
    - Return the results
    """,
    tools=[get_persons_tool],
    output_schema=PersonInfo,
    output_key="person_info"
)
```

**Performance gain**: Avoids 200-500ms API calls on subsequent turns

---

### Artifact Management (Large Data)

For large files/data that shouldn't bloat context:

```python
from google.adk.tools import LoadArtifactsTool

# Store large data as artifact (not in context)
def process_large_csv(file_path: str) -> dict:
    """Tool that stores CSV as artifact"""
    # Read CSV (e.g., 5MB file)
    data = pd.read_csv(file_path)
    
    # Store as artifact (external to context)
    artifact_id = f"csv-{uuid.uuid4()}"
    artifact_service.store(artifact_id, data.to_json())
    
    # Return only metadata (not full data)
    return {
        "artifact_id": artifact_id,
        "rows": len(data),
        "columns": list(data.columns),
        "preview": data.head(5).to_dict()  # Small preview only
    }

# Agent that works with artifacts
analyzer = Agent(
    name="csv_analyzer",
    tools=[process_large_csv, LoadArtifactsTool],
    instruction="""
    Process the CSV file using process_large_csv.
    If you need to analyze the full data, use LoadArtifactsTool with the artifact_id.
    Otherwise, work with the preview.
    """
)
```

**Benefits**:
- 5MB CSV → 500 bytes in context (metadata only)
- Agent loads full data only when needed
- Supports context caching (metadata is stable prefix)

---

## Performance & Budget Optimization

### Parallel Execution Patterns

#### Anti-Pattern: Sequential API Calls (SLOW)
```python
# ❌ BAD: 3 calls × 500ms = 1500ms total
async def slow_collection():
    person = await api_call_person()      # 500ms
    client = await api_call_client()      # 500ms
    cost_center = await api_call_cost()   # 500ms
    return person, client, cost_center
```

#### Best Practice: Parallel with asyncio.gather
```python
import asyncio

# ✅ GOOD: max(500ms) = 500ms total (3x faster!)
async def fast_collection():
    person, client, cost_center = await asyncio.gather(
        api_call_person(),
        api_call_client(),
        api_call_cost(),
        return_exceptions=True  # Don't fail if one errors
    )
    return person, client, cost_center
```

#### ADK Pattern: ParallelAgent
```python
from google.adk.agents import ParallelAgent

# Create independent lookup agents
person_agent = Agent(name="PersonLookup", tools=[get_persons], output_key="person")
client_agent = Agent(name="ClientLookup", tools=[get_client], output_key="client")
cost_agent = Agent(name="CostLookup", tools=[get_cost], output_key="cost")

# Execute in parallel (ADK handles concurrency)
parallel_collector = ParallelAgent(
    name="ParallelCollector",
    sub_agents=[person_agent, client_agent, cost_agent]
)
```

**From your test_parallelism.py - Performance Targets**:
```python
# Current (sequential): ~1.4s
# Target (parallel): <0.7s
# Method: ParallelAgent with asyncio.gather

assert duration < 0.8, "API calls must be parallel (<0.8s)"
```

---

### Multi-Level Parallelism (Your askEngage-Bot Pattern)

```python
# 4-way parallel: 3 API calls + 1 CSV search simultaneously
mega_parallel_collector = ParallelAgent(
    name="MegaParallelCollector",
    description="API lookups + CSV search in parallel",
    sub_agents=[
        person_lookup_agent,        # API call #1
        cdm_lookup_agent,           # API call #2
        cost_center_lookup_agent,   # API call #3
        taxonomy_presearch_agent,   # CSV search
    ]
)

# Total time: max(500ms API, 300ms CSV) = 500ms (not 2.0s sequential)
```

**Performance Formula**:
```
Sequential: T_total = T1 + T2 + T3 + T4
Parallel:   T_total = max(T1, T2, T3, T4)

Your case:
Sequential: 500 + 500 + 500 + 300 = 1800ms
Parallel:   max(500, 500, 500, 300) = 500ms
Speedup:    3.6x
```

---

### Cost-Aware Model Routing

```python
from google.adk.models.lite_llm import LiteLlm

# Define model tiers
CHEAP_MODEL = LiteLlm(model="gemini-2.0-flash")       # $0.10/1M tokens
SMART_MODEL = LiteLlm(model="gemini-2.0-pro")         # $1.25/1M tokens
ULTRA_MODEL = LiteLlm(model="claude-opus-4-5")        # $15/1M tokens

# Route by complexity
simple_classifier = Agent(
    model=CHEAP_MODEL,
    instruction="Classify intent: greeting, question, command"
)

complex_analyzer = Agent(
    model=SMART_MODEL,
    instruction="Analyze sentiment, extract entities, identify risks"
)

critical_decision = Agent(
    model=ULTRA_MODEL,
    instruction="Make high-stakes decision with full reasoning"
)

# Cost optimization workflow
workflow = SequentialAgent(
    sub_agents=[
        simple_classifier,      # $0.10/1M - fast triage
        complex_analyzer,       # $1.25/1M - if needed
        # critical_decision only called for high-stakes cases
    ]
)
```

**Budget Impact**:
- 90% of requests: Cheap model only ($0.10/1M)
- 8% of requests: + Smart model ($1.35/1M total)
- 2% of requests: + Ultra model ($16.60/1M total)
- Weighted average: $0.50/1M (vs $15/1M if always using Ultra)
- **30x cost reduction**

---

### Deterministic Intent Routing (Code-First)

From your test_parallelism.py:

```python
# ❌ BAD: Call LLM for obvious intents (~500ms, costs tokens)
def classify_intent_with_llm(user_input: str) -> str:
    return llm.run(f"Classify intent: {user_input}")

# ✅ GOOD: Code-first for deterministic cases (<20ms, free)
def classify_intent_deterministic(user_input: str, pending_hitl: Any) -> str:
    """Classify without LLM for obvious patterns"""
    text = user_input.lower().strip()
    
    # Meta commands (instant)
    if text in ["reset", "restart", "start over"]:
        return "META_RESET"
    
    # HITL responses (instant)
    if pending_hitl:
        if text in ["yes", "y", "confirm", "ok"]:
            return "HITL_RESPONSE"
        if text.isdigit():  # Numeric selection
            return "HITL_RESPONSE"
    
    # Only use LLM for ambiguous cases
    return classify_intent_with_llm(user_input)
```

**Performance Target**: <20ms for 80% of requests (vs 500ms LLM call)

---

### Caching Strategies

#### 1. State-Based Caching (Your Pattern)
```python
# Check state before calling expensive operations
if state.get("person_info", {}).get("found"):
    return {"cached": True, "result": state["person_info"]}
else:
    result = await expensive_api_call()
    state["person_info"] = result
    return result
```

#### 2. Context Prefix Caching (ADK Feature)
```python
# Stable system instruction (cached)
agent = Agent(
    instruction="""You are a McKinsey engagement assistant.
    
    Standard operating procedures:
    1. Always validate client data
    2. Confirm with user before creation
    3. Use GOC codes for billing
    ... (2000 tokens of stable context)
    """,
    # This instruction is cached across calls
)

# Variable user input (not cached)
result = agent.run("Create engagement for Acme Corp")
# Only user input processed, system instruction served from cache
```

**Performance**: 2-10x faster for repeated calls with same system prompt

---


## Knowledge Graphs & Structured Memory

### When to Use Knowledge Graphs vs. Vector Stores

| Aspect | Vector Stores | Knowledge Graphs |
|--------|--------------|------------------|
| **Best for** | Semantic similarity, fuzzy matching | Relationships, reasoning, time-series |
| **Query type** | "Find similar to X" | "Who worked with X on Y during Z?" |
| **Accuracy** | Good for isolated facts | Excellent for complex relationships |
| **Performance** | Fast O(log n) lookups | Fast with proper indexing |
| **Cost** | Higher (embeddings + storage) | Lower (structured data) |

**Recommendation**: Use **hybrid approach** - vector store for discovery, knowledge graph for reasoning.

---

### Temporal Knowledge Graphs (Zep Architecture)

Modern agent memory should track **when** events occurred and **when** they were learned:

```python
from datetime import datetime, timezone

class TemporalEdge:
    """Edge in knowledge graph with time bounds"""
    source: str          # Entity ID
    target: str          # Related entity ID
    relationship: str    # Type of relationship
    valid_from: datetime # When relationship started
    valid_to: datetime | None  # When it ended (None = ongoing)
    created_at: datetime # When we learned about it

# Example: Professional relationships over time
edges = [
    TemporalEdge(
        source="person:rajat",
        target="company:mckinsey",
        relationship="WORKS_AT",
        valid_from=datetime(2020, 1, 1),
        valid_to=datetime(2024, 6, 30),  # Left company
        created_at=datetime(2020, 1, 15)
    ),
    Temporal Edge(
        source="person:rajat",
        target="company:anthropic",
        relationship="WORKS_AT",
        valid_from=datetime(2024, 7, 1),
        valid_to=None,  # Current
        created_at=datetime(2024, 7, 10)
    )
]

# Query: "Where did Rajat work in 2023?"
# Answer: McKinsey (valid_from <= 2023 <= valid_to)
```

**Benefits**:
- Track changing relationships over time
- Answer "what was true when?" queries
- Distinguish between event time and knowledge time

---

### Entity Extraction & Resolution

#### EDC Framework (Extract-Define-Canonicalize)

**Phase 1: Extract** - Pull entities from conversations
```python
from pydantic import BaseModel

class Entity(BaseModel):
    type: str  # "person", "company", "project", "date"
    value: str  # Raw mention
    context: str  # Surrounding text
    confidence: float

def extract_entities(text: str, history: list[str]) -> list[Entity]:
    """
    Extract entities using LLM with context.
    
    Context window: current message + last 4 messages
    """
    context = "\n".join(history[-4:] + [text])
    
    extraction_agent = Agent(
        model="gemini-2.0-flash",
        instruction="""Extract entities from the conversation.
        
        For each entity, provide:
        - type: category (person, company, project, date, location)
        - value: the actual mention
        - context: surrounding words
        - confidence: 0.0-1.0
        
        Examples:
        - "John Smith from Acme" → person:John Smith, company:Acme
        - "NYC office" → location:NYC
        """,
        output_schema=list[Entity]
    )
    
    return extraction_agent.run(context)
```

**Phase 2: Define** - Classify and structure
```python
def define_entity(entity: Entity) -> dict:
    """Classify entity and extract attributes"""
    if entity.type == "person":
        return {
            "id": f"person:{normalize(entity.value)}",
            "name": entity.value,
            "mentions": [entity.value],
            "attributes": extract_person_attributes(entity.context)
        }
    elif entity.type == "company":
        return {
            "id": f"company:{normalize(entity.value)}",
            "name": entity.value,
            "mentions": [entity.value],
            "industry": infer_industry(entity.context)
        }
```

**Phase 3: Canonicalize** - Resolve to single ID
```python
def canonicalize_entity(new_entity: dict, existing_entities: list[dict]) -> str:
    """
    Resolve entity to canonical ID (handle duplicates).
    
    "John Smith", "J. Smith", "Smith" → same person
    """
    for existing in existing_entities:
        if entity_match(new_entity, existing):
            # Merge mentions
            existing["mentions"].append(new_entity["name"])
            return existing["id"]
    
    # New entity
    return new_entity["id"]

def entity_match(e1: dict, e2: dict) -> bool:
    """Fuzzy matching for entity resolution"""
    if e1["type"] != e2["type"]:
        return False
    
    # Use LLM for ambiguous cases
    if needs_llm_matching(e1, e2):
        match_agent = Agent(
            model="gemini-2.0-flash",
            instruction=f"""Do these refer to the same entity?
            Entity 1: {e1}
            Entity 2: {e2}
            
            Return: {{"match": true/false, "confidence": 0.0-1.0}}"""
        )
        result = match_agent.run("")
        return result["match"] and result["confidence"] > 0.8
    
    # Simple cases: exact match, acronyms, etc.
    return simple_match(e1["name"], e2["name"])
```

---

### Integration with ADK Memory Service

```python
from google.adk.memory import VertexAiMemoryBankService
from google.adk.tools import load_memory

# Step 1: Extract entities and build knowledge graph during conversation
knowledge_graph = {}

def update_kg_from_conversation(user_input: str, agent_response: str):
    """Extract entities and relationships from turn"""
    entities = extract_entities(user_input + " " + agent_response, [])
    
    for entity in entities:
        canonical_id = canonicalize_entity(entity, knowledge_graph.values())
        knowledge_graph[canonical_id] = entity

# Step 2: Store in Memory Service at session end
memory_service = VertexAiMemoryBankService(...)

async def save_session_to_memory(session: Session):
    """Persist conversation with entity graph"""
    # ADK automatically extracts memories
    await memory_service.add_session_to_memory(session)
    
    # Optional: Also store knowledge graph explicitly
    for entity_id, entity in knowledge_graph.items():
        await memory_service.store_entity(entity_id, entity)

# Step 3: Agent with memory recall
agent = Agent(
    name="kg_aware_agent",
    tools=[load_memory],
    instruction="""You have access to load_memory tool.
    
    Use it to:
    - Recall past conversations
    - Find related entities
    - Track relationships over time
    
    Example: "When did we discuss Project Phoenix?"
    """
)
```

---

### Hybrid Memory Pattern (Your Use Case)

For askEngage-Bot combining episodic + semantic + knowledge graph:

```python
class HybridMemoryService:
    """Combines multiple memory backends"""
    
    def __init__(self):
        self.session_service = PostgresSessionService()  # Episodic (conversations)
        self.memory_service = VertexAiMemoryBankService()  # Semantic (searchable facts)
        self.kg_store = Neo4jGraphStore()  # Knowledge graph (relationships)
    
    async def store_conversation_turn(
        self,
        session_id: str,
        user_input: str,
        agent_response: str
    ):
        """Store in all three systems"""
        
        # 1. Episodic: Full conversation history
        await self.session_service.log_conversation(
            session_id=session_id,
            role="user",
            content=user_input
        )
        await self.session_service.log_conversation(
            session_id=session_id,
            role="assistant",
            content=agent_response
        )
        
        # 2. Semantic: Key facts for search
        facts = extract_facts(user_input, agent_response)
        await self.memory_service.store_facts(session_id, facts)
        
        # 3. Knowledge Graph: Entities + relationships
        entities = extract_entities(user_input + " " + agent_response)
        relationships = extract_relationships(entities)
        await self.kg_store.add_entities(entities)
        await self.kg_store.add_relationships(relationships)
    
    async def recall(self, query: str) -> dict:
        """Query all memory systems"""
        # Semantic search for relevant facts
        facts = await self.memory_service.search_memory(query)
        
        # Graph traversal for related entities
        entities_in_facts = extract_entities_from_facts(facts)
        related = await self.kg_store.find_related(entities_in_facts, depth=2)
        
        # Retrieve full conversations for context
        relevant_sessions = [f["session_id"] for f in facts]
        conversations = await self.session_service.get_sessions(relevant_sessions)
        
        return {
            "facts": facts,
            "related_entities": related,
            "conversations": conversations
        }
```

**When to use which**:
- **Episodic (Session)**: "What did I tell you last time?"
- **Semantic (Memory)**: "What do you know about AI agents?"
- **Knowledge Graph**: "Who have I worked with on fintech projects?"

---

## UX Design Patterns (Concierge Experience)

### The Ritz-Carlton Service Model for AI Agents

Apply hospitality industry's gold standard to agent interactions:

#### Three Steps of Service

```python
# 1. WARM WELCOME - Acknowledge user immediately
welcome_agent = Agent(
    name="greeter",
    instruction="""Provide warm, personalized greeting.
    
    If returning user (check session history):
    - "Welcome back, {user_name}! I remember you were working on {last_task}."
    
    If new user:
    - "Hello! I'm your McKinsey engagement assistant. I'll help you create charge codes efficiently."
    
    Always:
    - Use user's name if available
    - Acknowledge their goal
    - Set expectations ("This will take about 2 minutes")
    """,
    output_key="greeting"
)

# 2. ANTICIPATE NEEDS - Proactive assistance
anticipation_agent = Agent(
    name="anticipator",
    instruction="""Based on context, proactively offer help:
    
    If user mentioned "Blackstone":
    - "I see you're working with Blackstone. That's typically Private Equity industry. Shall I pre-fill that?"
    
    If engagement name > 39 chars:
    - "Your engagement name is a bit long. May I suggest shortening it to '{shortened_version}'?"
    
    If Friday 4pm:
    - "I notice it's late Friday. Would you like me to set the start date to next Monday?"
    
    Be helpful without being intrusive.
    """,
    output_key="proactive_suggestions"
)

# 3. FOND FAREWELL - Celebrate completion
farewell_agent = Agent(
    name="celebrator",
    instruction="""When task completes successfully:
    
    🎉 "Congratulations! Your charge code {charge_code} has been created for {client_name}.
    
    Here's your confirmation:
    - Project ID: {project_id}
    - Charge Code: {charge_code}
    - Start Date: {start_date}
    
    Would you like to:
    1. Create another engagement
    2. Email this confirmation to your team
    3. View the project in Workday
    
    I'm here if you need anything else!"
    
    Make them feel accomplished. Offer next steps.
    """,
    output_key="celebration"
)
```

---

### Empowerment Pattern ($2000 Autonomy)

Ritz-Carlton gives every employee $2000 authority to solve problems. Apply to agents:

```python
# Agent can make autonomous decisions within boundaries
autonomous_agent = Agent(
    name="autonomous_resolver",
    instruction="""You have authority to make these decisions WITHOUT asking:
    
    ✅ CAN DO AUTONOMOUSLY:
    - Auto-correct obvious typos (e.g., "Goggle" → "Google")
    - Default to "No Growth Platform Used" if not mentioned
    - Round dates to nearest business day
    - Shorten engagement names intelligently if >39 chars
    - Use most recent cost center if multiple found
    
    ⛔ MUST ASK USER:
    - Selecting between multiple companies (requires confirmation)
    - Changing partner name (might be intentional)
    - Modifying budget/financial data
    - Creating charge code (final confirmation needed)
    
    Your authority level: MEDIUM (can fix obvious issues, must confirm important decisions)
    """,
    tools=[auto_correct_tool, smart_default_tool]
)
```

**Empowerment Levels**:
- **Low**: Agent always asks before any action
- **Medium**: Agent fixes obvious issues, confirms important decisions (recommended)
- **High**: Agent makes most decisions, only escalates critical issues

---

### Conversation Design Patterns

#### Pattern 1: Guided Conversation (Structured Tasks)
```python
guided_agent = Agent(
    name="guided_assistant",
    instruction="""Guide user through structured process step-by-step.
    
    **Progress Tracking**:
    Step 1/6: ✅ Company name (Acme Corp)
    Step 2/6: ✅ Billing office (NYC)
    Step 3/6: ⏳ Responsible partner (NEEDED)
    Step 4/6: ⬜ Engagement name
    Step 5/6: ⬜ Dates
    Step 6/6: ⬜ Taxonomies
    
    **Next Step Guidance**:
    "Great! I have the company and billing office. Now I need the responsible partner name.
    
    This must be a McKinsey Partner or Senior Partner. Who's leading this engagement?"
    
    Keep user oriented. Show progress. Clear next action.
    """
)
```

**When to use**: Onboarding, compliance workflows, complex forms

---

#### Pattern 2: Suggest-and-Confirm (Options with Validation)
```python
suggest_confirm_agent = Agent(
    name="suggester",
    instruction="""When you find options, present them clearly:
    
    "I found 3 companies matching 'Apax':
    
    1. **Apax Partners LLP** (United Kingdom) - Private Equity
    2. **Apax Partners SA** (France) - Investment Management  
    3. **Apax Digital Fund** (United States) - Venture Capital
    
    Which one are you working with? Reply with the number or company name."
    
    Format:
    - Numbered list (1, 2, 3...)
    - Bold company names
    - Include differentiating details (location, industry)
    - Clear call-to-action
    """
)
```

**When to use**: Multiple matches, ambiguous input, important selections

---

#### Pattern 3: Proactive Nudge (Anticipatory Assistance)
```python
proactive_agent = Agent(
    name="proactive_assistant",
    instruction="""Anticipate needs based on context:
    
    **Scenario 1**: User working late
    - Check current time from context
    - If after 5pm: "I notice it's {time}. Would you like me to schedule the start date for tomorrow instead of today?"
    
    **Scenario 2**: Common pattern detected  
    - If user creates 3rd engagement for same client: "I see you're creating multiple engagements for {client}. Would you like me to remember these details for next time?"
    
    **Scenario 3**: Potential error
    - If end_date < start_date: "I notice the end date ({end_date}) is before the start date ({start_date}). Should I swap these?"
    
    Be genuinely helpful, not annoying. Offer, don't force.
    """
)
```

**When to use**: Repeated patterns, potential errors, context-aware improvements

---

### Personalization & Memory

```python
personalization_agent = Agent(
    name="personalizer",
    tools=[load_memory],  # Access past conversations
    instruction="""Personalize interactions using memory:
    
    **First-time user**:
    - "Welcome! Let me show you how this works..."
    
    **Returning user** (check memory):
    - "Welcome back, {name}! Last time you created an engagement for {last_client}."
    - If they often work with same partner: "Working with {usual_partner} again?"
    - If they prefer certain settings: Auto-apply their preferences
    
    **Power user** (10+ engagements):
    - Shorter explanations
    - Offer bulk operations
    - Skip basic confirmations
    
    Adapt to user expertise level. Remember preferences.
    """
)
```

---

### Error Handling with Empathy

```python
empathetic_error_agent = Agent(
    name="error_handler",
    instruction="""When errors occur, be empathetic and solution-focused:
    
    ❌ DON'T SAY:
    - "Error: Invalid input"
    - "Request failed"
    - "System unavailable"
    
    ✅ DO SAY:
    - "I'm having trouble finding that partner in our system. Could you double-check the spelling? Or would you like me to show you a list of partners in the {office} office?"
    
    - "The system is temporarily busy. I'll retry this in a moment... [Still working on it] ... Success! Got the data."
    
    - "I couldn't create the charge code because the start date is in the past. McKinsey policy requires future dates. Would you like me to use tomorrow's date instead?"
    
    Pattern:
    1. Acknowledge the issue (without blaming user or system)
    2. Explain why it matters
    3. Offer 2-3 solutions
    4. Stay positive and helpful
    """
)
```

---

### Multi-Turn Conversation Context

```python
context_aware_agent = Agent(
    name="context_tracker",
    instruction="""Maintain conversation context across turns:
    
    **Turn 1**:
    User: "Create charge code for Blackstone"
    You: "Great! I'll help you create a charge code for Blackstone. What's the billing office?"
    
    **Turn 2**:
    User: "NYC"
    You: ✅ Remember: company=Blackstone, office=NYC
         Ask: "Perfect. Who's the responsible partner?"
    
    **Turn 3**:
    User: "Marcus"
    You: ✅ Remember: company=Blackstone, office=NYC, partner=Marcus
         "Looking up Marcus... I found Marcus Keutel (Partner, NYC office). Is that correct?"
    
    **Turn 4**:
    User: "yes"
    You: ✅ PRESERVE all previous context
         Continue with next question
    
    CRITICAL: Each turn ADDS to context, never replaces it.
    Your extracted_entities should accumulate information.
    """
)
```

---

### Celebration & Success Patterns

```python
celebration_agent = Agent(
    name="celebrator",
    instruction="""When user succeeds, CELEBRATE appropriately:
    
    **Minor success** (validated data):
    - "✓ Got it! Acme Corp confirmed."
    
    **Medium success** (completed section):
    - "Excellent! I have all the company details now. Moving on to dates."
    
    **Major success** (charge code created):
    - "🎉 **Success! Your charge code is ready!**
       
       Charge Code: **12345.001.001**
       Client: Acme Corp
       Duration: 90 days
       
       You're all set! The engagement is now active in Workday.
       
       Would you like to create another engagement?"
    
    Match enthusiasm to accomplishment level.
    Make user feel their work mattered.
    """
)
```

---

## Multi-Agent Orchestration Patterns

### Pattern 1: Sequential Pipeline
**When to use**: Deterministic, ordered workflows

```python
# Document processing: parse → extract → summarize
parser = Agent(
    name="parser",
    instruction="Parse PDF and extract text",
    tools=[pdf_parser_tool],
    output_key="raw_text"
)

extractor = Agent(
    name="extractor",
    instruction="Extract key entities from text: {raw_text}",
    output_schema=ExtractedEntities,
    output_key="entities"
)

summarizer = Agent(
    name="summarizer",
    instruction="Summarize based on entities: {entities}",
    output_key="summary"
)

pipeline = SequentialAgent(
    name="doc_processor",
    sub_agents=[parser, extractor, summarizer]
)
```

---

### Pattern 2: Coordinator/Dispatcher
**When to use**: Intelligent routing to specialists

```python
# Customer service routing
billing_specialist = Agent(
    name="billing",
    instruction="Handle billing inquiries only",
    description="Specialist in billing, invoices, payments"
)

tech_support = Agent(
    name="tech",
    instruction="Handle technical issues only",
    description="Specialist in bugs, features, technical problems"
)

coordinator = Agent(
    name="coordinator",
    model="gemini-2.0-flash",
    sub_agents=[billing_specialist, tech_support],
    instruction="""Route user query to appropriate specialist:
    
    - Billing questions → BillingSpecialist
    - Technical issues → TechSupport
    
    Use sub-agent descriptions to decide routing.
    ADK's AutoFlow will handle the transfer.
    """
)
```

---

### Pattern 3: Parallel Fan-Out/Gather
**When to use**: Independent tasks that can run simultaneously

```python
# Code review with parallel checks
security_scanner = Agent(
    name="security",
    instruction="Scan for security vulnerabilities",
    output_key="security_report"
)

style_checker = Agent(
    name="style",
    instruction="Check code style and formatting",
    output_key="style_report"
)

complexity_analyzer = Agent(
    name="complexity",
    instruction="Analyze code complexity metrics",
    output_key="complexity_report"
)

# All run in parallel
parallel_review = ParallelAgent(
    name="code_reviewers",
    sub_agents=[security_scanner, style_checker, complexity_analyzer]
)

# Then synthesize results
synthesizer = Agent(
    name="synthesizer",
    instruction="""Create consolidated code review:
    Security: {security_report}
    Style: {style_report}
    Complexity: {complexity_report}"""
)

workflow = SequentialAgent(
    sub_agents=[parallel_review, synthesizer]
)
```

---

### Pattern 4: Hierarchical Decomposition
**When to use**: Complex tasks exceeding single agent capacity

```python
from google.adk.tools import AgentTool

# Sub-workflow: Research assistant
web_searcher = Agent(name="web_search", tools=[google_search], output_key="web_results")
summarizer = Agent(name="summarizer", instruction="Summarize: {web_results}", output_key="summary")

research_assistant = SequentialAgent(
    name="researcher",
    sub_agents=[web_searcher, summarizer]
)

# Parent agent treats sub-workflow as a tool
report_writer = Agent(
    name="writer",
    tools=[AgentTool(research_assistant)],
    instruction="""Write comprehensive report.
    Use research_assistant tool to gather information.
    Then write the report based on findings."""
)
```

---

### Pattern 5: Generator-Critic (Quality Gates)
**When to use**: Output must meet hard criteria

```python
generator = Agent(
    name="generator",
    instruction="Generate Python code for the task",
    output_key="code"
)

critic = Agent(
    name="critic",
    instruction="""Review code: {code}
    
    Check:
    - Syntax correctness
    - Security issues
    - PEP 8 compliance
    
    Return: {{"status": "PASS" or "FAIL", "feedback": "..."}}
    """,
    output_key="review"
)

refiner = Agent(
    name="refiner",
    instruction="""Fix code based on feedback:
    Original: {code}
    Feedback: {review}
    
    Generate improved version.""",
    output_key="code"  # Overwrites with improved version
)

# Loop until critic approves
quality_loop = LoopAgent(
    name="quality_loop",
    sub_agents=[generator, critic, refiner],
    max_iterations=3
    # Exits when critic returns Event(escalate=True)
)
```

---

### Pattern 6: Iterative Refinement
**When to use**: Progressive quality improvement

```python
initial_generator = Agent(
    name="generator",
    instruction="Create initial draft",
    output_key="draft"
)

critic = Agent(
    name="critic",
    instruction="""Rate quality 0-100: {draft}
    Return: {{"score": N, "suggestions": [...]}}""",
    output_key="feedback"
)

improver = Agent(
    name="improver",
    instruction="""Improve draft based on feedback:
    Draft: {draft}
    Feedback: {feedback}
    
    Generate improved version.""",
    output_key="draft"  # Iteratively improves
)

refinement_loop = LoopAgent(
    name="refiner",
    sub_agents=[critic, improver],
    max_iterations=3
)

workflow = SequentialAgent(
    sub_agents=[initial_generator, refinement_loop]
)
```

---

### Pattern 7: Human-in-the-Loop (HITL)
**When to use**: High-stakes decisions need approval

```python
def approval_tool(action: str, details: dict) -> dict:
    """Human approval tool"""
    print(f"\n🚨 APPROVAL NEEDED 🚨")
    print(f"Action: {action}")
    print(f"Details: {json.dumps(details, indent=2)}")
    response = input("Approve? (yes/no): ")
    return {"approved": response.lower() == "yes"}

transaction_agent = Agent(
    name="transaction_handler",
    tools=[approval_tool, execute_transaction_tool],
    instruction="""Process financial transactions:
    
    1. Validate transaction details
    2. If amount > $10,000: Call approval_tool
    3. If approved: Call execute_transaction_tool
    4. If not approved: Log rejection and notify user
    """
)
```

**Production HITL Pattern** (Your askEngage-Bot):
```python
# Agent presents options to user
taxonomy_selector = Agent(
    name="TaxonomySelector",
    instruction="""Present taxonomy options to user:
    
    Based on CDM data and RP history, I found:
    Industry: {industry_options}
    
    Ask user: "I suggest 'Private Equity' based on Blackstone. Is that correct?"
    
    Wait for user confirmation before proceeding.
    """
)

# Next turn processes user's selection
# This creates a natural HITL loop within conversation
```

---

### Pattern 8: Composite (Real-World)
**Combining multiple patterns**

From your askEngage-Bot architecture:

```python
# Phase 1: Sequential entity extraction
entity_extractor = Agent(...)

# Phase 2: 4-WAY PARALLEL (Fan-Out)
parallel_collector = ParallelAgent(
    sub_agents=[
        person_lookup,      # API
        client_lookup,      # API
        cost_center_lookup, # API
        taxonomy_presearch  # CSV
    ]
)

# Phase 3: Sequential preprocessing
snowflake_preloader = Agent(...)

# Phase 4: HITL with iterative refinement
taxonomy_selector = Agent(...)  # Asks user, may loop multiple times

# Phase 5: Generator-Critic pattern
project_creator = Agent(...)  # Only creates if validation passes

# Phase 6: Response with celebration
response_generator = Agent(...)

# COMPOSITE ORCHESTRATION
root_agent = SequentialAgent(
    name="CompositeWorkflow",
    sub_agents=[
        entity_extractor,     # Sequential
        parallel_collector,   # Parallel Fan-Out
        snowflake_preloader,  # Sequential
        taxonomy_selector,    # HITL
        project_creator,      # Generator-Critic
        response_generator    # Sequential
    ]
)
```

**Pattern Selection Guide**:
| Need | Pattern | Agent Type |
|------|---------|------------|
| Ordered steps | Sequential Pipeline | SequentialAgent |
| Route by intent | Coordinator | Agent + sub_agents |
| Speed up independent tasks | Parallel | ParallelAgent |
| Break down complexity | Hierarchical | AgentTool |
| Enforce quality | Generator-Critic | LoopAgent |
| Improve quality | Iterative Refinement | LoopAgent |
| Need approval | HITL | Custom approval tool |
| Real application | Composite | Mix of above |

---


## Quality Gates & Approval Workflows

### Gate 1: Pre-Execution Validation
```python
class PreExecutionGate:
    """Validates inputs before agent execution"""
    
    @staticmethod
    def validate_input(data: dict) -> tuple[bool, str]:
        """Returns: (is_valid, error_message)"""
        # Schema validation
        required_fields = ['task_type', 'priority', 'requester']
        if not all(field in data for field in required_fields):
            return False, f"Missing required fields: {required_fields}"
        
        # Business rules validation
        if data['priority'] not in ['low', 'medium', 'high', 'critical']:
            return False, "Invalid priority level"
        
        # Authorization check
        if not has_permission(data['requester'], data['task_type']):
            return False, "Requester lacks necessary permissions"
        
        return True, "Validation passed"

# Apply gate before agent execution
is_valid, message = PreExecutionGate.validate_input(user_request)
if not is_valid:
    raise ValueError(f"Pre-execution gate failed: {message}")

result = agent.run(user_request)
```

### Gate 2: Output Quality Check
```python
class QualityCheckGate:
    """Validates agent output quality"""
    
    @staticmethod
    def check_quality(output: dict, criteria: dict) -> tuple[bool, list[str]]:
        """Returns: (meets_criteria, list_of_issues)"""
        issues = []
        
        # Completeness check
        if len(output.get('content', '')) < criteria.get('min_length', 100):
            issues.append("Output too brief")
        
        # Hallucination check (cite sources)
        if criteria.get('require_citations', False):
            if not output.get('citations'):
                issues.append("No sources cited")
        
        # Tone check
        if 'formal' in criteria.get('tone', []):
            if contains_informal_language(output['content']):
                issues.append("Tone not formal enough")
        
        # Compliance check
        if contains_pii(output['content']):
            issues.append("Contains PII - compliance violation")
        
        return len(issues) == 0, issues

# Apply gate after agent execution
meets_quality, issues = QualityCheckGate.check_quality(
    output=agent_result,
    criteria={'min_length': 500, 'require_citations': True, 'tone': ['formal']}
)

if not meets_quality:
    # Trigger revision workflow
    revision_agent.run(f"Improve output. Issues: {issues}")
```

### Gate 3: Human-in-the-Loop Approval
```python
class ApprovalGate:
    """Requires human approval for sensitive operations"""
    
    REQUIRES_APPROVAL = {
        'data_deletion': ['senior_analyst', 'team_lead'],
        'external_communication': ['communications_lead'],
        'financial_transactions': ['finance_manager'],
    }
    
    @staticmethod
    async def request_approval(
        task_type: str,
        agent_output: dict,
        requester: str
    ) -> bool:
        """Returns: True if approved, False if rejected"""
        # Check if approval needed
        if task_type not in ApprovalGate.REQUIRES_APPROVAL:
            return True  # No approval needed
        
        # Identify required approvers
        required_roles = ApprovalGate.REQUIRES_APPROVAL[task_type]
        
        # Send approval request (implementation-specific)
        approval_request_id = send_approval_request(
            to_roles=required_roles,
            content=agent_output,
            requested_by=requester
        )
        
        # Wait for approval (with timeout)
        approved = await wait_for_approval(
            request_id=approval_request_id,
            timeout_minutes=120
        )
        
        return approved

# Apply gate for sensitive operations
if task_requires_approval(task_type):
    approved = await ApprovalGate.request_approval(
        task_type=task_type,
        agent_output=agent_result,
        requester=current_user
    )
    
    if not approved:
        log_rejection(task_type, agent_result, current_user)
        raise PermissionError("Operation rejected by approver")

# Proceed only after approval
execute_sensitive_operation(agent_result)
```

---

## Testing Strategies

### 1. Unit Testing Individual Agents
```python
import pytest
from google.adk.agents import Agent

def test_research_agent_single_query():
    """Test agent handles single query correctly"""
    agent = Agent(
        model="gemini-2.0-flash",
        instruction="Research the given topic and provide summary"
    )
    
    result = agent.run("What is SOLID?")
    
    assert 'single responsibility' in result.lower()
    assert len(result) > 100  # Meaningful response
    assert result.startswith("SOLID")  # Proper formatting

def test_agent_with_invalid_input():
    """Test agent error handling"""
    agent = Agent(model="gemini-2.0-flash", instruction="...")
    
    with pytest.raises(ValueError):
        agent.run("")  # Empty input should raise error

def test_agent_determinism():
    """Test agent produces consistent results"""
    agent = Agent(
        model="gemini-2.0-flash",
        instruction="Count to 5",
        temperature=0  # Deterministic
    )
    
    result1 = agent.run("Count")
    result2 = agent.run("Count")
    
    assert result1 == result2  # Should be identical
```

### 2. Integration Testing Agent Workflows
```python
def test_sequential_workflow_integration():
    """Test multi-agent pipeline works end-to-end"""
    validator = Agent(name="validator", ...)
    enricher = Agent(name="enricher", ...)
    formatter = Agent(name="formatter", ...)
    
    pipeline = SequentialAgent(
        sub_agents=[validator, enricher, formatter]
    )
    
    test_input = {"document": "test.pdf", "content": "..."}
    result = pipeline.run(test_input)
    
    # Verify complete pipeline execution
    assert result['validation_status'] == 'pass'
    assert 'enriched_data' in result
    assert result['formatted'] == True
```

### 3. Property-Based Testing
```python
from hypothesis import given, strategies as st

@given(
    query=st.text(min_size=10, max_size=500),
    priority=st.sampled_from(['low', 'medium', 'high'])
)
def test_agent_handles_arbitrary_inputs(query, priority):
    """Test agent robustness with random valid inputs"""
    agent = create_research_agent()
    
    try:
        result = agent.run({"query": query, "priority": priority})
        
        # Invariants that should ALWAYS hold
        assert isinstance(result, dict)
        assert 'status' in result
        assert result['status'] in ['success', 'failure']
        
        if result['status'] == 'success':
            assert 'content' in result
            assert len(result['content']) > 0
    
    except Exception as e:
        # Should never crash - only controlled errors
        assert isinstance(e, (ValueError, TimeoutError))
```

---

## Observability & Monitoring

### Instrumentation Pattern
```python
import logging
from datetime import datetime

class AgentTelemetry:
    """Centralized observability for agent execution"""
    
    @staticmethod
    def log_agent_start(agent_name: str, input_data: dict):
        """Log agent execution start"""
        logging.info(
            f"[AGENT_START] {agent_name}",
            extra={
                'timestamp': datetime.utcnow().isoformat(),
                'agent': agent_name,
                'input_size': len(str(input_data)),
                'event_type': 'agent_start'
            }
        )
    
    @staticmethod
    def log_agent_complete(
        agent_name: str,
        duration_ms: float,
        output_data: dict,
        tokens_used: int
    ):
        """Log agent execution completion"""
        logging.info(
            f"[AGENT_COMPLETE] {agent_name} in {duration_ms}ms",
            extra={
                'timestamp': datetime.utcnow().isoformat(),
                'agent': agent_name,
                'duration_ms': duration_ms,
                'tokens_used': tokens_used,
                'output_size': len(str(output_data)),
                'event_type': 'agent_complete'
            }
        )
    
    @staticmethod
    def log_agent_error(agent_name: str, error: Exception):
        """Log agent execution error"""
        logging.error(
            f"[AGENT_ERROR] {agent_name}: {str(error)}",
            extra={
                'timestamp': datetime.utcnow().isoformat(),
                'agent': agent_name,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'event_type': 'agent_error'
            },
            exc_info=True
        )

# Usage with callbacks
def before_agent_callback(context):
    AgentTelemetry.log_agent_start(context.agent.name, context.user_content)
    context.state["start_time"] = time.time()

def after_agent_callback(context):
    duration = (time.time() - context.state.get("start_time", 0)) * 1000
    AgentTelemetry.log_agent_complete(
        context.agent.name,
        duration,
        context.session.state,
        context.get("token_count", 0)
    )

agent = Agent(
    name="monitored_agent",
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    ...
)
```

---

## LangGraph → ADK Migration

### Key Differences

| LangGraph Concept | ADK Equivalent | Notes |
|-------------------|----------------|-------|
| `StateGraph` | `SequentialAgent` or `ParallelAgent` | Workflow orchestration |
| `Node` | `Agent` | Individual agent/step |
| Conditional `Edge` | `LoopAgent` with exit conditions | Branching logic |
| `MessageGraph` | Agent with sessions | Conversation memory |
| `ToolNode` | `Agent` with `tools` | Tool execution |
| `Checkpoint` | `SessionService` (automatic) | State persistence |

### Migration Steps

```python
# STEP 1: Map LangGraph nodes to ADK agents
# LangGraph (before)
from langgraph.graph import StateGraph

graph = StateGraph()
graph.add_node("research", research_function)
graph.add_node("analyze", analyze_function)
graph.add_edge("research", "analyze")

# ADK (after)
from google.adk.agents import Agent, SequentialAgent

research_agent = Agent(
    name="research",
    instruction="Research the topic",
    output_key="research_data"
)

analyze_agent = Agent(
    name="analyze",
    instruction="Analyze research findings: {research_data}",
    output_key="analysis"
)

workflow = SequentialAgent(
    sub_agents=[research_agent, analyze_agent]
)

# STEP 2: Handle state management
# LangGraph stores state in graph
# ADK uses session.state with output_key

# STEP 3: Session persistence
# LangGraph checkpoints manually
# ADK SessionService handles automatically

from google.adk.sessions import InMemorySessionService
from google.adk.runner import Runner

session_service = InMemorySessionService()  # or PostgresSessionService
runner = Runner(agent=workflow, session_service=session_service)

result = runner.run(
    user_input="Research AI agents",
    session_id="migration-test",
    user_id="user-123"
)
```

---

## Enterprise Deployment Checklist

### Pre-Deployment Checklist

#### Security
- [ ] All API keys stored in secret manager (not hardcoded)
- [ ] Agent service account follows least-privilege principle
- [ ] VPC Service Controls enabled (if using Vertex AI)
- [ ] Audit logging configured for all agent executions
- [ ] PII detection enabled in output validation

#### Quality Assurance
- [ ] Unit tests pass (>80% coverage)
- [ ] Integration tests pass (all workflows)
- [ ] Load testing completed (expected 3x peak traffic)
- [ ] Regression tests pass (snapshot comparisons)
- [ ] Human evaluation completed (sample outputs reviewed)

#### Observability
- [ ] Structured logging configured
- [ ] Metrics dashboards created (latency, success rate, cost)
- [ ] Alerting rules defined (error rate > 5%, latency p99 > 10s)
- [ ] Distributed tracing enabled (OpenTelemetry)
- [ ] Cost tracking implemented (token usage per agent)

#### Compliance
- [ ] Data retention policy documented
- [ ] GDPR compliance verified (if EU users)
- [ ] SOC 2 audit requirements met (if applicable)
- [ ] Incident response plan defined
- [ ] Change management process followed

#### Operational Readiness
- [ ] Runbook documented (common failure modes)
- [ ] On-call rotation defined
- [ ] Rollback procedure tested
- [ ] Disaster recovery plan validated
- [ ] Capacity planning completed (scaling limits known)

---

## Real-World Examples

See your askEngage-Bot implementation at:
- `/Users/Rajat_Bhatia/dev/askEngage-Bot/adk/askexaadk/`

Key files:
- `agents/root_agent.py` - Complete multi-agent orchestration
- `persistence/session_service.py` - Production PostgreSQL sessions
- `tests/integration/test_parallelism.py` - Performance optimization patterns

---

## Troubleshooting & Common Pitfalls

### Issue 1: "Agent not using tools"
**Symptom**: Agent responds conversationally instead of calling tools.

**Solution**: Make tool usage explicit in instruction:
```python
# ❌ BAD
instruction = "You can search the web if needed"

# ✅ GOOD
instruction = """You MUST use the google_search tool to find current information.

Steps:
1. Call google_search with the query
2. Analyze the results
3. Provide summary

Do NOT guess - always search first."""
```

### Issue 2: "State not persisting between turns"
**Symptom**: Agent forgets previous conversation.

**Solution**: Use Runner with SessionService:
```python
# ❌ BAD - No session management
agent.run("Hello")
agent.run("What's my name?")  # Agent doesn't know

# ✅ GOOD - With sessions
runner = Runner(agent=agent, session_service=session_service)
runner.run("My name is Rajat", session_id="user-123")
runner.run("What's my name?", session_id="user-123")  # Agent remembers
```

### Issue 3: "Parallel agents running sequentially"
**Symptom**: ParallelAgent takes sum of times instead of max.

**Solution**: Ensure agents are truly independent (no shared mutable state):
```python
# ✅ GOOD - Independent agents
parallel = ParallelAgent(
    sub_agents=[
        Agent(name="a1", output_key="result1"),  # Unique output keys
        Agent(name="a2", output_key="result2"),
        Agent(name="a3", output_key="result3"),
    ]
)
```

### Issue 4: "Context window exceeded"
**Symptom**: Error about token limit.

**Solution**: Use context compaction:
```python
run_config = RunConfig(
    context_window_compression={
        "compaction_interval": 5,
        "overlap_size": 1,
    }
)
runner = Runner(agent=agent, run_config=run_config)
```

### Issue 5: "Custom agents don't work - InvocationContext has no .state"
**Symptom**: `AttributeError: 'InvocationContext' object has no attribute 'state'` when trying to create conditional agents.

**Root Cause**: State access is different in different contexts:
- `CallbackContext.state` → Available in callbacks (before_agent_callback, after_agent_callback)
- `InvocationContext` → Used in agent `_run_async_impl()`, has NO `.state` attribute

**CRITICAL**: Do NOT create custom agents that override `_run_async_impl()` and try to access state.

```python
# ❌ WRONG - This will fail!
class ConditionalAgent(SequentialAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        if ctx.state.get("cached"):  # ❌ ctx has no .state!
            return cached_result

# ✅ CORRECT - Use LLM-based guards in instructions
Agent(
    instruction="""
    STEP 1: Check conversation history for cached results
    If result exists → return it immediately

    STEP 2: Do work only if no cache
    """,
)
```

**The ADK Way**: Put conditional logic in LLM instructions, not in code. The LLM can read conversation history and make intelligent decisions.

### Issue 6: "Logging error - KeyError: 'Attempt to overwrite args in LogRecord'"
**Symptom**: Logging fails with KeyError about reserved fields.

**Root Cause**: Python's `LogRecord` has reserved field names that cannot be used in the `extra` dict.

**Reserved Fields** (DO NOT USE):
- `name`, `msg`, `args`, `created`, `filename`, `funcName`
- `levelname`, `levelno`, `lineno`, `module`, `msecs`, `message`
- `pathname`, `process`, `processName`, `relativeCreated`
- `thread`, `threadName`, `exc_info`, `stack_info`

```python
# ❌ WRONG
logger.info("Tool started", extra={"args": tool_args})  # Conflicts with LogRecord.args!

# ✅ CORRECT
logger.info("Tool started", extra={"tool_arguments": tool_args})
logger.info("Tool started", extra={"tool_params": tool_args})
logger.info("Tool started", extra={"api_args": tool_args})
```

### Issue 7: "@tool decorator doesn't exist"
**Symptom**: `ImportError: cannot import name 'tool' from 'google.adk.tools'`

**Root Cause**: Some ADK versions don't expose `@tool` decorator in the public API.

**Solution**: Functions don't need decorators to be tools in ADK. Just pass the function directly.

```python
# ❌ WRONG - @tool may not exist
from google.adk.tools import tool

@tool
async def my_function():
    pass

# ✅ CORRECT - No decorator needed
async def my_function():
    """
    Tool function called directly by agents.

    Note: No @tool decorator required - ADK calls this directly.
    """
    pass

# Use it:
Agent(
    tools=[my_function],  # Just pass the function
)
```

### Issue 8: "Guard clauses / Performance optimization"
**Symptom**: Agents running redundantly even when work is already done (e.g., 168 invocations for 5-message conversation).

**Root Cause**: No mechanism to skip agents programmatically in ADK architecture.

**CRITICAL LEARNING**: You CANNOT programmatically skip agents in ADK. Callbacks cannot prevent execution. Custom conditional agents don't work (see Issue 5).

**The ADK Way - LLM-Based Guards**:

Put guard logic in agent instructions. The LLM checks conversation history and decides whether to do work or return cached results.

```python
# ✅ ADK-NATIVE APPROACH
person_lookup_agent = Agent(
    name="PersonLookup",
    instruction="""
    **STEP 1: CHECK CACHE FIRST**
    Look at conversation history for previous person_info results.
    If person_info shows:
    - "found": true with an "fmno" value → DATA ALREADY EXISTS
    - Return immediately: {"cached": true, "found": true, "message": "Using cached data"}

    **STEP 2: CHECK PREREQUISITES**
    Get responsible_partner name from EntityExtractor's output.
    If name is null/empty:
    - Return: {"found": false, "reason": "no partner name provided"}

    **STEP 3: CALL API (only if needed)**
    If partner name exists AND no cached result:
    - Call get_persons tool
    - Return results
    """,
    tools=[get_persons_tool],
    output_key="person_info",
)
```

**Why This Works**:
- LLM reads conversation history naturally
- Can make context-aware decisions
- No custom agent code needed
- Follows ADK's "smart agents, dumb framework" philosophy

**What Doesn't Work**:
- ❌ Hard-coded guards in custom agents (state not accessible)
- ❌ Callbacks preventing agent execution (not possible)
- ❌ Custom routing logic in `_run_async_impl()` (breaks ADK)

### Issue 9: "Missing helper methods in custom classes"
**Symptom**: `AttributeError: object has no attribute '_get_or_create_counter'`

**Root Cause**: Custom classes (like MetricsRegistry) call helper methods that don't exist yet.

**Solution**: Ensure all helper methods exist before calling them.

```python
# ❌ WRONG - Method doesn't exist
class MetricsRegistry:
    def record_agent_invocation(self, agent_name: str):
        counter = self._get_or_create_counter("...")  # ❌ Method not defined!

# ✅ CORRECT - Define helper first
class MetricsRegistry:
    def _get_or_create_counter(self, name: str):
        """Helper to get or create counter."""
        if name not in self._counters:
            self._counters[name] = self._meter.create_counter(name=name)
        return self._counters[name]

    def record_agent_invocation(self, agent_name: str):
        counter = self._get_or_create_counter("agent_invocations_total")
        counter.add(1, {"agent_name": agent_name})
```

---

## References & Documentation

### Official Resources
1. [ADK Documentation](https://google.github.io/adk-docs/) - Main documentation site
2. [ADK Python API Reference](https://google.github.io/adk-docs/api-reference/python/) - API docs
3. [ADK GitHub Repository](https://github.com/google/adk-python) - Source code
4. [Google Cloud ADK Overview](https://docs.cloud.google.com/agent-builder/agent-development-kit/overview) - GCP integration

### Multi-Agent Patterns
5. [Developer's Guide to Multi-Agent Patterns](https://developers.googleblog.com/developers-guide-to-multi-agent-patterns-in-adk/)
6. [Context-Aware Frameworks for Production](https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/)

### Session & Memory
7. [ADK Session Management](https://google.github.io/adk-docs/sessions/)
8. [Context Compaction](https://google.github.io/adk-docs/context/compaction/)
9. [Memory Service Guide](https://google.github.io/adk-docs/sessions/memory/)
10. [Agent State and Memory Blog](https://cloud.google.com/blog/topics/developers-practitioners/remember-this-agent-state-and-memory-with-adk)

### Knowledge Graphs & Memory
11. [Graphiti: Knowledge Graph Memory](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)
12. [Zep: Temporal Knowledge Graph Architecture](https://arxiv.org/html/2501.13956v1)
13. [Knowledge Graphs for Agentic AI](https://zbrain.ai/knowledge-graphs-for-agentic-ai/)

### UX Design Patterns
14. [Ritz-Carlton Service Principles](https://www.renascence.io/journal/how-the-ritz-carlton-enhances-customer-experience-cx-through-personalized-service-and-luxury)
15. [AI Agent UX Design Patterns](https://fuselabcreative.com/ui-design-for-ai-agents/)
16. [Designing Conversational AI](https://www.smashingmagazine.com/2024/07/how-design-effective-conversational-ai-experiences-guide/)

---


---

## Sources & References

### Official Google ADK Documentation
- [ADK Overview](https://docs.cloud.google.com/agent-builder/agent-development-kit/overview) - Google Cloud official docs (updated 2026-01-23)
- [ADK Main Documentation](https://google.github.io/adk-docs/) - Comprehensive guide to ADK
- [Python API Reference](https://google.github.io/adk-docs/api-reference/python/) - Complete API documentation
- [Multi-Agent Patterns Guide](https://developers.googleblog.com/developers-guide-to-multi-agent-patterns-in-adk/) - Official patterns blog
- [Context-Aware Multi-Agent Framework](https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/) - Production patterns
- [ADK Sessions Documentation](https://google.github.io/adk-docs/sessions/) - Session, State, and Memory
- [Context Compaction Guide](https://google.github.io/adk-docs/context/compaction/) - Token optimization
- [Memory Management with ADK](https://cloud.google.com/blog/topics/developers-practitioners/remember-this-agent-state-and-memory-with-adk) - Enterprise memory patterns
- [GitHub: adk-python](https://github.com/google/adk-python) - Open-source repository

### UX Design & Service Patterns
- [Ritz-Carlton Service Excellence](https://www.renascence.io/journal/how-the-ritz-carlton-enhances-customer-experience-cx-through-personalized-service-and-luxury) - Customer experience case study
- [Ritz-Carlton Approach to Customer Service](https://www.effectiveretailleader.com/effective-retail-leader/the-ritz-carlton-approach-to-customer-service-how-can-you-apply-those-principles-to-your-business) - Service principles
- [Ritz-Carlton Gold Standards](https://www.nist.gov/blogs/blogrige/ritz-carlton-practices-building-world-class-service-culture) - NIST case study
- [AI Agent UX Design Patterns 2026](https://www.uxdesigninstitute.com/blog/design-experiences-for-ai-agents/) - Modern agent UX
- [Conversational AI UX Design](https://www.smashingmagazine.com/2024/07/how-design-effective-conversational-ai-experiences-guide/) - Comprehensive guide
- [AI Interface Design Patterns](https://smart-interface-design-patterns.com/articles/ai-design-patterns/) - Best practices
- [Agent-Based Experience Design](https://standardbeagle.com/agent-based-experience-design/) - Future of UX

### Knowledge Graphs & Semantic Memory
- [Graphiti: Knowledge Graph Memory](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/) - Neo4j implementation
- [Zep: Temporal Knowledge Graph for Agents](https://arxiv.org/html/2501.13956v1) - Academic paper (January 2025)
- [Building AI Agents with Knowledge Graph Memory](https://medium.com/@saeedhajebi/building-ai-agents-with-knowledge-graph-memory-a-comprehensive-guide-to-graphiti-3b77e6084dec) - Practical guide
- [Production-Ready Graph Systems 2025](https://medium.com/@claudiubranzan/from-llms-to-knowledge-graphs-building-production-ready-graph-systems-in-2025-2b4aff1ec99a) - Implementation patterns
- [Knowledge Graphs for Agentic AI](https://zbrain.ai/knowledge-graphs-for-agentic-ai/) - Architecture and reasoning
- [LangGraph Long-Term Memory](https://docs.langchain.com/oss/python/langgraph/memory) - Memory patterns (for migration)
- [Cognee + LangGraph Integration](https://www.cognee.ai/blog/integrations/langgraph-cognee-integration-build-langgraph-agents-with-persistent-cognee-memory) - Persistent memory
- [MongoDB Store for LangGraph](https://www.mongodb.com/company/blog/product-release-announcements/powering-long-term-memory-for-agents-langgraph) - Cross-session memory

### LangGraph Migration Resources
- [LangGraph Memory Overview](https://docs.langchain.com/oss/python/langgraph/memory) - Official docs
- [LangGraph to ADK Migration Patterns](https://google.github.io/adk-docs/agents/multi-agents/) - Official guide
- [ADK vs LangGraph Comparison](https://developers.googleblog.com/agent-development-kit-easy-to-build-multi-agent-applications/) - Feature comparison

### Performance & Optimization
- [ADK Context Management](https://google.github.io/adk-docs/context/) - Context engineering
- [LiteLLM Documentation](https://docs.litellm.ai/docs/projects/Google%20ADK) - Multi-model support
- [Async Python Best Practices](https://docs.python.org/3/library/asyncio.html) - For parallel execution

### Your Implementation Reference
This skill incorporates patterns from your askEngage-Bot implementation:
- `askexaadk/persistence/session_service.py` - PostgreSQL session management
- `askexaadk/agents/root_agent.py` - 4-way parallel orchestration
- `adk/tests/integration/test_parallelism.py` - Performance targets and testing

## Summary

This skill provides **enterprise-grade Google ADK development** guidance emphasizing:

✅ **Correct API Usage** - Fixed all imports, parameters, and patterns from ADK 1.20.0  
✅ **SOLID Principles** - Every agent design follows software engineering best practices  
✅ **Session & Context Management** - Production PostgreSQL sessions, context compaction, memory services  
✅ **Performance Optimization** - 4-way parallelization, caching, deterministic routing, cost-aware models  
✅ **Knowledge Graphs** - Temporal graphs, entity extraction (EDC framework), hybrid memory  
✅ **UX Design Patterns** - Ritz-Carlton concierge experience, conversation design, celebration patterns  
✅ **Multi-Agent Orchestration** - All 8 patterns with real examples  
✅ **Quality Gates** - Pre-validation, output checks, human approval workflows  
✅ **Testing Strategies** - Unit, integration, property-based testing  
✅ **Observability** - Structured logging, metrics, callbacks  
✅ **Production Deployment** - Enterprise checklist, security, compliance  

**When to use this skill:**
- Building new multi-agent systems from scratch
- Migrating from LangGraph to ADK (especially for conversational systems)
- Implementing performance-critical agentic workflows
- Requiring production-grade quality gates and observability
- Designing exceptional user experiences for agents
- Enterprise/consulting contexts requiring rigorous engineering

**Version History:**
- 1.0.0 (Original) - Basic SOLID principles and patterns
- 2.0.0 (January 2026) - Complete rewrite with correct API, sessions, performance, KG, UX patterns

---

*Maintained by: Rajat Bhatia*  
*Last Updated: January 25, 2026*  
*ADK Version: google-adk==1.20.0*


