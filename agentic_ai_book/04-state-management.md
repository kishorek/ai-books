# Chapter 4: Agent State Management

State management is the difference between a demo and a production agent system. How you store, share, and recover agent state determines your system's reliability, scalability, and operational maturity. This chapter covers the state types, storage patterns, consistency models, and topology-specific strategies that define production-grade agent state management.

## Why State Management Matters

Without proper state management, agents lose context between interactions, workflows crash and lose progress, and multi-agent systems produce inconsistent results. The consequences scale with system complexity:

| System Type | State Failure Impact | Recovery Cost |
|------------|---------------------|---------------|
| Single agent chat | User loses conversation context | Low — restart conversation |
| Single agent workflow | Workflow loses progress | Medium — restart from last step |
| Multi-agent workflow | Agents produce conflicting results | High — manual reconciliation |
| Autonomous long-running agent | Agent loses accumulated knowledge | Critical — may be unrecoverable |

## State Types

```mermaid
flowchart TD
    subgraph Ephemeral["Ephemeral State"]
        A[Session State]
        B[Conversation State]
    end
    
    subgraph Operational["Operational State"]
        C[Workflow State]
        D[Event State]
    end
    
    subgraph Persistent["Persistent State"]
        E[Shared State]
        F[Persistent State]
    end
    
    Ephemeral --> Operational
    Operational --> Persistent
```

### Session State

Session state tracks the current interaction between a user and the agent. It is ephemeral — created at session start, discarded at session end.

```python
# Session state in LangGraph
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
from operator import add

class SessionState(TypedDict):
    session_id: str
    user_id: str
    messages: Annotated[list, add]  # Appended to, not replaced
    current_topic: str
    turn_count: int
```

**Storage:** Redis (fast read/write, TTL for auto-expiry) or in-memory (development only).

**Cost:** ~$0.10/GB/month on Redis. A 10KB session state × 100K concurrent sessions = ~1GB = ~$0.10/month.

### Workflow State

Workflow state tracks the progress of a multi-step task. It persists across agent restarts and is critical for long-running workflows.

```python
# Workflow state with LangGraph checkpointing
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from operator import add

class WorkflowState(TypedDict):
    workflow_id: str
    current_step: int
    total_steps: int
    step_results: Annotated[list, add]
    status: str  # running, paused, completed, failed
    created_at: str
    updated_at: str

# Production checkpointer
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/agent_state",
    autocommit=True
)

graph = StateGraph(WorkflowState)
# ... add nodes and edges ...
workflow = graph.compile(checkpointer=checkpointer)

# Execute with thread_id for state persistence
config = {"configurable": {"thread_id": "workflow-123"}}
result = workflow.invoke({"workflow_id": "wf-123", "current_step": 0}, config)
```

**Storage:** Postgres (ACID guarantees, relational queries) or DynamoDB (managed, auto-scaling).

**Cost:** Postgres on RDS: ~$50/month for db.t3.medium. Handles 10K concurrent workflows easily.

### Shared State

Shared state is the state that multiple agents read and write. This is where consistency challenges emerge.

```mermaid
flowchart LR
    A[Agent A] -->|Read/Write| S[Shared State Store]
    B[Agent B] -->|Read/Write| S
    C[Agent C] -->|Read/Write| S
    S --> D[Consistency Layer]
```

**Consistency models:**

| Model | Guarantee | Latency | Use Case |
|-------|-----------|---------|----------|
| Strong | All agents see same state at same time | 50-200ms | Financial transactions, inventory |
| Causal | Agents see causally related updates in order | 10-50ms | Most agent workflows |
| Eventual | All agents eventually see same state | 0-1000ms | Analytics, non-critical coordination |

### Persistent State

Persistent state survives system restarts, deployments, and failures. It is the authoritative source of truth for long-lived agent systems.

```python
# Persistent state with PostgresSaver (LangGraph production)
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

# Connection pool for production
pool = ConnectionPool(
    conninfo="postgresql://user:pass@localhost:5432/agent_state",
    min_size=5,
    max_size=20,
    autocommit=True,
)

checkpointer = PostgresSaver(pool)

# Initialize tables
checkpointer.setup()

# Use in workflow
config = {"configurable": {"thread_id": "agent-session-456"}}
state = workflow.get_state(config)
```

**Pruning strategy:** Do not let checkpoint tables grow unbounded. Implement a background job to delete old checkpoints:

```sql
-- Prune checkpoints older than 7 days
DELETE FROM checkpoints 
WHERE created_at < NOW() - INTERVAL '7 days'
AND thread_id NOT IN (
    SELECT thread_id FROM active_sessions
);

-- Keep only last 10 checkpoints per thread
DELETE FROM checkpoints 
WHERE id NOT IN (
    SELECT id FROM (
        SELECT id, ROW_NUMBER() OVER (
            PARTITION BY thread_id ORDER BY created_at DESC
        ) as rn
        FROM checkpoints
    ) ranked
    WHERE rn <= 10
);
```

## Storage Patterns

```mermaid
flowchart TD
    subgraph StorageDecision["Storage Selection Decision"]
        A{State type?}
        A -->|Session| B[Redis]
        A -->|Workflow| C[Postgres]
        A -->|Shared| D{Consistency?}
        A -->|Event| E[Kafka/EventStore]
        D -->|Strong| F[Postgres + Transactions]
        D -->|Eventual| G[Redis/MongoDB]
        D -->|Causal| H[Kafka with partition ordering]
    end
```

### Redis

**Best for:** Session state, caching, ephemeral coordination.

**Characteristics:**
- Read/write latency: < 1ms
- Throughput: 100K+ ops/second
- Persistence: optional (RDB snapshots or AOF)
- Clustering: Redis Cluster for horizontal scaling

**Agent use cases:**
- Session context cache
- Agent lock coordination (prevent concurrent tool calls)
- Rate limiting for LLM API calls
- Real-time feature flags

```python
# Redis state management
import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Store agent session state
def save_session(session_id: str, state: dict, ttl: int = 3600):
    r.setex(f"session:{session_id}", ttl, json.dumps(state))

def get_session(session_id: str) -> dict:
    data = r.get(f"session:{session_id}")
    return json.loads(data) if data else None

# Distributed lock for concurrent agent access
def acquire_agent_lock(agent_id: str, timeout: int = 30) -> bool:
    return r.set(f"lock:{agent_id}", "1", nx=True, ex=timeout)

def release_agent_lock(agent_id: str):
    r.delete(f"lock:{agent_id}")
```

### Postgres

**Best for:** Workflow state, shared state requiring strong consistency, audit logs.

**Characteristics:**
- Read/write latency: 2-10ms
- Throughput: 10K-50K ops/second (with connection pooling)
- Persistence: full ACID guarantees
- Query capability: SQL for complex state queries

**Agent use cases:**
- Workflow checkpoint storage
- Multi-agent shared state with transactions
- Audit trail for agent decisions
- Long-term conversation history

### MongoDB

**Best for:** Semi-structured agent state, document-oriented state, flexible schemas.

**Characteristics:**
- Read/write latency: 2-5ms
- Throughput: 10K+ ops/second
- Persistence: replica set with journaling
- Schema: flexible, document-oriented

**Agent use cases:**
- Agent configuration and persona storage
- Knowledge base state
- Conversation history with rich metadata

### Event Stores (Kafka, EventStoreDB)

**Best for:** Event state, event sourcing, audit trails, replay capability.

**Characteristics:**
- Write latency: 2-5ms (append-only)
- Throughput: millions of events/second
- Persistence: durable, ordered, immutable
- Replay: full event history available

**Agent use cases:**
- Agent action audit trail
- Event-driven coordination
- State reconstruction from event log
- Debugging agent behavior

## State Techniques

### State Rehydration

State rehydration reconstructs agent state from persisted data after a failure or restart.

```python
# State rehydration with LangGraph checkpointing
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(conn_string)

# Rehydrate state from last checkpoint
config = {"configurable": {"thread_id": "workflow-789"}}
state_snapshot = checkpointer.get(config)

# Resume workflow from last checkpoint
if state_snapshot:
    workflow.invoke(None, config)  # Resumes from last checkpoint
else:
    workflow.invoke(initial_state, config)  # Fresh start
```

**Rehydration time by storage:**
- Redis: < 1ms (in-memory)
- Postgres: 2-10ms (disk read)
- Kafka replay: 100ms-1s (depends on position)

### Context Summarization

When state exceeds context window limits, summarise before persisting.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5.4-mini")

def summarise_state(state: dict) -> dict:
    """Compress state for persistence when it exceeds token limits."""
    conversation = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in state["messages"]
    )
    
    summary = llm.invoke(
        f"Summarise this conversation concisely, preserving key facts and decisions:\n\n{conversation}"
    ).content
    
    return {
        **state,
        "messages": [{"role": "system", "content": f"Previous conversation summary: {summary}"}],
        "summarised": True
    }
```

**Token reduction:** Typical 80-90% reduction in state size.

### Working Memory vs Long-Term Memory

| Memory Type | Scope | Storage | Access Pattern | Use Case |
|------------|-------|---------|---------------|----------|
| Working | Current task | In-memory / Redis | Every LLM call | Active reasoning |
| Episodic | Past interactions | Postgres / Vector DB | On-demand recall | Learning from history |
| Semantic | Facts and knowledge | Knowledge Graph | On-demand lookup | Domain knowledge |

## Per-Topology State Management

### Hierarchical State

```mermaid
flowchart TD
    S[Supervisor State Store] --> A[Agent A: ephemeral]
    S --> B[Agent B: ephemeral]
    S --> C[Agent C: ephemeral]
    
    style S fill:#4CAF50
```

**Pattern:** Supervisor owns all persistent state. Workers are stateless — they receive task input, execute, and return results. No cross-worker state.

**Consistency:** Strong. Single writer (supervisor) for workflow state.

**Failure recovery:** Checkpoint supervisor state. Workers can be restarted from scratch.

### Peer-to-Peer State

```mermaid
flowchart TD
    A[Agent A State] <-->|Sync| B[Agent B State]
    B <-->|Sync| C[Agent C State]
    A <-->|Sync| C
    
    A --> ES[Event Store]
    B --> ES
    C --> ES
```

**Pattern:** Each agent owns local state. Coordination through event store with eventual consistency.

**Consistency:** Eventual or causal. Agents may see stale state.

**Failure recovery:** Each agent recovers from its own checkpoints. Global state reconstructed from event log.

### Hybrid State

```mermaid
flowchart TD
    S[Supervisor: Workflow State] --> A[Agent A: Operational State]
    S --> B[Agent B: Operational State]
    S --> C[Agent C: Operational State]
    
    A --> EB[Event Bus]
    B --> EB
    C --> EB
    
    EB --> S
```

**Pattern:** Supervisor holds workflow state (strong consistency). Agents hold operational state (eventual consistency). Event bus coordinates between tiers.

**Consistency:** Split — strong for workflow, eventual for operations.

**Failure recovery:** Supervisor recovers from checkpoints. Agents recover independently. Event bus provides replay.

## Topology State Comparison

| Dimension | Hierarchical | Peer-to-Peer | Hybrid |
|-----------|-------------|--------------|--------|
| State ownership | Centralised (supervisor) | Distributed (each agent) | Split (workflow + operational) |
| Consistency | Strong (simple) | Eventual/causal (complex) | Split (strong + eventual) |
| Failure recovery | Checkpoint supervisor | Checkpoint each agent | Multi-tier checkpointing |
| State size at 100 agents | ~10MB (supervisor) | ~100KB × 100 = ~10MB total | ~5MB (supervisor) + ~100KB × 100 |
| Concurrency | Single writer | Concurrent writers | Split writers |
| Query capability | SQL on single DB | Distributed queries | SQL on supervisor + local queries |

## Enterprise Constraint Table

| Constraint | Mandates | Storage Choice |
|-----------|----------|---------------|
| ACID guarantees for transactions | Postgres with transactions | Postgres |
| Sub-millisecond state reads | Redis in-memory | Redis |
| Full audit trail of state changes | Event store (Kafka) | EventStoreDB |
| GDPR right to erability | Centralised state (single delete) | Postgres |
| Cross-region replication | Distributed database | DynamoDB Global Tables |
| > 100K concurrent sessions | Redis Cluster | Redis Cluster |
| Long-running workflows (days/weeks) | Durable checkpointing | Postgres + PostgresSaver |
| Real-time coordination | Redis pub/sub or Kafka | Redis / Kafka |

## Case Study: Order Processing State Management

An e-commerce order system processing 50K orders/day with multi-agent coordination.

```mermaid
flowchart TD
    A[Order Received] --> B[Order Agent]
    B --> C{State: Postgres}
    C --> D[Inventory Agent]
    D --> C
    C --> E[Shipping Agent]
    E --> C
    C --> F[Payment Agent]
    F --> C
    C --> G[Complete]
```

**State architecture:**
- **Workflow state:** Postgres (order status, step progress, timestamps)
- **Operational state:** Redis (agent locks, in-progress calculations)
- **Event state:** Kafka (order events for audit and replay)

**State sizes:**
- Order workflow state: ~2KB per order
- 50K orders/day × 2KB = ~100MB/day
- Retained 30 days: ~3GB
- Postgres storage: ~$1/month (on RDS)

**Consistency guarantees:**
- Order status updates: strong consistency (Postgres transactions)
- Inventory checks: causal consistency (eventual acceptable for reads)
- Payment processing: strong consistency (financial compliance)

**Recovery:**
- Postgres checkpoint every step: recovery time < 100ms
- Kafka event log: full order history replay available
- Redis: ephemeral, rebuilt from Postgres on failure

## Key Takeaways

- **State type determines storage:** session → Redis, workflow → Postgres, events → Kafka
- **Consistency model matters:** strong for financial/legal, eventual for most agent coordination
- **Checkpoint every step** in LangGraph — PostgresSaver is the production standard
- **Prune checkpoints** — implement a retention policy, do not let tables grow unbounded
- **Hybrid state** is the production default — strong consistency for workflow, eventual for operations
- **State rehydration** must be fast — Redis for hot state, Postgres for warm, event log for cold
- **Context summarisation** reduces state size by 80-90% when approaching context limits
- **Topology determines state ownership** — hierarchical is simplest, P2P is most resilient, hybrid is most practical

## Further Reading

- "Designing Data-Intensive Applications" — Kleppmann (2017)
- "LangGraph: Persistence and Checkpointing" — LangChain (2024)
- "PostgresSaver: Production State Persistence" — LangChain (2025)
- "Event Sourcing" — Martin Fowler (2017)
- "CQRS Documents" — Greg Young (2010)
- "Building Reliable Distributed Systems" — Pat Helland (2007)
