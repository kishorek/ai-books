# Chapter 3: Multi Agent System Design

Multi agent systems are where most AI architects face their hardest design decisions. The shift from single agent to multi agent introduces coordination, shared state, communication protocols, and failure modes that have no analog in single-agent systems. This chapter covers the topologies, patterns, and trade-offs that define production multi-agent architectures.

## Why Multi Agent

Single agent systems have a ceiling. When a task requires multiple specializations, parallel execution, or complex coordination, a single agent either cannot handle it or handles it unreliably. Multi agent systems address this by decomposing the problem into focused components.

The justification for multi agent is not always about capability. Sometimes it is about:
- **Isolation:** failures in one agent do not cascade to others
- **Specialization:** different agents with different prompts and models for different sub-tasks
- **Parallelism:** independent tasks executing simultaneously
- **Cost:** smaller, focused agents can use cheaper models for routine tasks
- **Testability:** individual agents can be tested independently

The cost is coordination complexity. Every additional agent introduces communication overhead, shared state challenges, and new failure modes. Multi agent is not a default choice — it is a decision that must justify its coordination overhead.

## Agent Topologies

The topology of a multi agent system defines how agents are connected and how work flows between them. The choice of topology determines your system's scalability, fault tolerance, and operational complexity.

### Hierarchical Systems

Hierarchical topologies organize agents in a tree structure with clear authority relationships. Work flows down from supervisors to workers, and results flow back up.

#### Supervisor Worker

The simplest hierarchical pattern. A supervisor agent receives tasks, decomposes them, and delegates sub-tasks to worker agents. Workers execute independently and report results back.

```
         [Supervisor]
        /      |      \
   [Worker A] [Worker B] [Worker C]
```

The supervisor is responsible for:
- Task decomposition
- Worker selection based on task type
- Result aggregation
- Error handling and retry
- Progress tracking

Workers are responsible for:
- Executing their assigned sub-task
- Reporting results or failures
- No cross-worker communication

This pattern is predictable and easy to debug. The supervisor provides a single point of control and visibility. The risk is the supervisor becoming a bottleneck — if the supervisor agent is slow or makes poor decomposition decisions, the entire system suffers.

**Production considerations:** Cache supervisor decisions for repeated task types. Use structured outputs from workers to simplify aggregation. Implement supervisor health checks and fallback supervisors for high-availability scenarios.

#### Manager Delegate

A refinement of supervisor-worker where the manager does not just delegate but actively manages the lifecycle of delegated work. The manager tracks which workers are busy, queues tasks for available workers, and handles worker failures with automatic reassignment.

This pattern is more resilient than simple supervisor-worker because it handles worker failures without restarting the entire workflow. The manager maintains a registry of worker capabilities and availability.

#### Tree Based Routing

In tree-based routing, agents are organized in a hierarchy where each level handles a different type of decision. A top-level router directs tasks to domain-specific routers, which further route to specialized workers.

```
            [Root Router]
           /             \
    [Domain Router A]  [Domain Router B]
    /         \              |         \
[Agent 1] [Agent 2]    [Agent 3]  [Agent 4]
```

This is effective when the task space has clear hierarchical structure — for example, a customer support system where the first level routes by product category, and the second level routes by issue type.

#### Command and Control

Command and control models have a strict authority hierarchy where the top agent has absolute control over all subordinate agents. Subordinates do not make autonomous decisions — they execute commands.

This is the most predictable hierarchical pattern but also the most rigid. It works well in scenarios where the command logic is well-defined and the value of subordinate autonomy is low.

### Peer to Peer Systems

Peer to peer topologies distribute decision-making across agents without a central authority. Agents collaborate as equals, sharing information and coordinating actions through direct communication.

#### Choreography

In choreography, each agent knows what to do based on the current state. There is no conductor — agents observe the shared state and act accordingly. The behavior of the system is the aggregate of individual agent actions, not a centrally planned sequence.

```
Agent A observes state → acts → changes state
Agent B observes new state → acts → changes state
Agent C observes new state → acts → changes state
```

Choreography is highly scalable because there is no central coordinator to become a bottleneck. It is also resilient — if one agent fails, others continue operating. The trade-off is reduced predictability. Debugging choreography requires understanding the full state transition graph, which can be complex.

#### Event Driven Collaboration

Agents communicate exclusively through events. An agent emits an event when something significant happens, and other agents subscribe to events they care about.

This decouples agents temporally — producers and consumers do not need to be running at the same time — and spatially — agents do not need to know about each other. The event bus becomes the coordination mechanism.

Event driven collaboration works well when:
- Agents have different processing rates
- You need temporal decoupling for resilience
- The system needs to scale individual agents independently
- The workflow is naturally asynchronous

The challenge is event ordering, idempotency, and correlation. When multiple events are in flight and agents process them independently, maintaining consistency requires careful design.

#### Agent Mesh

Agent mesh extends the event-driven model with service discovery. Agents register their capabilities, and other agents discover and communicate with them dynamically. No agent has a fixed set of collaborators — the mesh topology evolves as agents join and leave.

This provides maximum flexibility and resilience. Agents can be added, removed, or updated without changing other agents. The challenge is that the system's behavior is harder to predict and analyze because the communication graph is dynamic.

#### Decentralized Coordination

Decentralized coordination uses distributed consensus or agreement protocols to make collective decisions. Agents propose actions, vote on them, and execute the consensus result.

This is complex to implement but provides strong consistency guarantees without a central authority. It is most appropriate for scenarios where collective decision-making is essential — multi-agent resource allocation, distributed planning, or collaborative problem-solving.

### Hybrid Systems

Most production systems combine hierarchical and peer-to-peer patterns. Hybrid topologies use the strengths of each approach where they fit best.

#### Supervisor + Event Bus

The most common hybrid pattern. A supervisor coordinates high-level task decomposition and workflow, while agents communicate through an event bus for the actual data exchange and status updates.

```
Supervisor assigns tasks via commands
Agents communicate status and data via events
Supervisor monitors event stream for progress tracking
```

This provides the predictability of hierarchical control with the scalability and decoupling of event-driven communication. The supervisor handles the "what" and the event bus handles the "how."

#### Dynamic Agent Networks

Agents can form temporary teams for specific tasks and dissolve when the task is complete. A registration mechanism allows agents to advertise capabilities, and a team formation protocol组建s groups based on task requirements.

This is powerful for systems where the set of required capabilities varies per task. A complex request might require a team of five agents, while a simple one needs only one. Dynamic formation optimizes resource usage.

The complexity is in team formation — matching task requirements to agent capabilities, negotiating roles, and managing team lifecycle.

#### Federated Agent Systems

Federated systems combine agents from different organizations or domains under a common coordination protocol. Each domain maintains its own agents, but a federation layer allows cross-domain collaboration.

This is common in enterprise scenarios where different departments or business units have their own agent systems that need to collaborate on cross-cutting concerns.

## Trade Off Analysis: Hierarchical vs Peer-to-Peer

This section provides a deep, enterprise-grade comparison of hierarchical and peer-to-peer topologies across the dimensions that matter most in production: state management, token consumption, and horizontal scalability. For each dimension, we analyse the concrete mechanisms, quantify the trade-offs, and provide decision criteria.

### State Management: Centralised vs Distributed Ownership

State management is the most consequential architectural decision in multi agent design. It determines consistency guarantees, failure recovery characteristics, and operational complexity.

#### Hierarchical State Model

In a hierarchical (supervisor-worker) topology, state ownership is centralised. The supervisor maintains the authoritative state for the entire workflow. Workers are stateless or hold only ephemeral task-local state that is discarded after the task completes.

```
Supervisor State Store (authoritative)
├── Workflow state (current step, status, history)
├── Task assignments (worker → task mapping)
├── Aggregated results (partial outputs from workers)
└── Checkpoint data (for recovery)

Worker State (ephemeral)
├── Current task input
├── In-progress computation
└── No persistent state
```

**Consistency model:** Strong consistency is straightforward because there is a single writer (the supervisor) for workflow state. Workers read their task assignment and write results back to the supervisor. There are no concurrent state modifications.

**Storage implications:** A single relational database (Postgres) or key-value store (Redis) is sufficient. The supervisor serialises state mutations through its single-threaded reasoning loop, so you get ACID guarantees without distributed transactions.

**Failure recovery:** Recovery is simple — checkpoint the supervisor state periodically. If the supervisor fails, restore from the last checkpoint and replay from there. Workers do not need recovery because they are stateless.

**State size:** The supervisor state grows with the number of active tasks and their accumulated results. For a supply chain system with 10,000 active orders, the supervisor state may reach several MB. This is manageable but requires periodic pruning of completed task state.

**Limitations:**
- Supervisor state becomes a single point of failure (SPOF)
- State contention if multiple supervisors are needed for high availability
- State migration between supervisor instances requires careful coordination

#### Peer-to-Peer State Model

In a peer-to-peer topology, state is distributed across agents. Each agent owns its local state, and coordination requires explicit state sharing or consensus protocols.

```
Agent A State (local)
├── Agent A's task history
├── Agent A's view of the shared problem
└── Agent A's communication log

Agent B State (local)
├── Agent B's task history
├── Agent B's view of the shared problem
└── Agent B's communication log

Shared State (event store / blackboard)
├── Events emitted by all agents
├── Current global state (eventually consistent)
└── Causal history (for debugging)
```

**Consistency model:** Eventual consistency is the default. Agents may have different views of the shared state at any point in time. Strong consistency requires distributed consensus (Paxos, Raft) which adds latency and complexity.

For most agent systems, eventual consistency is acceptable — agents can tolerate slightly stale information because they continuously observe state changes. But for critical operations (financial transactions, inventory updates), you need causal consistency at minimum to prevent agents from acting on outdated information.

**Storage implications:** You need a distributed event store (Kafka, EventStoreDB) or a shared database with optimistic concurrency control. Each agent reads from and writes to the store independently. Conflicts are resolved through last-write-wins, merge functions, or domain-specific resolution logic.

**Failure recovery:** Each agent recovers independently from its last checkpoint. The global state is reconstructed from the event log. No single agent failure blocks recovery of other agents.

**State size:** State is distributed, so no single agent holds the full picture. But each agent needs enough context to make decisions, which means either replicating state or fetching it on demand — both have cost implications.

**Limitations:**
- Distributed consensus adds latency (typically 50-200ms for Raft)
- Conflict resolution is complex for concurrent state modifications
- Debugging requires correlation across multiple state stores
- State synchronisation overhead increases with agent count

#### Hybrid State Model

The hybrid model splits state ownership by type:

```
Supervisor State (workflow-level)
├── Task decomposition
├── Assignment decisions
├── Aggregated results
└── Checkpoint data

Agent State (operational-level)
├── Local computation state
├── Tool call history
└── Episode-specific memory

Event Bus State (coordination-level)
├── Event log (append-only)
├── Subscription state
└── Delivery guarantees
```

**Consistency model:** Workflow state is strongly consistent (managed by supervisor). Operational state is eventually consistent (managed by agents). Coordination state is ordered (managed by the event bus). This gives you strong consistency where it matters and eventual consistency where it is acceptable.

**Storage implications:** Three storage tiers — a relational DB for workflow state, local storage for agent state, and an event store for coordination. Each tier is optimised for its access pattern.

**Failure recovery:** Supervisor recovery uses checkpointing. Agent recovery is independent. Event bus provides replay capability for reconstructing state after failures.

**Decision framework for state management:**

| Constraint | Hierarchical | Peer-to-Peer | Hybrid |
|-----------|-------------|--------------|--------|
| Strong consistency required | ✓ Simple | Complex (consensus needed) | ✓ For workflow state |
| Concurrent state modifications | Not applicable (single writer) | Requires conflict resolution | ✓ Split ownership |
| State recovery complexity | Low (single checkpoint) | Medium (distributed checkpoints) | Medium (multi-tier) |
| State size > 10MB per agent | Not applicable | Requires state partitioning | ✓ Distributed |
| Multi-team state ownership | Complex (supervisor must be shared) | ✓ Natural per-team ownership | ✓ Team-level supervisors |

### Token Consumption and Context Window Bloat

Token consumption is the hidden cost of multi agent systems. Every message between agents, every context window refill, and every state synchronisation consumes tokens. The topology you choose determines how tokens scale.

#### Hierarchical Token Model

In hierarchical systems, context flows through a funnel. The supervisor holds the full problem context. Workers receive only the context relevant to their specific sub-task.

```
Supervisor Context (full problem context)
├── Original user request: ~500 tokens
├── Workflow history: ~2,000 tokens
├── All worker results (aggregated): ~3,000 tokens
└── Supervisor reasoning: ~1,000 tokens
    Total supervisor context: ~6,500 tokens

Worker A Context (filtered)
├── Task description: ~200 tokens
├── Relevant input data: ~500 tokens
├── Tool definitions: ~300 tokens
    Total worker context: ~1,000 tokens

Worker B Context (filtered)
├── Task description: ~200 tokens
├── Relevant input data: ~500 tokens
├── Tool definitions: ~300 tokens
    Total worker context: ~1,000 tokens
```

**Total tokens for a 5-worker task:** ~6,500 (supervisor) + 5 × 1,000 (workers) = **~11,500 tokens**

**Context filtering benefit:** Workers receive only what they need. A supply chain agent handling inventory checking does not need to see the customer's payment history. The supervisor filters context before delegation.

**Scaling characteristics:** Token consumption grows linearly with the number of workers, but each worker's context remains bounded. The supervisor context grows with the number of worker results it must aggregate. For N workers, total tokens ≈ Supervisor_base + (N × Worker_base) + (N × avg_result_size).

**Bloat risk:** The primary bloat risk is in the supervisor. As workers accumulate, the supervisor must aggregate all results, which grows its context. Mitigation: summarise worker results before aggregation, or use a tree structure where intermediate supervisors aggregate subsets.

#### Peer-to-Peer Token Model

In peer-to-peer systems, each agent needs sufficient context to make autonomous decisions. Without a central coordinator filtering context, agents often receive the full shared state or a large subset.

```
Agent A Context (needs global awareness)
├── Shared state snapshot: ~3,000 tokens
├── Recent events: ~2,000 tokens
├── Agent A's local history: ~1,000 tokens
├── Tool definitions: ~300 tokens
    Total agent context: ~6,300 tokens

Agent B Context (needs global awareness)
├── Shared state snapshot: ~3,000 tokens
├── Recent events: ~2,000 tokens
├── Agent B's local history: ~1,000 tokens
├── Tool definitions: ~300 tokens
    Total agent context: ~6,300 tokens
```

**Total tokens for a 5-agent system:** 5 × 6,300 = **~31,500 tokens**

**Why it is higher:** Each agent independently needs to understand the global state to make correct decisions. Without a coordinator, there is no context filtering step. Agents must either receive the full state or implement their own filtering logic, which adds complexity.

**Context window bloat in P2P:** The most dangerous pattern is event accumulation. In a choreography system, agents process events from an event stream. If an agent falls behind, it accumulates unprocessed events. When it catches up, it may need to process hundreds of events in a single context window, causing bloat.

**Mitigation strategies for P2P:**
- **Event summarisation:** Summarise batches of events before processing
- **Sliding window:** Only process the most recent N events
- **State snapshotting:** Periodically snapshot state so agents can start from a summary instead of replaying all events
- **Relevance filtering:** Agents subscribe only to events relevant to their domain

#### Hybrid Token Model

The hybrid model combines hierarchical filtering with event-driven communication:

```
Supervisor Context (workflow-level)
├── Full problem context: ~6,500 tokens

Worker Context (task-level, filtered by supervisor)
├── Task-specific context: ~1,000 tokens

Event Bus Messages (coordination)
├── Event payload: ~200-500 tokens per event
├── Agents process only relevant events
```

**Total tokens for a 5-worker hybrid:** ~6,500 (supervisor) + 5 × 1,000 (workers) + event overhead ≈ **~13,000 tokens**

**Benefit:** You get hierarchical context filtering for the main workflow, plus lightweight event-driven coordination for status updates and inter-agent communication. The event payloads are small because they carry only state deltas, not full context.

#### Token Cost Comparison Table

| Topology | 5 Agents | 20 Agents | 100 Agents |
|----------|---------|-----------|------------|
| Hierarchical | ~11,500 | ~46,500 | ~231,500 |
| Peer-to-Peer | ~31,500 | ~126,000 | ~630,000 |
| Hybrid | ~13,000 | ~52,000 | ~260,000 |

**At scale:** The token cost gap widens dramatically. At 100 agents, P2P uses 2.7× more tokens than hierarchical. For a system processing 10,000 requests/day, this translates to significant cost differences.

**Cost at $3/1M input tokens (GPT-4 class):**
- Hierarchical (100 agents): ~$0.70/1K requests
- Peer-to-Peer (100 agents): ~$1.89/1K requests
- Hybrid (100 agents): ~$0.78/1K requests

**Context window limits:** If your model has a 128K context window, hierarchical systems can support larger agent counts before hitting limits because each agent's context is bounded. P2P systems hit context limits sooner because each agent carries more global state.

### Horizontal Scalability

Horizontal scalability determines how well your system handles increasing load by adding more agents.

#### Hierarchical Scalability

Hierarchical systems scale by adding workers under the supervisor. The supervisor can distribute tasks in parallel, and workers execute independently.

**Scaling mechanism:**
```
Request arrives → Supervisor decomposes → N workers execute in parallel → Supervisor aggregates
```

**Bottleneck analysis:**

The supervisor is the fundamental bottleneck. It must:
1. Receive and process every incoming request (serialisation point)
2. Decompose the request into sub-tasks (reasoning overhead)
3. Wait for all workers to complete (synchronisation barrier)
4. Aggregate results (processing overhead)

**Quantifying the bottleneck:**
- Supervisor reasoning time: ~2-5 seconds per request (LLM call)
- Worker execution time: ~1-3 seconds per sub-task
- Supervisor aggregation time: ~1-2 seconds
- Total latency per request: ~4-10 seconds (supervisor-bound)

**Throughput ceiling:** If the supervisor can process 10 requests/second, that is your system's maximum throughput regardless of how many workers you add. Adding workers reduces per-request latency (parallelism) but not system throughput (supervisor bottleneck).

**Scaling strategies for hierarchical:**
1. **Supervisor pooling:** Run multiple supervisor instances behind a load balancer. Each handles a subset of requests. Effective until state synchronisation between supervisors becomes the new bottleneck.
2. **Hierarchical supervision:** Add a level above the supervisor — a meta-supervisor that distributes requests across supervisors. Effective up to 2-3 levels before coordination overhead dominates.
3. **Request batching:** The supervisor processes batches of requests together, amortising the reasoning overhead across multiple requests.
4. **Caching:** Cache supervisor decomposition decisions for repeated request types. A supply chain system may see the same order patterns repeatedly — cache the decomposition and skip reasoning.

#### Peer-to-Peer Scalability

Peer-to-peer systems scale without a central bottleneck. Each agent processes requests independently, and the event bus distributes work across agents.

**Scaling mechanism:**
```
Event arrives → Event bus routes to subscribed agents → Agents process in parallel → Results emitted as events
```

**Bottleneck analysis:**

No single agent is a bottleneck, but other limits emerge:
1. **Event bus throughput:** The event bus must handle all messages. Kafka can handle millions of events/second, but each agent's subscription adds processing overhead.
2. **State synchronisation:** As agents grow, the volume of state updates increases. The shared state store must handle concurrent reads and writes from all agents.
3. **Coordination overhead:** Agents may need to wait for information from other agents, creating implicit dependencies that limit parallelism.
4. **Discovery overhead:** In dynamic meshes, discovering and connecting to new agents adds latency.

**Throughput characteristics:** P2P systems scale nearly linearly with agent count up to the event bus and state store limits. Adding 2x agents roughly doubles throughput for embarrassingly parallel tasks.

**Scaling strategies for P2P:**
1. **Partitioning:** Partition the event stream so each agent processes only its partition. Kafka consumer groups provide this natively.
2. **Agent pooling:** Run multiple instances of the same agent type behind a consumer group. The event bus load-balances across instances.
3. **Backpressure:** Implement backpressure when agents cannot keep up. Prevents event accumulation and context bloat.
4. **State sharding:** Shard the state store by task type or agent domain. Each agent reads/writes only its shard.

#### Scalability Comparison

| Dimension | Hierarchical | Peer-to-Peer | Hybrid |
|-----------|-------------|--------------|--------|
| Max throughput | Limited by supervisor | Limited by event bus | Limited by supervisor (workflow) + event bus (coordination) |
| Adding agents | Reduces latency, not throughput | Increases throughput linearly | Increases throughput for P2P portion |
| Failure at scale | Supervisor failure = total outage | Agent failure = graceful degradation | Supervisor failure = workflow paused, agents continue |
| State at scale | Supervisor state grows with tasks | Distributed state, manageable per agent | Multi-tier state, each tier scaled independently |
| Cost at scale | Linear with agent count | Linear with agent count | Sub-linear (hierarchical portion uses cheaper models) |

### Enterprise Constraint Mapping

This section provides a decision framework for topology selection based on specific enterprise constraints. Use this table when evaluating which topology to mandate for a given system.

#### Regulatory and Compliance Constraints

| Constraint | Mandates | Rationale |
|-----------|----------|-----------|
| SOC2 audit trail for all decisions | **Hierarchical** | Supervisor logs every decomposition, delegation, and aggregation decision in a single audit log. P2P requires correlating logs across agents. |
| HIPAA data access controls | **Hierarchical or Hybrid** | Centralised access control at the supervisor. Workers access only the data they need. P2P requires per-agent access control, which is harder to audit. |
| GDPR right to erasure | **Hierarchical** | Data deletion from a single supervisor state store is straightforward. P2P requires deletion across all agent state stores and event logs. |
| Financial regulatory reporting | **Hierarchical** | Regulators expect a single authoritative source of truth. Supervisor state provides this. P2P distributed state makes reporting complex. |
| Cross-org data sharing (GDPR Art. 26) | **Hybrid (Federated)** | Each org maintains its own agents. Federation layer controls cross-org data flow. Neither pure hierarchical nor pure P2P handles this well. |

#### Operational Constraints

| Constraint | Mandates | Rationale |
|-----------|----------|-----------|
| 99.99% availability SLA | **P2P or Hybrid** | Hierarchical has supervisor SPOF. P2P degrades gracefully. Hybrid isolates workflow failure from operational execution. |
| < 100ms inter-agent latency | **P2P** | No coordinator hop. Direct agent-to-agent communication via event bus or shared memory. |
| Multi-team ownership | **Hybrid** | Each team owns its agents (P2P within team). Cross-team coordination via shared supervisor or event bus. |
| Dynamic agent fleet (auto-scaling) | **P2P or Hybrid** | Agents register/deregister dynamically. No static configuration. P2P with service discovery handles this naturally. |
| Complex long-running workflows | **Hierarchical or Hybrid** | Durable execution with supervisor checkpointing. P2P long-running workflows are harder to track and recover. |
| Real-time event processing | **P2P (Event Driven)** | No coordinator latency. Agents react to events as they arrive. |

#### Business Constraints

| Constraint | Mandates | Rationale |
|-----------|----------|-----------|
| Budget ceiling on token spend | **Hierarchical** | Context filtering reduces token usage by 50-70% vs P2P. Supervisor summarises before delegating. |
| Rapid prototyping required | **Hierarchical** | Supervisor-worker is simpler to implement, debug, and iterate on. P2P requires infrastructure investment upfront. |
| Multiple LLM providers | **Hybrid** | Supervisor uses one provider, workers use another. Or different workers use different providers based on task requirements. |
| Global deployment (multi-region) | **P2P** | No single-region coordinator. Agents deployed independently across regions with eventual consistency. |
| Vendor lock-in avoidance | **P2P or Hybrid** | No dependency on a single orchestration framework. Agents are independent services. |

#### The Enterprise Constraint Decision Framework

When choosing between hierarchical and P2P, apply this decision process:

```
1. Is strong consistency required for workflow state?
   YES → Hierarchical or Hybrid (supervisor provides strong consistency)
   NO → P2P is acceptable

2. Is 99.99% availability required?
   YES → P2P or Hybrid (no SPOF)
   NO → Hierarchical is acceptable

3. Is strict audit trail required?
   YES → Hierarchical (centralised logging)
   NO → P2P is acceptable

4. Is token budget constrained?
   YES → Hierarchical (context filtering saves 50-70%)
   NO → P2P cost is acceptable

5. Are teams independent?
   YES → Hybrid (team-level autonomy with cross-team coordination)
   NO → Hierarchical (centralised control)

6. Are agents dynamically scaled?
   YES → P2P (natural service discovery)
   NO → Hierarchical (static worker pool is fine)

Score: If 4+ answers point to Hierarchical → Hierarchical
       If 4+ answers point to P2P → P2P
       Otherwise → Hybrid
```

## Case Study: Autonomous Supply Chain System

Consider an autonomous supply chain system that manages order fulfilment across multiple warehouses, carriers, and suppliers. This system must:
- Process orders in real-time
- Coordinate across 15+ agent types (order, inventory, shipping, billing, etc.)
- Maintain audit trails for compliance
- Scale to 50,000 orders/day
- Recover from agent failures without losing orders

### Hierarchical Implementation

```
Order Supervisor
├── Receives all orders
├── Decomposes into: inventory check, shipping assignment, billing
├── Delegates to specialised workers
├── Aggregates results and confirms order
└── Logs every decision for audit trail
```

**State management:** Supervisor maintains order state in Postgres. Workers are stateless. Audit trail is the supervisor's decision log.

**Token cost:** ~12,000 tokens per order (supervisor + 3 workers). At 50K orders/day: ~600M tokens/day ≈ $1,800/day at GPT-4 rates.

**Scalability limit:** Supervisor can handle ~5 orders/second. At peak (10 orders/second), you need supervisor pooling with shared state.

### P2P Implementation

```
Order Event → OrderAgent processes → emits order_validated
order_validated → InventoryAgent checks stock → emits stock_reserved
order_validated → ShippingAgent checks rates → emits shipping_quoted
stock_reserved + shipping_quoted → FulfilmentAgent confirms → emits order_confirmed
order_confirmed → BillingAgent processes payment → emits payment_complete
```

**State management:** Each agent owns its state. Shared order state in Kafka event log. Causal consistency ensures agents act on the correct event sequence.

**Token cost:** ~6,500 tokens per agent × 5 agents = ~32,500 tokens per order. At 50K orders/day: ~1.6B tokens/day ≈ $4,800/day at GPT-4 rates.

**Scalability:** No bottleneck. Each agent type scales independently via Kafka consumer groups.

### Hybrid Implementation (Recommended)

```
Order Supervisor (workflow-level)
├── Receives orders
├── Decomposes into sub-tasks
├── Delegates to agent pools via event bus
├── Monitors event stream for progress
└── Logs decisions for audit trail

Agent Pools (operational-level)
├── InventoryAgent pool (3 instances, consumer group)
├── ShippingAgent pool (2 instances, consumer group)
├── BillingAgent pool (2 instances, consumer group)
└── Each agent processes events independently
```

**State management:** Supervisor state in Postgres (strong consistency for workflow). Agent state distributed (eventual consistency for operations). Event bus provides ordered delivery.

**Token cost:** ~13,000 tokens per order (supervisor + filtered worker contexts + event overhead). At 50K orders/day: ~650M tokens/day ≈ $1,950/day.

**Scalability:** Supervisor handles workflow coordination (bottleneck at ~20 orders/second with pooling). Agent pools scale independently via consumer groups. Event bus handles coordination overhead.

**Why hybrid wins here:**
1. Audit trail: Supervisor logs all decisions (SOC2 compliance)
2. Availability: Agent failures do not block order processing
3. Cost: Context filtering saves 60% vs P2P
4. Scalability: Agent pools scale independently
5. Recovery: Supervisor checkpoints enable order recovery

## Topology Selection Guide

| Requirement | Recommended Topology |
|------------|---------------------|
| Simple task decomposition | Supervisor Worker |
| High availability (99.99%) | P2P or Hybrid |
| Strict governance / audit trail | Hierarchical or Hybrid |
| Independent agent scaling | Event Driven (P2P) |
| Dynamic agent fleet | P2P with service discovery |
| Cross-organisation collaboration | Hybrid (Federated) |
| Token budget constrained | Hierarchical |
| Real-time event processing | P2P (Event Driven) |
| Complex long-running workflows | Hybrid (Supervisor + Event Bus) |
| Multi-team ownership | Hybrid |
| Rapid prototyping | Hierarchical |
| Global multi-region deployment | P2P |

## Key Takeaways

- **State management** is the primary architectural decision — hierarchical provides strong consistency simply; P2P requires distributed consensus for the same guarantees
- **Token consumption** in P2P is 2-3× higher than hierarchical due to context duplication across agents; hybrid captures most of the hierarchical savings
- **Horizontal scalability** — hierarchical is bottlenecked by the supervisor; P2P scales linearly; hybrid scales the operational layer independently
- **Enterprise constraints** (audit, compliance, availability SLA) are the strongest drivers of topology choice — start with constraints, not preferences
- **Hybrid is the production default** for most enterprise systems because it combines hierarchical governance with P2P scalability
- **Quantify before deciding** — model token costs, throughput requirements, and failure scenarios before committing to a topology

## Further Reading

- "Multi-Agent Systems: A Modern Approach to Distributed Artificial Intelligence" — Weiss (2013)
- "Building Scalable Multi-Agent Systems with LangGraph" — LangChain (2024)
- "Microsoft AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" — Wu et al. (2023)
- "CrewAI: Role-Playing Autonomous AI Agents" — Moura (2024)
- "Swarm: Lightweight Multi-Agent Orchestration" — OpenAI (2024)
- "Agent Protocol: A Standard for Agent Communication" — Agent Protocol Working Group (2024)
