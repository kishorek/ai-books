# Chapter 11: Agent Frameworks

Agent frameworks provide the scaffolding for building agent systems — state management, tool integration, workflow orchestration, and observability. Choosing the right framework saves months of development. Choosing the wrong one creates technical debt.

## The Major Frameworks

### LangGraph

LangGraph is the best choice for complex, stateful agents in production. It provides fine-grained control over agent flow as a directed graph, built-in state management with checkpointing, conditional routing and cycles, and human-in-the-loop support.

The learning curve is steep — you need to understand graph theory concepts (nodes, edges, state). But the investment pays off for production systems that need reliability, observability, and the ability to pause and resume long-running tasks.

### LangChain

LangChain has the broadest ecosystem with 700+ integrations. It is excellent for prototyping and applications that need many third-party connections. The trade-off is abstraction overhead — more layers between you and the LLM means more potential points of failure and harder debugging.

For new projects, LangGraph is generally preferred over LangChain for agent workflows. LangChain remains useful for its integration ecosystem and for simpler applications.

### CrewAI

CrewAI focuses on role-based multi-agent teams. You define agents with roles ("Senior Researcher," "Technical Writer"), assign tasks, and let the crew collaborate. This is intuitive and easy to set up, making it great for content creation pipelines and research workflows.

The limitation is flexibility. Complex state management, conditional routing, and production reliability patterns are harder to implement than in LangGraph.

### AutoGen

Microsoft's framework excels at code generation tasks with built-in code execution support. It handles multi-agent conversations and human-in-the-loop patterns well. The learning curve is steep and the ecosystem is smaller than LangChain's.

### PydanticAI

PydanticAI is the framework for type-safe, performance-sensitive applications. Built on Pydantic, it provides type-safe output by default with minimal overhead. The trade-off is a smaller ecosystem and less built-in multi-agent support.

## Choosing a Framework

The decision matrix is straightforward:

| Need | Choose |
|------|--------|
| Complex stateful workflows | LangGraph |
| Many integrations, prototyping | LangChain |
| Role-based multi-agent teams | CrewAI |
| Code generation with execution | AutoGen |
| Type safety, performance | PydanticAI |

But the most important principle is that framework choice matters less than architecture. A well-designed agent in any framework beats a poorly designed agent in the "best" framework. Start simple, add framework complexity only when needed.

## Orchestration Patterns

### State Machines

Explicit state management for agent workflows. The agent transitions between defined states (idle, planning, executing, observing, complete, failed) with enforced valid transitions. This prevents invalid state changes and makes the agent's behavior predictable.

### Graph Workflows

LangGraph-style directed graphs. Nodes represent actions, edges represent transitions. Conditional edges enable dynamic routing based on state. This is the most flexible pattern for complex workflows.

### Durable Execution

For long-running tasks (hours or days), durable execution frameworks like Temporal provide checkpointing, recovery, and reliability. The agent's state is persisted at each step, so failures can be recovered without restarting from scratch.

## Key Takeaways

- LangGraph is the best choice for complex, stateful agents in production
- CrewAI is best for role-based multi-agent workflows — intuitive but less flexible
- PydanticAI is best for type-safe, performance-sensitive applications
- Framework choice matters less than architecture — design well in any framework
- Start simple, add complexity only when needed — a simple ReAct loop often suffices

## Further Reading

- LangGraph, LangChain, CrewAI, AutoGen, and PydanticAI documentation
