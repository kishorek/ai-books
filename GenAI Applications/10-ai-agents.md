# Chapter 10: AI Agents

Agents are the most complex GenAI architecture pattern. An agent is not just an LLM with tools — it is a system that plans, reasons, observes, and acts in a loop to accomplish goals. Agents decompose tasks, recover from failures, adapt strategies, and coordinate with other agents. They are also the hardest to get right.

## What Makes an Agent

An agent has four capabilities that distinguish it from a simple chain or pipeline. It **observes** its environment (receives information from tools, documents, or other agents). It **thinks** about what to do next (reasons about the goal and available actions). It **acts** by calling tools or taking actions. And it **learns** from observations (updates its understanding and adjusts its strategy).

The key difference from a chain is adaptability. A chain follows a fixed sequence of steps. An agent dynamically decides what to do next based on observations. This makes agents more powerful but also more unpredictable and harder to debug.

## The Agent Lifecycle

An agent runs in a loop: think, act, observe, repeat. At each iteration, the agent analyzes the current state, decides on the next action (which tool to call, or whether the goal is achieved), executes the action, observes the result, and decides whether to continue or stop.

The critical guardrail is the iteration limit. Without one, agents can loop forever — retrying the same failed action, exploring dead ends, or generating increasingly verbose reasoning. Set a maximum number of iterations and a timeout. When either is reached, the agent should return its best partial result with a clear indication that it did not fully complete the goal.

## Memory and Planning

Agents need memory to track progress, learn from mistakes, and maintain context. Short-term memory holds the current task context — recent observations and actions. Long-term memory holds past experiences — strategies that worked, errors to avoid. Working memory holds the current state — what has been accomplished and what remains.

Planning is the ability to decompose a goal into steps before executing. A planning agent generates a step-by-step plan, then executes each step. If a step fails, it replans. Planning improves success rates on complex tasks but adds latency and cost.

## Agent Architectures

### ReAct

The most common pattern: Reasoning plus Acting. The model interleaves thinking (what to do next) with acting (calling tools), observes the result, and continues. ReAct is the foundation of most agent implementations. Start here before exploring specialized architectures.

### Planning Agents

Separate planning from execution. First, generate a complete plan. Then execute each step. If a step fails, replan from the current state. This works well for tasks with clear decomposition — research projects, data pipelines, document processing.

### Reflection Agents

Generate an output, critique it, and improve it. The reflection loop continues until the output meets quality standards. This is powerful for code generation, analysis, and writing — any task where iterative improvement produces better results.

### Tool Using Agents

Focused primarily on tool orchestration. The agent's main capability is deciding which tools to call and in what order. This is the simplest agent pattern and sufficient for many use cases.

## Multi-Agent Systems

### Supervisor-Worker

One supervisor agent coordinates specialized worker agents. The supervisor analyzes the task, delegates subtasks to the appropriate workers, reviews results, and synthesizes the final output. This is the most common multi-agent pattern.

### Peer-to-Peer

Agents communicate directly without a central coordinator. Each agent specializes in a domain and shares findings with others. This works well for parallel exploration — multiple agents investigating different aspects of a problem simultaneously.

### Agent Swarms

Large numbers of simple agents working on parallel tasks. Each agent tries a different approach, and the best result is selected. This is effective for creative tasks and exploration but expensive in token cost.

## The Cost-Complexity Trade-off

Agents consume five to twenty times more tokens than single-turn applications. A simple query that costs $0.001 with a direct API call might cost $0.01-$0.05 with an agent. This is justified only when the task genuinely requires multi-step reasoning, tool use, or adaptation.

Start with the simplest approach that could work. Use a single LLM call if possible. Add a chain if you need sequential steps. Add an agent only when chains cannot solve the problem. Most applications do not need agents.

## Key Takeaways

- Agents are the most complex GenAI pattern — use them only when simpler patterns are insufficient
- ReAct is the default agent pattern — start here before exploring specialized architectures
- Always set max iterations and timeout — agents can loop forever without guardrails
- Multi-agent systems improve quality but increase cost and complexity
- Human-in-the-loop is mandatory for production agents — define escalation paths
- Agent evaluation is harder than single-turn evaluation — measure goal completion, not just response quality

## Further Reading

- Yao et al., "ReAct: Synergizing Reasoning and Acting" (2023)
- Wang et al., "A Survey on LLM-based Autonomous Agents" (2023)
- LangGraph agent documentation
