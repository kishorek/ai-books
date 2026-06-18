# Chapter 9: Tool Calling and Function Calling

Tool calling is the bridge between LLMs and the real world. Without tools, a model can only generate text. With tools, it can query databases, call APIs, search the web, create tickets, send emails, and interact with any system accessible via code. Tool calling transforms LLMs from text generators into action executors — the foundation of every agent architecture.

## How Tool Calling Works

The process is straightforward. You define functions with typed parameters and descriptive names. The model analyzes the user's request, decides which tool to call based on the descriptions, generates structured arguments matching your schema, and returns a tool call. You execute the function with those arguments and return the result to the model, which then generates a response incorporating the tool's output.

The critical design decision is the quality of tool descriptions. The model decides which tool to call based entirely on the function name and description. Vague descriptions lead to wrong tool calls. Specific, action-oriented descriptions ("Search for products by name and category") outperform generic ones ("Search products").

## Designing Tools for Production

### Parameter Design

Parameters should have descriptive names, rich descriptions, and restricted types where possible. Enums for categorical parameters prevent the model from inventing invalid values. Required versus optional marking should reflect genuine requirements — not every parameter needs to be required.

### Safety Patterns

Production tool calling requires safety layers. Destructive actions (refunds, deletions, sends) should require user confirmation. SQL queries should be validated as read-only before execution. API calls should have timeout and rate limiting. All tool executions should be logged for audit.

### Parallel Execution

Independent tool calls should execute concurrently. If a user asks about weather in three cities, three weather API calls should happen in parallel, not sequentially. This reduces latency proportionally to the number of parallel calls.

## Integration Patterns

### REST APIs

The most common pattern. Define tools that map to API endpoints. The model generates the endpoint, parameters, and headers. You execute the HTTP request and return the response.

### Databases

Database tools require extra safety. The model generates SQL queries, but you must validate them before execution. Check for forbidden operations (INSERT, UPDATE, DELETE), enforce read-only access, and implement query timeouts.

### SaaS Platforms

Integrations with Slack, Jira, Salesforce, and other SaaS tools follow the same pattern. The model generates the action and parameters, you execute via the platform's API.

## Tool Orchestration Patterns

### Single Tool Calls

The simplest pattern — one tool call per turn. User asks a question, model calls one tool, returns the result.

### Multi-Tool Workflows

Multiple sequential tool calls to accomplish a task. The model might look up a customer, check their order status, and process a refund — three tool calls in sequence.

### Dynamic Tool Selection

When you have many tools, sending all of them to the model confuses it. Pre-filter tools based on query intent. A query about "database" only needs SQL tools. A query about "send message" only needs communication tools. This improves accuracy and reduces token consumption.

## Case Study: Customer Support Tool Orchestration

A customer support system used tool calling to handle complex requests. Available tools: account lookup, order status, refund processing, ticket creation, email sending, and human escalation.

An example flow: A customer reports a late order. The model looks up the account, checks order status (shipped 14 days ago, tracking shows "in transit"), offers reshipment or refund, processes the selected option, sends a confirmation email, and responds with the resolution.

Results: resolution rate improved from 45 to 82 percent, average handle time dropped from 8 to 2 minutes, and escalation rate fell from 55 to 18 percent.

## Key Takeaways

- Tool descriptions are as important as tool implementation — the model uses descriptions to decide which tool to call
- Always validate tool arguments before execution — the model can generate invalid or dangerous inputs
- Parallel tool execution reduces latency for independent operations
- Dynamic tool selection (pre-filter by intent) works better than sending all tools
- Destructive actions require user confirmation — never let an LLM delete data or send emails without approval

## Further Reading

- OpenAI Function Calling Guide
- Anthropic Tool Use Documentation
- LangChain and CrewAI tool documentation
