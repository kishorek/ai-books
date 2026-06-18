# Chapter 4: Prompt Engineering

> "Prompt engineering is not a hack or a trick — it is the primary interface between human intent and model behavior. An architect who masters prompt engineering can extract 80 percent of a model's capability without fine-tuning."

---

Last verified: June 2026. Verify current model specifications at provider documentation.

Prompt engineering is the art and science of communicating with large language models. It is not a hack or a trick — it is the primary interface between human intent and model behavior. An architect who masters prompt engineering can extract 80 percent of a model's capability without fine-tuning. An architect who ignores it will build systems that underperform regardless of the model.

The central thesis of this chapter is that **system prompts are architecture, not configuration**. A change in system prompt can change model behavior as dramatically as a code change. System prompts define the contract between your system and the model. They should be version controlled, tested, and treated with the same rigor as application code. Prompt engineering is an iterative, measurable discipline — not a one-time setup.

---

## 4.1 The Foundation: System, User, and Assistant Messages

Every conversation with an LLM consists of three message types. Understanding how each message type influences model behavior is the foundation of prompt engineering.

### 4.1.1 Message Types and Their Roles

**System messages** define the model's behavior, role, and constraints. They are processed first and have the strongest influence on output. The system message is the architectural contract — it defines what the model does and does not do.

**User messages** are the specific requests that vary per interaction. They provide the task-specific input that the model processes within the constraints defined by the system message.

**Assistant messages** are previous responses from the model — they maintain conversation continuity. They also influence the model's style and tone, creating a feedback loop where earlier responses shape later ones.

```python
from openai import OpenAI

client = OpenAI()

# Basic conversation structure
messages = [
    {
        "role": "system",
        "content": "You are a senior financial analyst. Always cite sources. Never provide investment advice."
    },
    {
        "role": "user", 
        "content": "What are the key risks for tech stocks in Q3 2026?"
    },
    {
        "role": "assistant",
        "content": "Based on current market conditions, here are three key risks..."
    },
    {
        "role": "user",
        "content": "How do these risks compare to Q3 2025?"
    }
]

response = client.chat.completions.create(
    model="gpt-5.4",
    messages=messages,
    temperature=0.3,
)
```

### 4.1.2 System Prompt Architecture

System prompts are part of your architecture. They define the contract between your system and the model. A change in system prompt can change model behavior as dramatically as a code change.

The most effective system prompts follow a consistent structure:

| Component | Purpose | Example |
|-----------|---------|---------|
| Role definition | Who the model is | "You are a senior financial analyst with 15 years of experience" |
| Behavior rules | What to do and not do | "Always cite sources. Never provide investment advice." |
| Output format | Structure of response | "Use markdown with headers: Analysis, Metrics, Risks" |
| Constraints | Boundaries | "If uncertain, state uncertainty explicitly" |
| Tone | Style of communication | "Professional, data-driven, objective" |

```python
# Production system prompt template
SYSTEM_PROMPT_TEMPLATE = """## Role
You are a {role} with {experience} in {domain}.

## Rules
{rules}

## Output Format
{format}

## Constraints
{constraints}

## Tone
{tone}

## Examples (if needed)
{examples}"""

# Financial analyst example
FINANCIAL_ANALYST_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(
    role="senior financial analyst",
    experience="15 years of experience",
    domain="equity research and market analysis",
    rules="""- Always cite sources for data points
- Never provide specific investment advice (buy/sell/hold)
- Include risk factors in every analysis
- Distinguish between factual data and your analysis
- If uncertain, state your uncertainty explicitly""",
    format="""## Analysis
[Your analysis with citations]

## Key Metrics
[Table of key financial metrics]

## Risk Factors
[Bullet list of risks]

## Confidence
[Your confidence level: High/Medium/Low with reasoning]""",
    constraints="""- Maximum response length: 500 words
- Must include at least one data citation
- Must include at least two risk factors""",
    tone="Professional, data-driven, objective. Avoid hype and sensationalism.",
    examples=""
)
```

---

## 4.2 Core Techniques

### 4.2.1 Zero Shot and Few Shot Prompting

**Zero shot** prompting provides no examples — the model relies entirely on its training. This works well for simple, well-defined tasks where the model has strong training signal: classification, summarization, translation.

**Few shot** prompting provides examples of input-output pairs. The improvement is significant: accuracy on classification tasks jumps from 70-80 percent (zero shot) to 80-90 percent (three to five shots). Beyond five examples, marginal improvements rarely justify the token cost.

| Technique | Examples | Accuracy (Classification) | Token Cost | Best For |
|-----------|---------|--------------------------|------------|----------|
| Zero shot | 0 | 70-80% | Low | Simple, well-defined tasks |
| 1-shot | 1 | 75-85% | Medium | Quick improvement |
| 3-shot | 3 | 82-90% | Medium | Optimal cost-quality |
| 5-shot | 5 | 84-91% | High | Marginal improvement |
| 10-shot | 10 | 85-92% | Very High | Diminishing returns |

The sweet spot is one to three examples. The choice between zero shot and few shot is not always obvious. Start with zero shot; add examples only when quality is insufficient. Few shot adds tokens and cost — every example consumes context window space.

```python
# Zero shot classification
zero_shot_prompt = """Classify the following customer message into one of these categories:
- billing: Questions about charges, payments, refunds
- technical: Bug reports, error messages, system issues
- general: Product questions, feature inquiries

Customer message: "I keep getting error 500 when I try to export my data"

Category:"""

# Few shot classification (3 examples)
few_shot_prompt = """Classify customer messages into categories.

Examples:
Message: "Why was I charged twice this month?"
Category: billing

Message: "The app crashes when I open settings"
Category: technical

Message: "Does this product support mobile devices?"
Category: general

Now classify:
Message: "I keep getting error 500 when I try to export my data"

Category:"""

# Measure accuracy difference
# Zero shot: ~75% accuracy on edge cases
# 3-shot: ~88% accuracy on same edge cases
# The examples disambiguate cases where categories overlap
```

### 4.2.2 Chain of Thought

Chain of thought (CoT) prompting instructs the model to show its reasoning step by step. The improvement on reasoning tasks is dramatic: multi-step math accuracy jumps from 45 to 85 percent, logic puzzles from 35 to 75 percent.

The key insight is that CoT is free in model cost but adds output tokens. For reasoning tasks, always include "think step by step" or equivalent instructions. For simple factual queries, CoT adds latency without quality improvement. Use it selectively.

| Task Type | Zero-Shot Accuracy | With CoT | Improvement |
|-----------|-------------------|----------|-------------|
| Multi-step math | 45% | 85% | +40% |
| Logic puzzles | 35% | 75% | +40% |
| Causal reasoning | 55% | 80% | +25% |
| Simple factual Q&A | 90% | 91% | +1% (not worth the tokens) |
| Classification | 80% | 81% | +1% (not worth the tokens) |

```python
# Without CoT (direct answer)
direct_prompt = """What is 15% of $84,500?"""

# With CoT (step-by-step reasoning)
cot_prompt = """What is 15% of $84,500?

Think step by step:
1. Identify the percentage and the base amount
2. Convert percentage to decimal
3. Multiply
4. Verify the result"""

# Direct answer: "The answer is $12,675" (may be wrong, no verification)
# CoT answer: "1. Percentage: 15%, Base: $84,500
#              2. 15% = 0.15
#              3. 0.15 × $84,500 = $12,675
#              4. Verification: $12,675 / $84,500 = 0.15 ✓
#              The answer is $12,675." (verified, traceable)
```

Reasoning models do CoT internally — they think before they speak. For standard models, explicit CoT prompting achieves similar benefits.

### 4.2.3 ReAct: Reasoning and Acting

ReAct is the pattern that bridges internal reasoning with external actions. The model interleaves thinking (what to do next) with acting (calling a tool), then observes the result and decides whether to continue.

This is the foundation of every agent architecture. When you see a model that can search the web, query databases, and execute code in a loop — that is ReAct under the hood.

```python
# ReAct pattern implementation
REACT_PROMPT = """You are a helpful assistant that can use tools to answer questions.

Available tools:
- search(query): Search the web for information
- calculate(expression): Perform mathematical calculations
- lookup(entity): Look up information in a database

For each step, use this format:
Thought: [Your reasoning about what to do next]
Action: [Tool name and arguments]
Observation: [Result of the tool call]
... (repeat as needed)
Thought: [Final reasoning]
Answer: [Final answer to the user]

Example:
Question: What is the population of the capital of France?
Thought: I need to find the capital of France first.
Action: search("capital of France")
Observation: Paris is the capital of France.
Thought: Now I need the population of Paris.
Action: search("population of Paris 2026")
Observation: Paris has approximately 2.1 million people (city proper).
Thought: I have the information needed.
Answer: The population of Paris, the capital of France, is approximately 2.1 million people (city proper)."""

# ReAct execution loop
def react_loop(query: str, max_iterations: int = 5):
    """Execute ReAct reasoning loop."""
    messages = [
        {"role": "system", "content": REACT_PROMPT},
        {"role": "user", "content": query},
    ]
    
    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=messages,
            temperature=0,
        )
        
        output = response.choices[0].message.content
        
        # Check if we have a final answer
        if "Answer:" in output:
            return output.split("Answer:")[-1].strip()
        
        # Parse action
        if "Action:" in output:
            action = parse_action(output)
            observation = execute_tool(action["tool"], action["args"])
            
            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
    
    return "Max iterations reached without final answer"
```

### 4.2.4 Self Consistency

Self consistency generates multiple reasoning paths and selects the most common answer. It doubles or triples cost but improves accuracy 10 to 20 percent. Use it when accuracy matters more than latency — multi-step reasoning, math, logic puzzles.

```python
import collections

def self_consistency(query: str, n_samples: int = 5) -> str:
    """Generate multiple reasoning paths and vote on the answer."""
    answers = []
    
    for _ in range(n_samples):
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": "Think step by step, then provide your final answer after 'Answer:'"},
                {"role": "user", "content": query},
            ],
            temperature=0.7,  # Higher temperature for diverse reasoning paths
        )
        
        output = response.choices[0].message.content
        if "Answer:" in output:
            answer = output.split("Answer:")[-1].strip()
            answers.append(answer)
    
    # Vote on most common answer
    vote_counts = collections.Counter(answers)
    most_common = vote_counts.most_common(1)[0]
    
    return {
        "answer": most_common[0],
        "confidence": most_common[1] / len(answers),
        "vote_distribution": dict(vote_counts),
    }

# Self consistency for medical triage
# 5 samples, 3 say "cardiology", 1 says "general", 1 says "emergency"
# → Final answer: "cardiology" (60% confidence)
# Without self consistency: might pick any of the three
```

### 4.2.5 Reflection Prompting

Reflection is a two-pass technique: generate an initial output, then ask the model to critique and improve it. It doubles cost but significantly improves quality for high-stakes outputs.

```python
def reflection_prompting(task: str) -> str:
    """Two-pass generation with reflection."""
    
    # Pass 1: Generate initial output
    initial_response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": "Generate a response to the following task."},
            {"role": "user", "content": task},
        ],
        temperature=0.7,
    )
    initial_output = initial_response.choices[0].message.content
    
    # Pass 2: Critique and improve
    reflection_response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {
                "role": "system",
                "content": """You are a critical reviewer. Analyze the following output and:
1. Identify any factual errors or unsupported claims
2. Note any logical gaps or missing considerations
3. Suggest specific improvements
4. Provide an improved version"""
            },
            {
                "role": "user",
                "content": f"""Task: {task}

Initial output:
{initial_output}

Please critique and provide an improved version."""
            },
        ],
        temperature=0.3,  # Lower temperature for careful critique
    )
    
    return reflection_response.choices[0].message.content

# Reflection for high-stakes content
# Initial: "Tech stocks are likely to outperform in Q3..."
# Critique: "The claim 'likely to outperform' is unsupported. Missing: 
#            historical context, sector-specific analysis, risk factors"
# Improved: "Based on historical Q3 patterns and current valuations, 
#            tech stocks show mixed signals. Key factors: [supported analysis]"
```

---

## 4.3 Advanced Techniques

### 4.3.1 Structured Outputs

Forcing the model to respond in a specific format is essential for production systems. JSON mode ensures valid JSON. Schema validation with Pydantic ensures the JSON matches your expected structure.

| Output Level | Guarantee | Failure Rate | Cost Overhead |
|-------------|-----------|-------------|---------------|
| Free-form text | None | 0% (but downstream parsing fails 15-25%) | 0% |
| JSON mode | Valid JSON syntax | 5-10% (schema mismatch) | 5-10% |
| JSON schema | Matches schema | <1% | 15-25% |
| Function calling | Structured arguments | <0.5% | 20-30% |

```python
from pydantic import BaseModel, Field
from typing import Literal

# Define expected output schema
class AnalysisResult(BaseModel):
    summary: str = Field(min_length=50, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high", "critical"]
    key_factors: list[str] = Field(min_length=2, max_length=5)
    recommendations: list[str] = Field(min_length=1, max_length=3)

# Use schema enforcement
response = client.chat.completions.create(
    model="gpt-5.4",
    messages=[
        {"role": "system", "content": "Analyze the following data and return structured results."},
        {"role": "user", "content": "Analyze Q2 financial performance..."},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "analysis_result",
            "schema": AnalysisResult.model_json_schema(),
        }
    },
)

# Validate with Pydantic (safety net)
import json
result = AnalysisResult(**json.loads(response.choices[0].message.content))
```

GPT-5.4 with structured output achieves 99 percent schema adherence. Without it, JSON generation succeeds only 85 percent of the time. The 14 percentage point improvement justifies the 15-25 percent cost overhead for any production system that processes LLM output programmatically.

### 4.3.2 Function Calling

Function calling lets the model invoke functions you define. The model analyzes the user's request, decides which tool to call, generates structured arguments, and returns a tool call. You execute the function and return the result.

This is the bridge between LLMs and the real world — the foundation of every agentic system.

```python
# Define tools the model can use
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price and change",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT)"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_portfolio_value",
            "description": "Calculate total portfolio value given holdings",
            "parameters": {
                "type": "object",
                "properties": {
                    "holdings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "shares": {"type": "number"}
                            }
                        },
                        "description": "List of stock holdings"
                    }
                },
                "required": ["holdings"]
            }
        }
    }
]

# Model decides which function to call
response = client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "What's the current price of Apple?"}],
    tools=tools,
    tool_choice="auto",
)

# Parse tool call
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    # Execute: get_stock_price(symbol="AAPL")
```

### 4.3.3 Guardrail Prompting

Guardrails constrain model behavior to prevent unwanted outputs. A customer support system might include rules like "never disclose internal pricing," "never provide legal advice," and "if you don't know, connect with a specialist."

But guardrail prompts are fragile. They can be bypassed with jailbreak attacks. Production systems need layered defense: prompt-level guardrails plus output validation plus content filtering. Do not rely on prompt alone for safety.

| Defense Layer | What It Catches | Strength | Weakness |
|--------------|----------------|----------|----------|
| System prompt rules | Obvious violations | Low | Bypassable with creativity |
| Output validation | Schema violations, forbidden content | Medium | Cannot catch all semantic issues |
| Content filtering | Harmful, toxic, or sensitive content | High | May be overly aggressive |
| Human review | Everything else | Highest | Does not scale, adds latency |
| Red-team testing | Identifies gaps in all layers | Essential | Must be ongoing |

```python
# Layered guardrail implementation
class GuardrailedLLM:
    """LLM with multiple defense layers."""
    
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.forbidden_topics = [
            "investment advice", "legal advice", "medical diagnosis",
            "internal pricing", "competitor information",
        ]
    
    def generate(self, user_message: str) -> dict:
        """Generate response with guardrails."""
        
        # Layer 1: Input validation
        if self._contains_jailbreak(user_message):
            return {"error": "Input rejected", "reason": "potential_jailbreak"}
        
        # Layer 2: LLM generation with system prompt guardrails
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )
        output = response.choices[0].message.content
        
        # Layer 3: Output validation
        violations = self._validate_output(output)
        if violations:
            return {"error": "Output rejected", "violations": violations}
        
        # Layer 4: Content filtering
        if self._contains_forbidden_content(output):
            return {"error": "Output contains forbidden content"}
        
        return {"response": output}
    
    def _contains_jailbreak(self, text: str) -> bool:
        """Detect common jailbreak patterns."""
        jailbreak_signals = [
            "ignore previous instructions",
            "you are now",
            "pretend you are",
            "bypass your restrictions",
        ]
        return any(signal in text.lower() for signal in jailbreak_signals)
    
    def _validate_output(self, text: str) -> list[str]:
        """Validate output against guardrails."""
        violations = []
        text_lower = text.lower()
        for topic in self.forbidden_topics:
            if topic in text_lower:
                violations.append(f"Contains forbidden topic: {topic}")
        return violations
    
    def _contains_forbidden_content(self, text: str) -> bool:
        """Check for harmful content."""
        # In production: use content moderation API
        return False
```

---

## 4.4 Multi-Step Prompting

Chain multiple prompts to accomplish complex tasks. A document analysis pipeline might extract key information in step one, verify claims in step two, and generate a summary in step three. Each step adds a full model call — latency and cost scale linearly with the number of steps.

### 4.4.1 Pipeline Design

```python
class DocumentAnalysisPipeline:
    """Multi-step document analysis with validation."""
    
    def __init__(self):
        self.steps = [
            ("extract", self.extract_information),
            ("verify", self.verify_claims),
            ("summarize", self.generate_summary),
            ("validate", self.validate_output),
        ]
    
    def analyze(self, document: str) -> dict:
        """Execute analysis pipeline."""
        context = {"document": document}
        results = {}
        
        for step_name, step_fn in self.steps:
            start_time = time.time()
            result = step_fn(context)
            latency = time.time() - start_time
            
            results[step_name] = {
                "output": result,
                "latency_ms": latency * 1000,
            }
            context[step_name] = result
        
        return results
    
    def extract_information(self, context: dict) -> dict:
        """Step 1: Extract key information from document."""
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {
                    "role": "system",
                    "content": "Extract key information as JSON with fields: title, date, author, key_claims, statistics"
                },
                {"role": "user", "content": context["document"][:4000]},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
    
    def verify_claims(self, context: dict) -> dict:
        """Step 2: Verify extracted claims."""
        claims = context["extract"].get("key_claims", [])
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {
                    "role": "system",
                    "content": "Verify each claim. Return JSON with: claim, status (verified/unverified/contradicted), confidence, reasoning"
                },
                {"role": "user", "content": f"Claims to verify: {claims}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
    
    def generate_summary(self, context: dict) -> str:
        """Step 3: Generate executive summary."""
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {
                    "role": "system",
                    "content": "Write an executive summary. Maximum 200 words. Include key findings and confidence level."
                },
                {
                    "role": "user",
                    "content": f"Document info: {context['extract']}\nVerification: {context['verify']}",
                },
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content
    
    def validate_output(self, context: dict) -> dict:
        """Step 4: Validate output quality."""
        summary = context["summarize"]
        return {
            "word_count": len(summary.split()),
            "has_key_findings": "finding" in summary.lower(),
            "has_confidence": any(w in summary.lower() for w in ["confidence", "likely", "uncertain"]),
        }
```

### 4.4.2 Cost and Latency Analysis

| Step | Model | Input Tokens | Output Tokens | Latency | Cost |
|------|-------|-------------|---------------|---------|------|
| Extract | GPT-5.4 | 4,000 | 200 | 400ms | $0.013 |
| Verify | GPT-5.4 | 500 | 300 | 350ms | $0.006 |
| Summarize | GPT-5.4 | 800 | 200 | 300ms | $0.005 |
| Validate | GPT-5.4 | 300 | 50 | 200ms | $0.002 |
| **Total** | | **5,600** | **750** | **1,250ms** | **$0.026** |

The decision to use multi-step prompting depends on whether the task genuinely benefits from decomposition. For many tasks, a single well-crafted prompt outperforms a chain. Multi-step is better when:
- Each step has a distinct evaluation criterion
- Intermediate results need validation
- Errors in early steps compound if not caught
- Different models are optimal for different steps

---

## 4.5 Case Study: Medical Triage

### 4.5.1 Problem Statement

A healthcare platform needed to triage patient messages. The requirements were strict:
- Classification accuracy above 95%
- Emergency recall above 98% (never miss an emergency)
- Response time under 2 seconds
- HIPAA compliant
- Cost under $0.01 per classification

### 4.5.2 Prompt Evolution

The system evolved through three iterations:

**Iteration 1: Zero-shot classification.**
Achieved 72 percent accuracy. Dangerous misclassification of urgent messages as routine. Emergency recall was only 82 percent — meaning 18 percent of emergencies were misclassified.

```python
# Iteration 1: Zero shot
ITERATION_1_PROMPT = """Classify this patient message into: emergency, urgent, routine, self-care.

Patient message: {message}

Category:"""

# Results: 72% accuracy, 82% emergency recall (UNACCEPTABLE)
```

**Iteration 2: Few-shot with chain-of-thought.**
Adding three few-shot examples and chain-of-thought instructions improved accuracy to 94 percent. Emergency recall improved to 89 percent.

```python
# Iteration 2: Few-shot + CoT
ITERATION_2_PROMPT = """Classify patient messages by urgency. Think step by step.

Examples:
Message: "I'm having chest pain and can't breathe"
Analysis: This describes acute cardiac symptoms. Chest pain with breathing difficulty is a medical emergency.
Category: emergency

Message: "My prescription refill is overdue and I'm running low"
Analysis: This is a medication access issue. Important but not immediately life-threatening.
Category: urgent

Message: "What are your office hours?"
Analysis: This is a general information request.
Category: routine

Now classify:
Message: {message}

Analysis:"""

# Results: 94% accuracy, 89% emergency recall (IMPROVED BUT NOT SUFFICIENT)
```

**Iteration 3: Two-pass validation.**
A validation layer that double-checked emergency classifications with a reasoning model pushed accuracy to 97 percent and emergency recall to 96 percent.

```python
# Iteration 3: Classification + Validation
class MedicalTriageSystem:
    def __init__(self):
        self.classifier_prompt = ITERATION_2_PROMPT
        self.validator_prompt = """You are a medical safety validator. 
Review this classification for potential errors.

Patient message: {message}
Initial classification: {classification}
Confidence: {confidence}

Is this classification correct? Could this be an emergency that was missed?
Respond with: CONFIRMED or REVISED (with corrected category and reasoning)."""
    
    def classify(self, message: str) -> dict:
        # Step 1: Initial classification
        classification = self._classify(message)
        
        # Step 2: Validate emergency and urgent classifications
        if classification["category"] in ["emergency", "urgent"]:
            validation = self._validate(message, classification)
            if validation["revised"]:
                classification = validation["corrected"]
        
        return classification
    
    def _classify(self, message: str) -> dict:
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": self.classifier_prompt.format(message=message)},
            ],
            temperature=0,
        )
        return parse_classification(response.choices[0].message.content)
    
    def _validate(self, message: str, classification: dict) -> dict:
        response = client.chat.completions.create(
            model="deepseek-r1",  # Reasoning model for validation
            messages=[
                {
                    "role": "system",
                    "content": self.validator_prompt.format(
                        message=message,
                        classification=classification["category"],
                        confidence=classification["confidence"],
                    ),
                },
            ],
            temperature=0,
        )
        return parse_validation(response.choices[0].message.content)

# Results: 97% accuracy, 96% emergency recall (ACCEPTABLE)
```

### 4.5.3 Results Comparison

| Metric | Iteration 1 | Iteration 2 | Iteration 3 | Target |
|--------|------------|------------|------------|--------|
| Overall accuracy | 72% | 94% | 97% | >95% |
| Emergency recall | 82% | 89% | 96% | >98% |
| Emergency precision | 78% | 91% | 95% | >90% |
| Urgent recall | 70% | 88% | 94% | >85% |
| Cost per classification | $0.003 | $0.003 | $0.005 | <$0.01 |
| P50 latency | 400ms | 600ms | 1,100ms | <2,000ms |
| P95 latency | 800ms | 1,200ms | 1,800ms | <3,000ms |

The validation layer improved emergency recall from 89 to 96 percent. The additional cost per classification ($0.005 versus $0.003) was justified by the risk reduction. The additional latency (500ms) was within the 2-second budget.

### 4.5.4 Cost Analysis

| Component | Per-Classification Cost | Monthly (50K messages) |
|-----------|------------------------|----------------------|
| Initial classification (GPT-5.4) | $0.003 | $150 |
| Validation (DeepSeek R1, 30% of cases) | $0.002 | $30 |
| **Total** | **$0.005** (avg) | **$180** |

The $0.005 average cost per classification was well under the $0.01 target. The validation layer added $0.002 only to the 30 percent of messages that received validation, keeping the average low while significantly improving safety.

### 4.5.5 Compliance and Safety

The system maintained HIPAA compliance through:
- No PHI in system prompts or logs
- Encrypted message processing
- Audit trail for every classification
- Human override capability for edge cases
- Monthly accuracy review with clinical staff

```json
{
  "timestamp": "2026-06-15T14:23:07.123Z",
  "message_id": "msg-2026-88431",
  "classification": "urgent",
  "confidence": 0.94,
  "validation_performed": true,
  "validation_result": "CONFIRMED",
  "latency_ms": 1050,
  "model_version": "gpt-5.4-v2026.06",
  "hipaa_compliant": true
}
```

---

## 4.6 Testing Prompt Engineering

### 4.6.1 Prompt Testing Framework

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class PromptTestCase:
    input: str
    expected_output: str
    category: str
    min_accuracy: float

class PromptEvaluator:
    """Systematic prompt evaluation."""
    
    def __init__(self, prompt: str, model: str = "gpt-5.4"):
        self.prompt = prompt
        self.model = model
        self.results = []
    
    def evaluate(self, test_cases: list[PromptTestCase]) -> dict:
        """Run evaluation on test cases."""
        correct = 0
        total = len(test_cases)
        
        for case in test_cases:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": case.input},
                ],
                temperature=0,
            )
            
            output = response.choices[0].message.content
            is_correct = self._compare(output, case.expected_output)
            
            self.results.append({
                "input": case.input,
                "expected": case.expected_output,
                "actual": output,
                "correct": is_correct,
                "category": case.category,
            })
            
            if is_correct:
                correct += 1
        
        return {
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
            "by_category": self._aggregate_by_category(),
        }
    
    def _compare(self, output: str, expected: str) -> bool:
        """Flexible comparison."""
        return output.strip().lower() == expected.strip().lower()
    
    def _aggregate_by_category(self) -> dict:
        """Aggregate results by category."""
        categories = {}
        for result in self.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"correct": 0, "total": 0}
            categories[cat]["total"] += 1
            if result["correct"]:
                categories[cat]["correct"] += 1
        
        for cat in categories:
            categories[cat]["accuracy"] = (
                categories[cat]["correct"] / categories[cat]["total"]
            )
        
        return categories

# Usage
evaluator = PromptEvaluator(FEW_SHOT_PROMPT)
results = evaluator.evaluate(TRIAGE_TEST_CASES)
print(f"Overall accuracy: {results['accuracy']:.1%}")
for category, metrics in results["by_category"].items():
    print(f"  {category}: {metrics['accuracy']:.1%}")
```

### 4.6.2 A/B Testing Prompts

```python
class PromptABTest:
    """A/B test two prompt variants."""
    
    def __init__(self, variant_a: str, variant_b: str):
        self.variant_a = variant_a
        self.variant_b = variant_b
    
    def run(self, test_cases: list, n_samples: int = 100) -> dict:
        """Run A/B test with statistical significance."""
        import random
        
        results_a = []
        results_b = []
        
        for case in test_cases[:n_samples]:
            # Randomly assign to variant
            if random.random() < 0.5:
                result = self._evaluate_prompt(self.variant_a, case)
                results_a.append(result)
            else:
                result = self._evaluate_prompt(self.variant_b, case)
                results_b.append(result)
        
        accuracy_a = sum(results_a) / len(results_a) if results_a else 0
        accuracy_b = sum(results_b) / len(results_b) if results_b else 0
        
        # Statistical significance (simplified)
        diff = abs(accuracy_a - accuracy_b)
        significant = diff > 0.05 and min(len(results_a), len(results_b)) > 30
        
        return {
            "variant_a_accuracy": accuracy_a,
            "variant_b_accuracy": accuracy_b,
            "difference": diff,
            "winner": "A" if accuracy_a > accuracy_b else "B",
            "statistically_significant": significant,
        }
    
    def _evaluate_prompt(self, prompt: str, case) -> bool:
        """Evaluate single prompt-case pair."""
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": case.input},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip().lower() == case.expected.strip().lower()
```

### 4.6.3 Evaluation Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Zero-shot accuracy | >75% | Evaluation dataset |
| Few-shot accuracy (3-shot) | >90% | Evaluation dataset |
| CoT accuracy (reasoning) | >85% | Evaluation dataset |
| Schema compliance | >99% | Pydantic validation |
| Emergency recall (medical) | >98% | Evaluation dataset |
| Guardrail bypass rate | <5% | Red-team testing |
| Prompt version drift | 0% | CI/CD prompt comparison |

---

## 4.7 Key Takeaways

1. **System prompts are architecture — version control them, test them, treat them as code.** A change in system prompt can change model behavior as dramatically as a code change. Version your prompts alongside your application code.

2. **Few shot provides the best quality-to-cost ratio for most tasks — three examples is usually sufficient.** Beyond five examples, marginal improvements rarely justify the token cost. Start with zero shot; add examples only when quality is insufficient.

3. **Chain of thought improves accuracy 15-40 percent on reasoning tasks at the cost of additional output tokens.** Use it for math, logic, and multi-step reasoning. Do not use it for simple factual queries where it adds latency without quality improvement.

4. **ReAct is the foundation of agent architectures — master this pattern.** The interleaving of reasoning (thinking) with acting (tool use) and observation (results) is how every agent system works. Understanding this pattern is essential for building agentic systems.

5. **Structured output requires validation — never trust JSON generation without schema checking.** API-level schema enforcement achieves 99% compliance. Pydantic validation is your safety net for the remaining 1%. Both are necessary.

6. **Prompt engineering is iterative — A/B test prompts in production, not just in development.** Statistical significance requires 100+ samples. Build prompt evaluation into your CI/CD pipeline.

7. **Guardrail prompts are fragile — use layered defense.** System prompt rules are the first layer, not the only layer. Output validation, content filtering, and human review are essential for production safety.

8. **Reflection prompting doubles cost but significantly improves quality for high-stakes outputs.** Use it for critical content: medical summaries, legal analysis, financial reports. Do not use it for low-stakes tasks.

9. **Self consistency improves accuracy 10-20 percent at 2-5x cost.** Use it when accuracy matters more than latency: medical triage, legal analysis, complex reasoning. Vote on the most common answer across multiple reasoning paths.

10. **Multi-step prompting scales cost and latency linearly with the number of steps.** Only decompose tasks that genuinely benefit from decomposition. For many tasks, a single well-crafted prompt outperforms a chain.

---

## 4.8 Further Reading

- **Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)** — The paper that established chain-of-thought prompting. Shows 40%+ accuracy improvements on arithmetic and commonsense reasoning tasks.

- **Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2023)** — The paper on ReAct. Describes the interleaving of reasoning traces and actions that underpins all modern agent architectures.

- **Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (2023)** — The paper on self-consistency voting. Shows 10-20% accuracy improvements by sampling multiple reasoning paths.

- **Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)** — The paper on reflection prompting. Describes how self-critique improves agent performance.

- **OpenAI, "Prompt Engineering Guide" (platform.openai.com/docs)** — Official documentation on system prompts, few-shot examples, and structured output. Essential reading for GPT-5.4 prompt design.

- **Anthropic, "Prompt Engineering Guide" (docs.anthropic.com)** — Official documentation on Claude prompt design, including best practices for system prompts and tool use.

- **Google, "Gemini Prompt Design Guide" (ai.google.dev)** — Official documentation on Gemini prompt design, including multi-modal prompting techniques.

- **"Designing Machine Learning Systems" by Chip Huyen** — Chapters on model evaluation and monitoring apply directly to prompt evaluation and A/B testing.

- **"Building LLM Apps" by Valentina Alto** — Practical guide to prompt engineering patterns for production applications. Covers guardrails, structured output, and evaluation.

- **"Prompt Engineering Guide" by DAIR.AI (promptingguide.ai)** — Comprehensive community resource covering all major prompting techniques with examples and research references.
