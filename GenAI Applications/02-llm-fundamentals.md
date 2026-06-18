# Chapter 2: Practical LLM Fundamentals

> "An architect who does not understand tokenization will build systems that fail on edge cases. An architect who does not understand temperature will build systems that produce inconsistent outputs. An architect who does not understand attention will not know why long-context retrieval degrades."

---

Last verified: June 2026. Verify current model specifications at provider documentation.

You do not need to implement a transformer from scratch. You do need to understand enough about how LLMs work to make good architecture decisions. This chapter provides the mental model to reason about tokenization costs, attention constraints, sampling behavior, and the parameters that control model output. Every concept in this chapter maps directly to an architectural decision you will make.

The central thesis of this chapter is that **LLM behavior is controllable through parameters, and the choice of parameters is an architectural decision, not a tuning exercise**. Temperature, top-p, max tokens, and system prompts are not knobs to twiddle — they are architectural constraints that define the behavior envelope of your system. Set them deliberately, version them, and test against them.

---

## 2.1 How LLMs Generate Text

The core mechanism of large language models is deceptively simple: given a sequence of tokens, predict the next token. The model assigns a probability distribution over all possible next tokens and selects one based on its sampling strategy. This process repeats — each generated token is appended to the sequence, and the model predicts the next one again — until it produces a stop token or reaches the output limit.

The critical architectural implication is that text generation is sequential. Each token requires a full forward pass through the model. Generating 100 tokens takes approximately ten times longer than generating 10. This is why generation latency scales linearly with output length, and why streaming is not optional for user-facing applications.

### 2.1.1 The Two Phases of Inference

When you send a prompt to an LLM, two distinct phases occur:

**Prefill phase** processes all input tokens in parallel. This is compute-bound and scales linearly with input length. For a 4,000-token prompt, the prefill phase might take 200-400ms. Multiple prefill operations can be batched efficiently on GPU hardware.

**Decoding phase** generates tokens one at a time, each depending on all previous tokens. This is memory-bandwidth-bound and scales linearly with output length. For 500 output tokens at 80 tokens/second, decoding takes approximately 6 seconds. Each token requires a full forward pass through the model.

| Phase | Bound By | Scaling | Optimization |
|-------|----------|---------|-------------|
| Prefill | Compute (FLOPS) | Linear with input length | Batch multiple requests |
| Decode | Memory bandwidth | Linear with output length | Streaming, speculative decoding |

This distinction matters for scaling decisions. You can batch multiple prefill operations efficiently, but decode operations are sequential. It also explains why streaming is technically necessary: without it, users wait the full generation time before seeing any output.

```python
import time
from openai import OpenAI

client = OpenAI()

def measure_inference_phases(prompt: str, max_tokens: int = 500):
    """Measure prefill and decode phases separately."""
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
    )
    
    first_token_time = None
    token_count = 0
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time() - start
            token_count += 1
    
    total_time = time.time() - start
    decode_time = total_time - first_token_time
    
    return {
        "prefill_ms": first_token_time * 1000,
        "decode_ms": decode_time * 1000,
        "total_ms": total_time * 1000,
        "tokens": token_count,
        "tokens_per_second": token_count / decode_time if decode_time > 0 else 0,
    }

# Typical results for GPT-5.4:
# Short prompt (100 tokens input):
#   Prefill: 150ms, Decode: 5800ms, 86 tokens/sec
# Long prompt (4000 tokens input):
#   Prefill: 380ms, Decode: 5900ms, 85 tokens/sec
# Insight: prefill scales with input length; decode does not
```

### 2.1.2 Tokenization and Its Pitfalls

Tokenization splits text into subword units using algorithms like Byte Pair Encoding (BPE). Modern tokenizers handle English well but vary significantly across languages and domains. The token count determines both cost and context consumption.

```python
import tiktoken

def analyze_tokenization(text: str, label: str = ""):
    """Detailed tokenization analysis for cost estimation."""
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    
    # Decode individual tokens to show boundaries
    token_strings = [enc.decode([t]) for t in tokens]
    
    print(f"\n{'='*50}")
    print(f"Text: {label}")
    print(f"Characters: {len(text)}")
    print(f"Tokens: {len(tokens)}")
    print(f"Tokens/Character: {len(tokens)/len(text):.2f}")
    print(f"Cost at $2.50/1M: ${len(tokens) * 2.50 / 1_000_000:.6f}")
    print(f"Token breakdown: {token_strings[:10]}...")
    return len(tokens)

# English prose: ~0.75 tokens/character
analyze_tokenization(
    "The patient presents with acute chest pain radiating to the left arm",
    "English medical note"
)

# Japanese text: ~1.20 tokens/character  
analyze_tokenization(
    "患者は左腕に放射する急性胸痛を訴えている",
    "Japanese medical note"
)

# Python code: ~0.55 tokens/character
analyze_tokenization(
    "def calculate_risk_score(patient_data: dict) -> float:",
    "Python function"
)

# JSON data: ~0.60 tokens/character
analyze_tokenization(
    '{"patient_id": "P-2025-88431", "risk_score": 0.87}',
    "JSON record"
)
```

The architectural implications are significant:

| Text Type | Tokens/Char | Relative Cost | Budget Impact |
|-----------|-------------|---------------|---------------|
| English prose | 0.75 | 1.0x | Baseline |
| Japanese text | 1.20 | 1.6x | 60% cost increase |
| Python code | 0.55 | 0.73x | 27% cost decrease |
| JSON data | 0.60 | 0.80x | 20% cost decrease |
| Mathematical notation | 0.90 | 1.2x | 20% cost increase |
| Markdown | 0.65 | 0.87x | 13% cost decrease |

A system processing Japanese medical records costs roughly 60 percent more per character than one processing English medical records. A system processing code documentation costs 27 percent less. These are not marginal differences — at scale, they compound into significant budget impacts.

### 2.1.3 Attention Mechanism and Context Limits

Attention is the mechanism that allows each token to "look at" all other tokens in the sequence and weigh their relevance. The computational cost is quadratic in sequence length — doubling the context window quadruples the attention compute and memory.

```
Attention computation: O(n² × d)
Where:
  n = sequence length (context window)
  d = model dimension (hidden size)

At 128K tokens:
  Attention matrix: 128K × 128K = 16.4 billion entries
  Memory (FP16): ~32 GB just for attention weights
  
At 1M tokens:
  Attention matrix: 1M × 1M = 1 trillion entries
  Memory (FP16): ~2 TB (requires parallelism across GPUs)
```

This is why context windows have practical limits, and why models degrade at extended lengths even within their advertised window. The "lost in the middle" effect is real: research shows that models attend more strongly to the beginning and end of long contexts, with information in the middle receiving less attention.

```python
def demonstrate_lost_in_the_middle():
    """
    Show how information placement affects retrieval accuracy.
    In a 128K context, accuracy varies by position.
    """
    # Simulated accuracy based on Liu et al. (2023)
    positions = {
        "beginning (0-10%)": 0.92,
        "early-middle (10-30%)": 0.85,
        "middle (30-70%)": 0.72,  # Lowest accuracy
        "late-middle (70-90%)": 0.84,
        "end (90-100%)": 0.91,
    }
    
    print("Retrieval Accuracy by Position in Context:")
    print("-" * 50)
    for position, accuracy in positions.items():
        bar = "█" * int(accuracy * 40)
        print(f"  {position:25s} {bar} {accuracy:.0%}")
    
    # Key insight: middle positions have 20% lower accuracy
    # Architectural response: place critical information at beginning/end
```

**Positional encoding** determines how the model understands token order. RoPE (Rotary Position Embeddings) and ALiBi enable context window extension beyond training length, but quality still degrades. Always test at your actual context length, not the advertised maximum.

| Context Utilization | Effective Capacity | Architectural Response |
|--------------------|--------------------|----------------------|
| 128K window | ~50-65K useful tokens | Design prompts for beginning/end placement |
| 1M window | ~300-500K useful tokens | Chunk and rerank; do not rely on full window |
| 10M window | ~3-5M useful tokens | Hierarchical retrieval; structured placement |

---

## 2.2 Key Parameters That Control Behavior

### 2.2.1 Temperature and Top P

Temperature and top-p control the randomness of text generation. These are not independent — they interact to define the sampling distribution.

**Temperature** scales the logits before softmax:
- Temperature = 0: Always picks the most likely token (deterministic)
- Temperature = 1: Standard sampling (model's trained distribution)
- Temperature > 1: Increased randomness (more creative/diverse)
- Temperature < 1: Decreased randomness (more focused/conservative)

**Top P** (nucleus sampling) limits sampling to the top P percent of probability mass:
- Top P = 0.1: Only sample from tokens comprising the top 10% of probability
- Top P = 0.9: Sample from tokens comprising the top 90% of probability
- Top P = 1.0: No filtering (all tokens possible)

```python
from openai import OpenAI

client = OpenAI()

def compare_sampling_strategies(prompt: str, n_samples: int = 5):
    """Show how temperature affects output consistency."""
    temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]
    
    for temp in temperatures:
        responses = []
        for _ in range(n_samples):
            response = client.chat.completions.create(
                model="gpt-5.4",
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=100,
            )
            responses.append(response.choices[0].message.content[:80])
        
        unique = len(set(responses))
        print(f"Temperature {temp:.1f}: {unique}/{n_samples} unique responses")
        for i, r in enumerate(responses[:2]):
            print(f"  Sample {i+1}: {r}")

# Example output for "Explain quantum computing":
# Temperature 0.0: 1/5 unique (deterministic)
# Temperature 0.3: 2/5 unique (focused but varied)
# Temperature 0.7: 4/5 unique (balanced creativity)
# Temperature 1.0: 5/5 unique (diverse)
# Temperature 1.5: 5/5 unique (very diverse, may be incoherent)
```

The choice of temperature is an architectural decision:

| Use Case | Temperature | Top P | Rationale |
|----------|-------------|-------|-----------|
| Factual Q&A | 0.0 | 1.0 | Deterministic, reproducible |
| Code generation | 0.0-0.2 | 0.95 | Mostly deterministic, slight variation |
| Customer support | 0.3-0.5 | 0.9 | Consistent but natural-sounding |
| Content creation | 0.7-0.9 | 0.95 | Creative, varied output |
| Brainstorming | 0.9-1.2 | 0.98 | Maximum diversity |
| Data extraction | 0.0 | 1.0 | Schema compliance requires determinism |

A customer support chatbot at temperature 0.7 will give different answers to the same question, which may confuse users. Factual systems should use temperature zero. Creative systems benefit from higher values. Make temperature a configurable parameter in your system, not a hardcoded value.

### 2.2.2 Max Tokens and Stop Sequences

**Max tokens** limits the maximum output length. Setting this too low truncates responses. Setting this too high wastes tokens and increases latency. The optimal value depends on the task:

```python
# Token budget allocation for different tasks
TASK_TOKEN_BUDGETS = {
    "classification": {"max_tokens": 10, "rationale": "Single label + confidence"},
    "extraction": {"max_tokens": 200, "rationale": "Structured JSON output"},
    "summarization": {"max_tokens": 500, "rationale": "Concise summary"},
    "chat_response": {"max_tokens": 1000, "rationale": "Conversational response"},
    "code_generation": {"max_tokens": 2000, "rationale": "Full function with docs"},
    "analysis": {"max_tokens": 3000, "rationale": "Detailed analysis with sections"},
}
```

**Stop sequences** terminate generation at specific patterns. They are essential for structured output:

```python
response = client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "List 5 risks:"}],
    stop=["\n\n", "6."],  # Stop at double newline or before item 6
    max_tokens=500,
)
```

### 2.2.3 Reasoning Tokens

Modern reasoning models like DeepSeek R1 use internal reasoning tokens — tokens the model generates for itself before producing the visible response. These hidden tokens cost money and add latency but dramatically improve accuracy on complex tasks.

| Model Type | Cost Ratio | Latency Ratio | Math Accuracy | Code Accuracy |
|------------|-----------|---------------|---------------|---------------|
| Standard (GPT-5.4) | 1x | 1x | 75% | 82% |
| Reasoning (DeepSeek R1) | 2-5x | 2-5x | 95% | 94% |
| Reasoning (GPT-5.5) | 2-3x | 2-3x | 93% | 91% |

Reasoning models cost two to five times more and are two to five times slower, but they improve math accuracy from about 60-80 percent to 90-99 percent. The architectural response is the model router pattern: route simple queries to cheap, fast models and complex reasoning tasks to expensive reasoning models.

```python
class ModelRouter:
    """Route queries to appropriate models based on complexity."""
    
    def __init__(self):
        self.complexity_indicators = [
            "calculate", "prove", "derive", "optimize",
            "multi-step", "reasoning", "logic", "proof",
            "compare and contrast", "trade-offs between",
        ]
    
    def classify_complexity(self, query: str) -> str:
        query_lower = query.lower()
        complexity_score = sum(
            1 for indicator in self.complexity_indicators
            if indicator in query_lower
        )
        
        if complexity_score >= 2:
            return "reasoning"  # DeepSeek R1 or GPT-5.5
        elif complexity_score == 1:
            return "standard"   # GPT-5.4
        else:
            return "simple"     # GPT-5.4 nano or Gemini Flash
    
    def route(self, query: str) -> dict:
        complexity = self.classify_complexity(query)
        models = {
            "simple": {
                "model": "gpt-5.4-nano",
                "cost_per_1m": 0.20,
                "latency_ms": 100,
            },
            "standard": {
                "model": "gpt-5.4",
                "cost_per_1m": 2.50,
                "latency_ms": 300,
            },
            "reasoning": {
                "model": "deepseek-r1",
                "cost_per_1m": 0.55,
                "latency_ms": 800,
            },
        }
        return {"complexity": complexity, **models[complexity]}

# Cost comparison at 1M queries/day:
# Without routing (all GPT-5.4): $2.50 × 1M × 30 = $75,000/month
# With routing (60% simple, 30% standard, 10% reasoning):
#   Simple: $0.20 × 600K × 30 = $3,600
#   Standard: $2.50 × 300K × 30 = $22,500
#   Reasoning: $0.55 × 100K × 30 = $1,650
#   Total: $27,750/month (63% reduction)
```

### 2.2.4 System Prompt Design

System prompts define the model's behavior, role, and constraints. They are processed first and have the strongest influence on output. A change in system prompt can change model behavior as dramatically as a code change.

```python
SYSTEM_PROMPT_STRUCTURE = {
    "role": "Who the model is (expertise, persona)",
    "rules": "What to do and not do (constraints)",
    "format": "Structure of response (headers, JSON, etc.)",
    "tone": "Style of communication (formal, casual, technical)",
    "examples": "Few-shot examples if needed",
}

# Production system prompt template
FINANCIAL_ANALYST_PROMPT = """You are a senior financial analyst with 15 years of experience in equity research.

RULES:
- Always cite sources for data points
- Never provide specific investment advice (buy/sell/hold)
- Include risk factors in every analysis
- Distinguish between factual data and your analysis
- If uncertain, state your uncertainty explicitly

FORMAT:
## Analysis
[Your analysis with citations]

## Key Metrics
[Table of key financial metrics]

## Risk Factors
[Bullet list of risks]

## Confidence
[Your confidence level: High/Medium/Low with reasoning]

TONE: Professional, data-driven, objective. Avoid hype and sensationalism."""
```

---

## 2.3 Context Window Budgeting

Every token in the context window must be allocated. Context window budgeting is a first-class architectural concern — allocate tokens explicitly, not arbitrarily.

### 2.3.1 Allocation Framework

A typical allocation for a 128K token window:

```python
class ContextBudget:
    """Manage token allocation across context window components."""
    
    def __init__(self, total_window: int = 128_000):
        self.total = total_window
        self.allocations = {
            "system_prompt": 500,
            "chat_history": 40_000,
            "retrieved_documents": 50_000,
            "user_query": 200,
            "reserved_output": 32_000,
            "buffer": 5_300,
        }
        self._validate()
    
    def _validate(self):
        used = sum(self.allocations.values())
        if used > self.total:
            raise ValueError(
                f"Allocations ({used}) exceed window ({self.total}). "
                f"Over by {used - self.total} tokens."
            )
    
    def available_for_documents(self, query_tokens: int) -> int:
        """Calculate how many tokens are available for retrieved docs."""
        return (
            self.total
            - self.allocations["system_prompt"]
            - self.allocations["chat_history"]
            - query_tokens
            - self.allocations["reserved_output"]
            - self.allocations["buffer"]
        )
    
    def report(self):
        print(f"Context Window Budget ({self.total:,} tokens)")
        print("-" * 45)
        for component, tokens in self.allocations.items():
            pct = tokens / self.total * 100
            bar = "█" * int(pct / 2)
            print(f"  {component:25s} {tokens:>7,} ({pct:4.1f}%) {bar}")

budget = ContextBudget(128_000)
budget.report()
# Context Window Budget (128,000 tokens)
# ---------------------------------------------
#   system_prompt                500 (  0.4%) ▏
#   chat_history              40,000 ( 31.3%) ███████████████▌
#   retrieved_documents       50,000 ( 39.1%) ███████████████████▌
#   user_query                  200 (  0.2%) ▏
#   reserved_output           32,000 ( 25.0%) ████████████
#   buffer                     5,300 (  4.1%) ██
```

### 2.3.2 Token Budget Optimization Strategies

The key principle is that every token used for one purpose is a token not available for another. Extensive chat history means less room for retrieved documents. A long system prompt means less room for context.

| Strategy | Token Savings | Quality Impact | Implementation |
|----------|--------------|----------------|----------------|
| Summarize chat history | 50-70% | Minimal | LLM-based summarization every N turns |
| Compress retrieved chunks | 40-60% | Low | Smaller chunks, more targeted retrieval |
| Cache system prompt | 100% (cached) | None | Anthropic prompt caching, OpenAI prefix caching |
| Remove low-relevance docs | 20-40% | Low | Reranking with score threshold |
| Use shorter prompts | 10-30% | None | Concise language, remove redundancy |

```python
class ChatHistoryManager:
    """Optimize chat history token usage."""
    
    def __init__(self, max_history_tokens: int = 40_000):
        self.max_tokens = max_history_tokens
        self.history: list[dict] = []
        self.summary: str = ""
    
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
    
    def get_context(self) -> list[dict]:
        """Return optimized chat context."""
        current_tokens = self._estimate_tokens(self.history)
        
        if current_tokens <= self.max_tokens:
            return self.history
        
        # Strategy 1: Keep last 5 turns + summary
        recent = self.history[-10:]  # 5 turns = 10 messages
        older = self.history[:-10]
        
        if older:
            self.summary = self._summarize(older)
        
        context = []
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.summary}"
            })
        context.extend(recent)
        
        return context
    
    def _summarize(self, messages: list[dict]) -> str:
        """Summarize older messages to save tokens."""
        # In production, use LLM to generate summary
        combined = " ".join(m["content"] for m in messages)
        # Simplified: in practice, call LLM for quality summarization
        return f"Summary of {len(messages)} earlier messages: {combined[:200]}..."
    
    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Rough token estimation (1.3 tokens per word)."""
        total_words = sum(len(m["content"].split()) for m in messages)
        return int(total_words * 1.3)
```

---

## 2.4 Streaming and User Experience

Streaming is technically necessary for user-facing applications. Without it, users wait the full generation time before seeing any output. With streaming, users see the first token within the prefill time (150-400ms) and subsequent tokens arrive at the model's generation speed.

### 2.4.1 Streaming Implementation

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def stream_response(prompt: str, system_prompt: str = ""):
    """Stream LLM response with real-time token delivery."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    stream = await client.chat.completions.create(
        model="gpt-5.4",
        messages=messages,
        stream=True,
        max_tokens=1000,
    )
    
    full_response = []
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response.append(token)
            # In production: send token to WebSocket/SSE
            yield token
    
    return "".join(full_response)

# Streaming metrics to monitor:
# - Time to first token (TTFT): should be < 500ms
# - Inter-token interval: should be < 100ms
# - Total generation time: task-dependent
# - Tokens per second: 60-100 for GPT-5.4
```

### 2.4.2 Streaming vs. Non-Streaming Trade-offs

| Metric | Streaming | Non-Streaming |
|--------|-----------|---------------|
| Time to first token | 150-400ms | Full generation time |
| User perception | Feels instant | Feels slow |
| Implementation complexity | Higher (SSE/WebSocket) | Lower (single response) |
| Error handling | Partial response recovery | Full retry |
| Token counting | Requires accumulation | Single response |
| Best for | User-facing chat | Backend processing |

---

## 2.5 Scaling Laws and Model Quality

Chinchilla scaling laws describe the optimal relationship between model size and training data: roughly 20 tokens per parameter. A 70B model trained on only 100B tokens may perform worse than a 7B model trained on 1T tokens. Parameter count alone is misleading — check benchmarks, not just size.

### 2.5.1 Emergent Capabilities

Emergent capabilities appear at certain model scales but not smaller ones. Chain-of-thought reasoning, code generation, and few-shot learning all emerged at specific scales. You cannot predict which capabilities a model will have — testing is essential.

| Capability | Emergence Scale | Implication |
|-----------|----------------|-------------|
| In-context learning | ~1B parameters | Few-shot prompting works |
| Chain-of-thought | ~10B parameters | Reasoning improves with scale |
| Code generation | ~50B parameters | Functional code from natural language |
| Tool use | ~100B parameters | Function calling becomes reliable |
| Multimodal reasoning | ~200B parameters | Cross-modal understanding |

### 2.5.2 Model Quality Evaluation

Benchmarks provide a baseline, but domain-specific evaluation is essential:

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class EvaluationCase:
    input: str
    expected_output: str
    category: str
    difficulty: Literal["easy", "medium", "hard"]
    weight: float = 1.0

@dataclass
class ModelEvaluation:
    model_name: str
    cases: list[EvaluationCase] = field(default_factory=list)
    
    def evaluate(self, model_fn) -> dict:
        """Run evaluation and compute metrics."""
        results = {"correct": 0, "total": 0, "by_category": {}}
        
        for case in self.cases:
            output = model_fn(case.input)
            is_correct = self._compare(output, case.expected_output)
            
            results["total"] += 1
            if is_correct:
                results["correct"] += 1
            
            if case.category not in results["by_category"]:
                results["by_category"][case.category] = {"correct": 0, "total": 0}
            results["by_category"][case.category]["total"] += 1
            if is_correct:
                results["by_category"][case.category]["correct"] += 1
        
        results["accuracy"] = results["correct"] / results["total"]
        return results
    
    def _compare(self, output: str, expected: str) -> bool:
        """Flexible comparison for evaluation."""
        # Normalize whitespace and case
        return output.strip().lower() == expected.strip().lower()

# Example: Build evaluation dataset
eval_dataset = ModelEvaluation(model_name="gpt-5.4")
eval_dataset.cases = [
    EvaluationCase(
        input="What is the capital of France?",
        expected_output="Paris",
        category="geography",
        difficulty="easy",
    ),
    EvaluationCase(
        input="Calculate 15% of $84,500",
        expected_output="$12,675",
        category="math",
        difficulty="medium",
    ),
    # Add 500+ cases for production evaluation
]
```

---

## 2.6 Case Study: Token Budget Optimization

### 2.6.1 Problem Statement

A customer support system was processing 500K queries per day. The initial architecture sent full chat history (20 turns, 4,000 tokens), retrieved documents (10 chunks at 500 tokens each, 6,000 tokens), system prompt (800 tokens), and user query (150 tokens) to GPT-5.4. Average cost was $0.012 per query — $180K per month.

### 2.6.2 Root Cause Analysis

The root cause was not the model choice but the context management:

| Component | Tokens | % of Context | Problem |
|-----------|--------|-------------|---------|
| Chat history (20 turns) | 4,000 | 36.5% | Full content from every turn |
| Retrieved documents (10 × 500) | 6,000 | 54.8% | Chunks too large, many irrelevant |
| System prompt | 800 | 7.3% | Reasonable |
| User query | 150 | 1.4% | Reasonable |
| **Total input** | **10,950** | **100%** | |

The model was receiving 10,950 input tokens per query. At GPT-5.4 pricing ($2.50/1M input tokens), that is $0.0274 per query for input alone. With output tokens, the total was $0.012 per query.

### 2.6.3 Optimization Strategy

Three changes were implemented:

**1. Chat history summarization.** Instead of sending all 20 turns, keep the last 5 turns and summarize earlier context. This reduces chat history from 4,000 to 1,200 tokens.

**2. Chunk size optimization.** Reduce retrieved chunks from 500 to 150 tokens. More chunks but smaller — better precision, less noise. This reduces document tokens from 6,000 to 3,000.

**3. Selective retrieval.** Only fetch documents relevant to the current query, not all potentially related documents. This reduces retrieved chunks from 10 to 5.

```python
class OptimizedContextBuilder:
    """Build optimized context for cost reduction."""
    
    def __init__(self, budget: int = 3500):
        self.budget = budget  # Target input tokens
    
    def build_context(
        self,
        query: str,
        chat_history: list[dict],
        retrieved_docs: list[str],
    ) -> list[dict]:
        """Build context within token budget."""
        context = []
        
        # 1. System prompt (500 tokens)
        context.append({"role": "system", "content": self.system_prompt})
        
        # 2. Summarized chat history (1200 tokens)
        summary = self._summarize_history(chat_history, max_tokens=1200)
        context.append({"role": "system", "content": summary})
        
        # 3. Optimized retrieved docs (1500 tokens)
        docs = self._select_relevant_docs(retrieved_docs, query, max_tokens=1500)
        context.append({"role": "system", "content": f"Relevant documents:\n{docs}"})
        
        # 4. User query (150 tokens)
        context.append({"role": "user", "content": query})
        
        return context
    
    def _summarize_history(self, history: list, max_tokens: int) -> str:
        """Summarize older messages, keep recent ones."""
        if len(history) <= 10:  # 5 turns
            return str(history)
        
        older = history[:-10]
        recent = history[-10:]
        
        # Summarize older messages
        summary = f"Earlier conversation ({len(older)} messages): "
        summary += " ".join(m["content"][:50] for m in older[:5])
        summary += "..."
        
        return summary
    
    def _select_relevant_docs(self, docs: list, query: str, max_tokens: int) -> str:
        """Select most relevant documents within token budget."""
        # In production: use reranker with score threshold
        selected = []
        current_tokens = 0
        
        for doc in docs:
            doc_tokens = len(doc.split()) * 1.3  # Rough estimate
            if current_tokens + doc_tokens > max_tokens:
                break
            selected.append(doc)
            current_tokens += doc_tokens
        
        return "\n\n".join(selected)
```

### 2.6.4 Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Input tokens per query | 10,950 | 3,200 | 71% reduction |
| Cost per query | $0.012 | $0.004 | 67% reduction |
| Monthly cost | $180,000 | $60,000 | $120,000 savings |
| Answer quality (accuracy) | 91% | 93% | +2 percentage points |
| P50 latency | 2.1s | 1.4s | 33% faster |

The insight that matters: reducing context did not just save money — it improved quality. Less noise means the model focuses on relevant information. The model was being distracted by irrelevant chat history and low-relevance document chunks.

### 2.6.5 Cost Breakdown Comparison

| Component | Before (tokens) | Before (cost/1M) | After (tokens) | After (cost/1M) | Savings |
|-----------|----------------|------------------|----------------|-----------------|---------|
| System prompt | 800 | $0.002 | 500 | $0.00125 | 37.5% |
| Chat history | 4,000 | $0.010 | 1,200 | $0.003 | 70% |
| Retrieved docs | 6,000 | $0.015 | 1,500 | $0.00375 | 75% |
| User query | 150 | $0.000375 | 150 | $0.000375 | 0% |
| Output (500 tokens) | 500 | $0.0075 | 500 | $0.0075 | 0% |
| **Total** | **11,450** | **$0.0349** | **3,850** | **$0.0159** | **54.5%** |

---

## 2.7 Testing LLM Fundamentals

### 2.7.1 Tokenization Tests

```python
import pytest

def test_token_count_within_budget():
    """Verify token count stays within context window budget."""
    prompt = build_prompt(user_query="What is the policy?")
    token_count = count_tokens(prompt)
    assert token_count <= 128_000, f"Prompt exceeds context window: {token_count}"

def test_language_cost_estimation():
    """Verify cost estimation accounts for language differences."""
    english_tokens = count_tokens("The patient presents with chest pain")
    japanese_tokens = count_tokens("患者は胸痛を訴えている")
    
    # Japanese should cost more per character
    assert japanese_tokens > english_tokens
    
    # But both should be within reasonable bounds
    assert english_tokens < 20
    assert japanese_tokens < 30
```

### 2.7.2 Sampling Behavior Tests

```python
def test_temperature_zero_is_deterministic():
    """Verify temperature 0 produces identical outputs."""
    responses = []
    for _ in range(3):
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            temperature=0.0,
        )
        responses.append(response.choices[0].message.content)
    
    assert len(set(responses)) == 1, "Temperature 0 should be deterministic"

def test_max_tokens_truncation():
    """Verify max tokens limits output length."""
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "Write a 1000 word essay"}],
        max_tokens=50,
    )
    token_count = count_tokens(response.choices[0].message.content)
    assert token_count <= 60, f"Output exceeded max_tokens: {token_count}"
```

### 2.7.3 Evaluation Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Token count accuracy | <5% variance | Compare tiktoken vs actual API |
| Temperature consistency | 100% at temp=0 | Multiple calls, compare outputs |
| Streaming TTFT | <500ms p95 | Production monitoring |
| Context budget compliance | 100% | Automated prompt size checks |
| Model routing accuracy | >90% | Evaluation dataset |

---

## 2.8 Key Takeaways

1. **LLMs generate text one token at a time — each token requires a full forward pass, making latency scale with output length.** Streaming is not optional for user-facing applications. Measure and optimize time-to-first-token separately from total generation time.

2. **Tokenization varies dramatically by language and domain — budget 2-6x more for non-English or code-adjacent text.** Japanese text costs 60 percent more per character than English. Mathematical notation costs 20 percent more. Build token cost estimates into your architecture.

3. **Attention is O(n²) — context windows have real computational costs, not just token limits.** The "lost in the middle" effect means models use only 30-50 percent of large context windows effectively. Design for the smallest window that meets your requirements.

4. **Temperature and top-p control output randomness — match them to the task.** Temperature 0 for factual and extraction tasks. Temperature 0.3-0.5 for conversational tasks. Temperature 0.7+ for creative tasks. Make these configurable, not hardcoded.

5. **Context window budgeting is a first-class architectural concern — allocate tokens explicitly, not arbitrarily.** Every token used for one purpose is a token not available for another. Summarize chat history, compress retrieved documents, and cache system prompts.

6. **Reasoning models are 2-5x more expensive but dramatically better at complex tasks — use model routing.** The model router pattern reduces cost by 60-80 percent compared to using reasoning models universally, while maintaining quality on the tasks that need them.

7. **The two phases of inference (prefill and decode) scale differently — optimize each independently.** Prefill scales with input length and can be batched. Decode scales with output length and is sequential. This is why streaming matters and why input optimization has outsized impact.

8. **Reducing context improves quality, not just cost.** Less noise means the model focuses on relevant information. The case study showed accuracy improved from 91 to 93 percent after context optimization — a quality gain from a cost reduction.

9. **Max tokens and stop sequences are precision tools — use them to control output shape.** Setting max tokens too low truncates responses. Setting too high wastes tokens. Use stop sequences to enforce structured output boundaries.

10. **Test tokenization, sampling, and context budget as you test any other component.** Token count variance, temperature consistency, and streaming performance are all testable and should be part of your CI/CD pipeline.

---

## 2.9 Further Reading

- **Vaswani et al., "Attention Is All You Need" (2017)** — The transformer paper. Section 3.2 describes the self-attention mechanism in detail. Essential for understanding why attention is O(n²) and why context windows have computational limits.

- **Kaplan et al., "Scaling Laws for Neural Language Models" (2020)** — The original scaling laws paper. Describes the relationship between model size, data, and compute. Foundation for understanding why bigger models are better and when they are not.

- **Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022)** — The Chinchilla paper. Corrected the scaling laws to show optimal token-to-parameter ratios. Essential for understanding why parameter count alone is misleading.

- **Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (2023)** — Empirical evidence for the lost-in-the-middle effect. Directly applicable to context window design and information placement strategies.

- **Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020)** — The paper on nucleus (top-p) sampling. Explains the theoretical foundation for temperature and top-p parameters.

- **OpenAI Tokenizer Tool** (platform.openai.com/tokenizer) — Visualize tokenization for GPT models. Essential for understanding token counts and cost estimation.

- **Anthropic Prompt Caching Documentation** (docs.anthropic.com) — How to use prompt caching to reduce costs by 90% for cached tokens. Directly applicable to system prompt optimization.

- **"Language Model Evaluation" by LMSYS** — Comprehensive benchmark suite for evaluating LLM capabilities. Essential for building domain-specific evaluation datasets.

- **Tiktoken Library** (github.com/openai/tiktoken) — Fast BPE tokenizer for GPT models. Essential for token counting and cost estimation in production systems.

- **"Build a Large Language Model from Scratch" by Sebastian Raschka** — Step-by-step implementation of a transformer-based LLM. Provides the implementation understanding that complements the architectural perspective in this chapter.
