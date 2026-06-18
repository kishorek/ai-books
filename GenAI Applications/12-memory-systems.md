# Chapter 12: Memory Systems

Memory is what separates a stateless API call from an intelligent assistant. Without memory, every interaction starts from scratch. Memory systems give applications continuity, personalization, and context across sessions.

## Memory Types

**Short-term memory** holds recent conversation context within a session. It is fast (Redis), ephemeral (expires after inactivity), and bounded (limited by token budget). The key challenge is trimming: keeping the most relevant recent messages while staying within the context window.

**Long-term memory** holds persistent knowledge across sessions. It is slower (vector database), permanent (until explicitly deleted), and searchable (semantic retrieval). The key challenge is relevance: retrieving the right memories for the current query from potentially thousands of stored entries.

**Episodic memory** stores specific past events and interactions. Unlike general long-term memory, episodic memory preserves the context of what happened — the user's situation, what was tried, and what worked. This enables the system to recall "last time this user had this issue, we solved it by..."

**Semantic memory** stores general knowledge and facts about the user or domain. Preferences, relationships, patterns — the distilled understanding that accumulates over time.

## Storage Architecture

The storage architecture follows a tiered pattern. Redis handles short-term memory with sub-millisecond latency and automatic expiration. A vector database (Pinecone, Qdrant, or pgvector) handles long-term semantic retrieval. Postgres handles structured metadata and relationships. An optional graph database handles relationship-rich knowledge.

The critical design decision is what to store and when. Not every interaction is worth persisting. The system needs to evaluate importance — a factual question might be worth storing if the user asks repeatedly, while a greeting is not.

## Memory Retrieval

Memory retrieval is fundamentally a RAG problem. Given the current query, find relevant past interactions, user facts, and episodic memories.

The retrieval process combines multiple sources: recent conversation (short-term, always included), semantically relevant memories (long-term vector search), relevant past episodes (episodic memory search), and user facts (semantic memory lookup). These are assembled into context alongside the current query.

## Memory Consolidation

Short-term memories need to be periodically consolidated into long-term storage. The consolidation process evaluates each short-term memory for importance, extracts key facts and patterns, stores significant memories in the vector database, and clears short-term storage.

Without consolidation, short-term memory fills up and older context is lost. With consolidation, important information persists across sessions.

## Privacy and Compliance

Memory systems raise privacy concerns. Stored conversations may contain sensitive information. Users have rights to deletion (GDPR). Data retention policies must be implemented. Access controls must prevent cross-user memory leakage.

The architecture should support: user-requested deletion of all memories, automatic expiration of old memories, encryption of stored content, and access logging for audit.

## Case Study: Personalized Customer Support

A SaaS company implemented memory for personalized support. Short-term memory maintained conversation context within a session. Long-term memory stored relevant past issues. Episodic memory recalled previous solutions. Semantic memory tracked user preferences.

Results: resolution rate improved from 72 to 89 percent, average handle time dropped from 6 to 3.5 minutes, and customer satisfaction increased from 3.6 to 4.4 out of 5. The most impactful improvement was in repeat issue handling — the system recalled previous solutions instead of starting fresh.

## Key Takeaways

- Short-term memory (Redis) handles session context; long-term memory (vector DB) handles persistent knowledge
- Memory consolidation is essential — periodically move important short-term memories to long-term storage
- Memory retrieval is a RAG problem — use semantic search to find relevant past interactions
- Privacy matters — implement data retention policies, user deletion rights, and access controls
- Memory quality degrades — implement forgetting mechanisms for stale or low-importance memories

## Further Reading

- Mem0 and Zep documentation — memory layers for LLM applications
- LangGraph memory documentation
