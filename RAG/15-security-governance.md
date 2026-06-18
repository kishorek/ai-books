# Chapter 15: Security and Governance

> **Last verified: June 2026.**

Enterprise RAG requires security, access control, audit logging, and compliance. This chapter covers the governance patterns that make RAG safe for enterprise use.

## Access Control

### RBAC (Role-Based Access Control)

Users are assigned roles (admin, editor, viewer). Documents are tagged with allowed roles. At query time, the retrieval pipeline filters documents by the user's role. A viewer sees only public documents. An editor sees public and internal documents. An admin sees everything.

### ABAC (Attribute-Based Access Control)

More granular than RBAC. Access is determined by attributes of the user (clearance level, department, location), the document (classification, owner, creation date), and the context (time of day, device type, location).

### Document-Level Permissions

Each document can have fine-grained access controls: specific users, specific groups, or specific roles. The retrieval pipeline checks permissions before returning results.

### Row-Level Security

For databases powering RAG, row-level security ensures queries only return rows the user has access to. PostgreSQL supports this natively. The retrieval pipeline passes the user context to the database, which enforces access at the query level.

## Multi-Tenancy Security

Tenant isolation is critical. Each tenant's data must be completely isolated — vectors, metadata, chat histories, and access controls. A query from Tenant A must never return results from Tenant B.

The strongest isolation is collection-per-tenant in the vector database. Filter-based isolation (shared collections with tenant metadata filters) is cheaper but relies on correct filter implementation — a bug could leak data across tenants.

## Audit Logging

Every access to every document must be logged: who accessed what, when, from where, and what they did (search, read, download). Audit logs are required for compliance (SOC2, HIPAA, GDPR) and for debugging access issues.

The audit log should be immutable and stored separately from the application database. It should capture sufficient detail to reconstruct any access pattern.

## Data Governance

### Data Retention

Documents and their embeddings must have retention policies. After the retention period, data is automatically deleted. GDPR requires this. The implementation: tag documents with expiry dates, run a periodic cleanup job, and cascade deletion to embeddings and indexes.

### PII Detection

Scan all indexed content for personally identifiable information. Detected PII should be flagged, anonymized, or excluded from indexing depending on policy. This prevents sensitive data from appearing in RAG responses.

### Data Lineage

Track where data came from, how it was transformed, and where it is used. When a document is updated, the system should know which embeddings need regeneration. When a response is generated, the system should know which source documents contributed.

## Compliance Mapping

| Regulation | Requirement | Implementation |
|-----------|-------------|----------------|
| GDPR | Data retention, right to deletion | Auto-expiry, deletion API, audit logging |
| HIPAA | PHI protection, encryption | PII detection, encryption at rest and in transit |
| SOC2 | Audit logging, access controls | Comprehensive logging, RBAC/ABAC |
| CCPA | Consumer data rights | Data export, opt-out mechanisms |

## Key Takeaways

- RBAC is the minimum — every document must have access controls
- ABAC provides fine-grained control for complex enterprise requirements
- Tenant isolation is mandatory for multi-tenant systems
- Audit logging is required for compliance — log every access
- PII detection prevents data leakage — scan all indexed content
- Data retention policies are legally required

## Further Reading

- OWASP Security Guidelines
- GDPR, HIPAA, and SOC2 compliance guides
