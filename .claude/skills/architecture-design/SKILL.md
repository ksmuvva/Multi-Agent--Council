---
name: architecture-design
description: Design software architectures with patterns, trade-offs, and documentation
version: 1.0.0
author: Multi-Agent System
category: architecture
tags: [architecture, design, patterns, scalability, trade-offs]
prerequisites: [system-design, software-engineering]
capabilities: [architectural-patterns, technology-selection, documentation]
output_format: architecture_document
---

# Architecture Design Skill

You are an expert in **software architecture design** - creating system architectures that are scalable, maintainable, secure, and aligned with business goals.

## Architecture Principles

### SOLID Principles

```python
# S - Single Responsibility
class UserService:
    """Handles user-related operations only."""
    def create_user(self, data): pass
    def get_user(self, id): pass
    # NOT: send_email(), create_order()

# O - Open/Closed
class PaymentProcessor:
    def process_payment(self, payment):
        for processor in self.processors:
            if processor.can_handle(payment):
                return processor.process(payment)
    # Open for extension (add processors), closed for modification

# L - Liskov Substitution
class SQLRepository(Repository):
    # Can substitute for parent Repository
    def save(self, entity): pass

# I - Interface Segregation
class ReadableUserRepository:
    def get_user(self, id): pass

class WritableUserRepository:
    def save_user(self, user): pass
    # Clients depend only on what they need

# D - Dependency Inversion
class UserService:
    def __init__(self, repository: Repository):
        # Depend on abstraction, not concretion
        self.repository = repository
```

### Architectural Patterns

### Layered Architecture

```
┌─────────────────────────────────┐
│      Presentation Layer         │  UI, API controllers
│      (Controllers, Views)       │
└────────────┬────────────────────┘
             │
┌────────────┴────────────────────┐
│      Business Logic Layer       │  Domain logic, rules
│      (Services, Domain)         │
└────────────┬────────────────────┘
             │
┌────────────┴────────────────────┐
│      Data Access Layer          │  Database, external APIs
│      (Repositories, ORM)         │
└────────────┬────────────────────┘
             │
┌────────────┴────────────────────┐
│      Database / External        │
└────────────────────────────────┘
```

### Microservices

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│   API    │  │   API    │  │   API    │
│ Gateway  │→ │ Gateway  │→ │ Gateway  │
└─────┬────┘  └─────┬────┘  └─────┬────┘
      │            │            │
┌─────┴────┐  ┌────┴─────┐  ┌────┴─────┐
│ Service A│  │ Service B│  │ Service C│
└──────────┘  └──────────┘  └──────────┘
     │            │            │
┌────┴────────┴────────────┴────┐
│   Event Bus / Message Queue   │
└───────────────────────────────┘
```

## Technology Selection

### Decision Framework

```markdown
## [Technology] Selection

### Options
1. **Option A**: Description - Pros/Cons
2. **Option B**: Description - Pros/Cons
3. **Option C**: Description - Pros/Cons

### Criteria
- Performance
- Scalability
- Team expertise
- Ecosystem support
- Licensing cost
- Vendor lock-in

### Decision
Selected: **Option A**

### Rationale
- Best performance for our use case
- Team has existing expertise
- Strong open-source community
- Avoids vendor lock-in

### Trade-offs
- Con: Steeper learning curve
- Mitigation: Training plan
```

### Database Selection

| Database | Best For | Avoid When |
|----------|----------|------------|
| **PostgreSQL** | Complex queries, ACID compliance | Massive simple reads |
| **MySQL** | Web apps, wide ecosystem | Complex JSON operations |
| **MongoDB** | Flexible schemas, document storage | Multi-record transactions |
| **Redis** | Caching, sessions, pub/sub | Primary data store |
| **Elasticsearch** | Full-text search | Primary data store |

### Cloud Platform Selection

| Platform | Strengths | Considerations |
|----------|-----------|----------------|
| **AWS** | Broadest services, mature | Cost complexity |
| **Azure** | Enterprise integration, hybrid | Learning curve |
| **GCP** | Data/analytics, Kubernetes | Smaller ecosystem |
| **DigitalOcean** | Simple, predictable pricing | Limited enterprise features |

## Scalability Patterns

### Horizontal Scaling

```python
# Stateless services enable horizontal scaling
# Load balancer distributes requests

# Bad: Session state in memory
@app.route("/profile")
def profile():
    user_id = session["user_id"]  # Tied to this server
    return get_user(user_id)

# Good: Session in shared store (Redis)
@app.route("/profile")
def profile():
    user_id = get_session_from_store(session_token)
    return get_user(user_id)  # Any server can handle
```

### Vertical Scaling

- Increase server resources (CPU, RAM)
- Easier to implement
- Has limits (max server size)
- Single point of failure risk

### Caching Strategies

```python
# Cache-Aside pattern
def get_user_data(user_id):
    # Try cache first
    data = cache.get(f"user:{user_id}")
    if data:
        return data

    # Cache miss - get from DB
    data = db.query_user(user_id)

    # Store in cache for next time
    cache.set(f"user:{user_id}", data, ttl=300)

    return data
```

## Security Architecture

### Defense in Depth

```
┌─────────────────────────────────┐
│      DDoS Protection           │
│      (Cloudflare, AWS Shield)   │
└────────────┬────────────────────┘
             │
┌────────────┴────────────────────┐
│      WAF / Web Firewall         │
│      (ModSecurity, AWS WAF)      │
└────────────┬────────────────────┘
             │
┌────────────┴────────────────────┐
│      Rate Limiting              │
│      (Application-level)         │
└────────────┬────────────────────┘
             │
┌────────────┴────────────────────┐
│      Authentication             │
│      (MFA, OAuth, SSO)           │
└────────────┬────────────────────┘
             │
┌────────────┴────────────────────┐
│      Authorization              │
│      (RBAC, ABAC)                │
└────────────┬────────────────────┘
             │
┌────────────┴────────────────────┐
│      Data Encryption             │
│      (At rest, in transit)        │
└─────────────────────────────────┘
```

### Security Best Practices

1. **Never trust client input**: Validate everything
2. **Principle of least privilege**: Minimal required access
3. **Defense in depth**: Multiple security layers
4. **Security by default**: Secure defaults, not opt-in
5. **Encrypt sensitive data**: At rest and in transit

## Documentation

### C4 Model

```markdown
# System Architecture

## 1. Context Diagram
Shows system boundaries and external entities

## 2. Container Diagram
Shows applications and how they communicate

## 3. Component Diagram
Shows components within each application

## 4. Code/Class Diagram
Shows classes and their relationships

## 5. Sequence Diagrams
Shows message flow for key scenarios
```

### Architecture Decision Records (ADRs)

```markdown
# ADR-001: Use Microservices Architecture

## Status
Accepted

## Context
Monolithic application becoming hard to maintain and deploy.

## Decision
Adopt microservices architecture.

## Consequences
**Positive:**
- Independent deployment
- Technology diversity
- Team autonomy

**Negative:**
- Increased complexity
- Network latency
- Distributed transactions
```

## Trade-off Analysis

### CAP Theorem

For distributed systems, pick two:

| System | Consistency | Availability | Partition Tolerance |
|--------|-------------|--------------|---------------------|
| **RDBMS** | ✓ | ✗ | ✓ |
| **NoSQL** | ✗ | ✓ | ✓ |
| **NewSQL** | ✓ | ✓ | ✗ (some trade-off) |

### Latency vs Throughput

| Approach | Latency | Throughput | Use Case |
|----------|---------|-----------|----------|
| **Sync processing** | Low | Low | Real-time |
| **Async queue** | High | High | Batch jobs |
| **Event streaming** | Medium | Very High | Analytics |

## When to Use This Skill

Use **architecture-design** when:
- Starting new projects
- Refactoring existing systems
- Evaluating technologies
- Planning scalability
- Designing integrations
- Documenting architecture

## Output Format

Generate architecture documents with:
1. **System Context**: Goals, scope, stakeholders
2. **Architecture Overview**: High-level design
3. **Component Diagrams**: Visual representations
4. **Technology Choices**: Rationale and trade-offs
5. **Data Flows**: Sequence diagrams for key flows
6. **Deployment Architecture**: Infrastructure design
7. **Security Model**: Authentication, authorization, data protection
8. **Scalability Plan**: Growth strategy
9. **ADRs**: Key architectural decisions with rationale
