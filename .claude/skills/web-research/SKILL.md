---
name: web-research
description: Conduct web research with source verification and evidence gathering
version: 1.0.0
author: Multi-Agent System
category: research
tags: [research, web-search, verification, sources, evidence]
prerequisites: []
capabilities: [source-evaluation, fact-checking, conflict-detection]
output_format: evidence_brief
---

# Web Research Skill

You are an expert in **conducting web research** - gathering, verifying, and synthesizing information from online sources with proper source attribution.

## Research Process

### 1. Query Formulation

```
❌ BAD: "python stuff"
✅ GOOD: "python async programming best practices 2024"
```

### 2. Source Evaluation

| Source Type | Reliability | Verification |
|-------------|--------------|--------------|
| **Official docs** | High | Check currency date |
| **Reputable blogs** | Medium | Cross-reference |
| **Forums** | Low | Verify with authoritative sources |
| **Social media** | Very Low | Treat as rumors only |

### 3. Evidence Gathering

```markdown
## Claim: Python asyncio is faster than threading

### Sources:
1. **Official docs**: python.org/3/library/asyncio.html
   - States asyncio uses single-threaded cooperative multitasking

2. **Blog**: realpython.com/asyncio-python
   - Benchmarks show 3-5x speedup for I/O-bound tasks

3. **Forum**: StackOverflow discussion
   - Multiple devs confirm for I/O-bound use cases

### Conclusion:
- TRUE for I/O-bound tasks
- FALSE for CPU-bound tasks (use multiprocessing instead)
```

## Authoritative Sources by Domain

### Programming

- **Python**: docs.python.org, realpython.com
- **JavaScript**: developer.mozilla.org, javascript.info
- **Go**: go.dev/blog, golang.org/doc

### Cloud Platforms

- **AWS**: docs.aws.amazon.com, aws.amazon.com/blogs
- **Azure**: learn.microsoft.com/azure, azure.microsoft.com/blog
- **GCP**: cloud.google.com/docs, cloud.google.com/blog

### Security

- **OWASP**: owasp.org
- **CVE**: cve.mitre.org
- **NIST**: nist.gov/cyberframework

### Data Science

- **Pandas**: pandas.pydata.org/docs
- **Scikit-learn**: scikit-learn.org/stable
- **TensorFlow**: tensorflow.org/guide

## Verification Checklist

For each claim found:
- [ ] Source is authoritative or reputable
- [ ] Information is current (check dates)
- [ ] Claim is supported by multiple sources
- [ ] No contradictions found elsewhere
- [ ] Source provides evidence/reasoning
- [ ] Avoid confirmation bias

## Conflict Resolution

When sources disagree:

1. **Check currency**: Newer info usually wins
2. **Check authority**: Official docs trump blogs
3. **Check context**: Different use cases may apply
4. **Check consensus**: What do most sources say?
5. **Note the disagreement**: Document it in findings

```markdown
## Conflict: Asyncio vs Threading for Python

**Source A (python.org)**: Asyncio for I/O, threading for CPU
**Source B (blog)**: Asyncio always faster
**Source C (expert)**: Context-dependent

**Resolution**: Official docs (Source A) are most reliable.
Different use cases apply (I/O vs CPU bound).
```

## Research Templates

### Technology Research

```markdown
## [Technology Name] Research

### Overview
- What it is
- When it was introduced
- Current version

### Key Features
1. Feature 1 - Description
2. Feature 2 - Description
3. Feature 3 - Description

### Use Cases
- Primary use case
- Secondary use cases
- Anti-patterns (when NOT to use)

### Pros & Cons
**Pros:**
- Advantage 1
- Advantage 2

**Cons:**
- Disadvantage 1
- Disadvantage 2

### Alternatives
- Alternative 1 (comparison)
- Alternative 2 (comparison)

### Sources
- [Source 1](URL) - Date
- [Source 2](URL) - Date
```

### Best Practices Research

```markdown
## [Topic] Best Practices

### Practice 1: [Title]
**Why:** Explanation
**How:** Implementation
**Source:** Attribution

### Practice 2: [Title]
...

### Common Mistakes
1. Mistake 1 - Why it's wrong
2. Mistake 2 - Correct approach

### Tools & Resources
- Tool 1: Link
- Tool 2: Link
```

## Search Strategies

### Effective Search Queries

```
# For tutorials
"[technology] tutorial" site:official-docs.com

# For best practices
"[technology] best practices" 2024

# For troubleshooting
"[error message]" solutions stackoverflow.com

# For comparisons
"[technology A] vs [technology B]" comparison
```

### Search Operators

| Operator | Usage | Example |
|----------|-------|---------|
| `""` | Exact phrase | `"asyncio await"` |
| `-` | Exclude | `python -snake` |
| `site:` | Specific site | `asyncio site:python.org` |
| `filetype:` | File type | `python guide filetype:pdf` |
| `after:` | Date range | `python asyncio after:2023` |

## Citation Format

```markdown
According to the [Python documentation](https://docs.python.org/3/library/asyncio.html),
"asyncio is a library to write concurrent code using the async/await syntax."

**Multiple sources:**
- [Python docs](https://docs.python.org/3/library/asyncio.html)
- [Real Python guide](https://realpython.com/asyncio-python)
```

## Common Pitfalls

- ❌ Trusting outdated information (always check dates)
- ❌ Using forums as primary sources (verify with docs)
- ❌ Ignoring source bias (vendor blogs may be promotional)
- ❌ Cherry-picking sources (include diverse perspectives)
- ❌ Conflicting correlation with causation

## When to Use This Skill

Use **web-research** when:
- Learning new technologies
- Verifying technical claims
- Finding best practices
- Troubleshooting issues
- Comparing solutions
- Gathering evidence for decisions

## Output Format

Return research findings with:
1. **Executive Summary**: Key findings in 2-3 sentences
2. **Detailed Findings**: Organized by topic with citations
3. **Source List**: All sources with URLs and dates accessed
4. **Confidence Level**: High (verified by docs), Medium (blog consensus), Low (forum only)
5. **Gaps Identified**: What couldn't be verified
6. **Recommendations**: Based on research findings
