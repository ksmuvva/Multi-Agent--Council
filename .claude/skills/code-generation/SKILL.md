---
name: code-generation
description: Best practices and patterns for generating clean, secure, maintainable code
version: 1.0.0
author: Multi-Agent System
category: development
tags: [code, programming, security, testing, best-practices]
prerequisites: [programming-basics]
capabilities: [code-structure, security-patterns, error-handling, testing-strategies]
output_format: code_with_documentation
---

# Code Generation Skill

You are an expert in **generating production-quality code** that is clean, secure, maintainable, and well-tested.

## Code Quality Principles

### 1. Clean Code

- **Meaningful names**: Variables, functions, classes describe their purpose
- **Single responsibility**: Each function does one thing well
- **DRY principle**: Don't repeat yourself - extract common patterns
- **Comments**: Explain "why", not "what" - code should be self-documenting

### 2. Security First

```python
# ❌ BAD - Hardcoded credentials
password = "admin123"

# ✅ GOOD - Environment variables
password = os.getenv("DB_PASSWORD")
if not password:
    raise ValueError("DB_PASSWORD not set")
```

### 3. Error Handling

```python
# Specific exceptions, not bare except
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise
```

### 4. Type Hints

```python
from typing import List, Optional

def process_items(
    items: List[str],
    threshold: Optional[int] = None
) -> dict[str, int]:
    """Process items and return statistics."""
    # Implementation...
```

## Language-Specific Guidelines

### Python

- Follow PEP 8 style guide
- Use type hints (Python 3.9+)
- Prefer dataclasses over classes for data containers
- Use context managers for resource management
- Avoid mutable default arguments

### JavaScript/TypeScript

- Use `const` and `let`, never `var`
- Prefer arrow functions for callbacks
- Use template literals for string interpolation
- Async/await over promise chains
- JSDoc for function documentation

### Go

- Error handling: never ignore errors
- Use goroutines sparingly
- Prefer interfaces over concrete types
- Keep goroutines lightweight
- Use channels for communication

## Code Structure

### File Organization

```
src/
├── __init__.py
├── models.py          # Data models
├── services.py         # Business logic
├── repositories.py     # Data access
├── schemas.py          # Validation schemas
├── utils.py            # Utilities
└── config.py           # Configuration
```

### Function Template

```python
def function_name(
    param1: type,
    param2: type,
    optional_param: type = default_value,
) -> return_type:
    """Brief description of function.

    Args:
        param1: Description of param1
        param2: Description of param2
        optional_param: Description of optional param

    Returns:
        Description of return value

    Raises:
        SpecificException: When condition occurs

    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        'output'
    """
    # Implementation
    pass
```

## Security Checklist

- [ ] No hardcoded credentials
- [ ] Input validation on all user inputs
- [ ] Parameterized queries for database
- [ ] Output encoding to prevent XSS
- [ ] Proper error handling (no info leakage)
- [ ] Principle of least privilege for permissions
- [ ] Secrets in environment variables
- [ ] Dependency security scanning

## Testing Guidelines

### Unit Tests

```python
def test_function_success_case():
    """Test the happy path."""
    result = function_name("valid_input", 10)
    assert result.status == "success"

def test_function_edge_case():
    """Test edge conditions."""
    result = function_name("", 0)
    assert result.status == "invalid"

def test_function_error_handling():
    """Test error cases."""
    with pytest.raises(ValueError):
        function_name("invalid", -1)
```

### Test Coverage

- Aim for 80%+ coverage on critical paths
- Test success cases, edge cases, and error cases
- Mock external dependencies
- Use fixtures for common test data

## Performance Considerations

1. **Algorithm choice**: Use appropriate data structures
2. **Caching**: Cache expensive operations
3. **Lazy loading**: Load resources only when needed
4. **Batching**: Group operations to reduce overhead
5. **Profiling**: Measure before optimizing

## When to Use This Skill

Use **code-generation** when:
- Writing implementation code
- Refactoring existing code
- Adding new features
- Creating API endpoints
- Building data pipelines

## Output Format

Generate code with:
1. File path as comment header
2. Imports organized (stdlib, third-party, local)
3. Type hints throughout
4. Docstrings for functions/classes
5. Example usage in comments
6. Error handling
7. Logging where appropriate
8. Test examples
