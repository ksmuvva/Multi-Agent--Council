---
name: test-case-generation
description: Generate comprehensive test cases with edge cases and coverage analysis
version: 1.0.0
author: Multi-Agent System
category: testing
tags: [testing, quality-assurance, test-cases, edge-cases, coverage]
prerequisites: [code-understanding]
capabilities: [test-design, edge-case-identification, coverage-planning]
output_format: test_suite
---

# Test Case Generation Skill

You are an expert in **generating comprehensive test suites** that cover happy paths, edge cases, error conditions, and integration scenarios.

## Testing Philosophy

### Test Pyramid

```
        /\
       /  \        E2E Tests (10%)
      /____\
     /      \      Integration Tests (30%)
    /________\
   /          \    Unit Tests (60%)
  /______________\
```

### Coverage Goals

- **Unit tests**: 80%+ line coverage
- **Integration tests**: All critical paths
- **E2E tests**: Key user journeys only

## Test Case Categories

### 1. Happy Path Tests

```python
def test_user_registration_success():
    """Test successful user registration."""
    user_data = {
        "email": "test@example.com",
        "password": "SecurePass123!",
        "name": "Test User"
    }
    result = register_user(user_data)
    assert result.status == "success"
    assert result.user_id is not None
    assert result.email_verified is False
```

### 2. Edge Cases

```python
def test_user_registration_with_existing_email():
    """Test registration with duplicate email."""
    existing_user = create_test_user()
    result = register_user({
        "email": existing_user.email,
        "password": "SecurePass123!",
        "name": "Test User"
    })
    assert result.status == "error"
    assert result.error_code == "EMAIL_EXISTS"

def test_user_registration_with_minimal_password():
    """Test registration with minimum valid password."""
    result = register_user({
        "email": "test@example.com",
        "password": "A1b!",  # Minimum valid
        "name": "Test User"
    })
    assert result.status in ["success", "weak_password"]
```

### 3. Error Cases

```python
def test_user_registration_invalid_email():
    """Test registration with invalid email."""
    result = register_user({
        "email": "not-an-email",
        "password": "SecurePass123!",
        "name": "Test User"
    })
    assert result.status == "error"
    assert result.error_code == "INVALID_EMAIL"

def test_user_registration_missing_fields():
    """Test registration with missing required fields."""
    result = register_user({
        "email": "test@example.com"
        # Missing password and name
    })
    assert result.status == "error"
    assert result.error_code == "MISSING_FIELDS"
```

### 4. Security Tests

```python
def test_sql_injection_prevention():
    """Test that SQL injection is prevented."""
    malicious_input = "'; DROP TABLE users; --"
    result = search_users(malicious_input)
    # Should not cause database error
    assert result.status == "success"
    assert len(result.users) == 0

def test_password_hashing():
    """Test that passwords are properly hashed."""
    user = create_test_user(password="plaintext123")
    stored_hash = get_user_password_hash(user.id)
    assert stored_hash != "plaintext123"
    assert stored_hash.startswith("$2b$")  # bcrypt
```

## Test Organization

### File Structure

```
tests/
├── unit/
│   ├── test_models.py
│   ├── test_services.py
│   └── test_utils.py
├── integration/
│   ├── test_api.py
│   ├── test_database.py
│   └── test_auth.py
├── e2e/
│   ├── test_user_journeys.py
│   └── test_workflows.py
└── fixtures/
    ├── test_data.json
    └── test_config.py
```

### Fixtures

```python
import pytest

@pytest.fixture
def test_user():
    """Create a test user."""
    user = User(
        email="test@example.com",
        name="Test User"
    )
    user.set_password("testpass123")
    user.save()
    return user

@pytest.fixture
def authenticated_client(test_user):
    """Create an authenticated test client."""
    client = APIClient()
    client.force_authenticate(user=test_user)
    return client
```

## Test Naming Conventions

```python
# Good: Descriptive, uses snake_case
def test_user_can_login_with_valid_credentials()
def test_api_returns_404_for_nonexistent_resource()
def test_calculates_correct_interest_rate()

# Bad: Vague, uses camelCase
def testLogin()
def TestAPI()
def interestCalculation()
```

## Coverage Analysis

### Coverage Types

| Type | Description | Example |
|------|-------------|---------|
| **Line** | Each line of code executed | All branches of if/else |
| **Branch** | Each decision point | True and false conditions |
| **Path** | All possible paths | All combinations of branches |
| **Function** | Each function called | Public and private methods |

### Coverage Targets

```python
# Minimum coverage by file type
COVERAGE_TARGETS = {
    "models.py": 90,
    "services.py": 85,
    "views.py": 80,
    "utils.py": 95,
    "__init__.py": 70
}
```

## Test Data Management

### Factory Pattern

```python
import factory

class UserFactory(factory.Factory):
    class Meta:
        model = User

    email = factory.Sequence(lambda n: f"user{n}@example.com")
    name = "Test User"
    password = factory.PostGenerationMethodCall(
        'set_password', 'defaultpassword123'
    )

# Usage in tests
def test_with_multiple_users():
    users = UserFactory.create_batch(5)
    assert len(users) == 5
```

### Test Data Files

```yaml
# tests/fixtures/test_scenarios.yaml
scenarios:
  - name: happy_path
    input:
      amount: 100
      rate: 0.05
    expected:
      result: 105.0

  - name: edge_case_zero_amount
    input:
      amount: 0
      rate: 0.05
    expected:
      result: 0.0
```

## Integration Test Patterns

### API Testing

```python
def test_api_endpoint_pagination():
    """Test API pagination works correctly."""
    response = client.get('/api/users?page=1&per_page=10')
    assert response.status_code == 200
    data = response.json()
    assert len(data['items']) == 10
    assert data['page'] == 1
    assert data['total_pages'] > 1
```

### Database Testing

```python
def test_database_transaction_rollback():
    """Test that transactions roll back on error."""
    with pytest.raises(IntegrityError):
        with transaction.atomic():
            user = User.objects.create(email="test@example.com")
            # This will violate unique constraint
            User.objects.create(email="test@example.com")

    # User should not exist due to rollback
    assert not User.objects.filter(email="test@example.com").exists()
```

## When to Use This Skill

Use **test-case-generation** when:
- Writing new code (write tests alongside)
- Reviewing code coverage
- Finding edge cases
- Preparing for releases
- Quality assurance planning

## Output Format

Generate test suites with:
1. Proper file organization
2. Descriptive test names
3. Fixtures for common setup
4. Happy path + edge cases + error cases
5. Security tests where applicable
6. Expected assertions clearly defined
7. Comments explaining complex scenarios
8. Coverage analysis summary
