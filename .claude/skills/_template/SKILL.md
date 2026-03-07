---
name: template-skill
description: A template skill for creating custom Claude Agent Skills. Copy this directory and modify as needed.
version: 1.0.0
author: Your Name
tags: [template, example]
---

# Template Skill

This is a template for creating custom Claude Agent Skills. Copy this entire directory to create a new skill.

## How to Create a New Skill

1. **Copy this template directory**:
   ```bash
   cp -r .claude/skills/_template .claude/skills/your-skill-name
   ```

2. **Edit the SKILL.md file**:
   - Update the YAML frontmatter (name, description, version, etc.)
   - Write your skill instructions below the frontmatter
   - Include examples and references if needed

3. **Add references (optional)**:
   - Create a `references/` subdirectory
   - Add any supporting documents, code examples, etc.

4. **Test your skill**:
   - The skill will be auto-discovered by the Claude Agent SDK
   - Agents can invoke it via the Skill tool

## Skill Instructions

Write your skill instructions here. These instructions will be loaded when an agent invokes this skill.

## Best Practices

- **Be specific**: Clear, actionable instructions work best
- **Include examples**: Show, don't just tell
- **Define scope**: What the skill can and cannot do
- **Handle edge cases**: What to do when things go wrong
- **Return format**: Specify expected output format

## Example

```python
# Example code or configuration
def example_function():
    return "Hello from the skill!"
```

## References

Add any reference materials in the `references/` subdirectory.
