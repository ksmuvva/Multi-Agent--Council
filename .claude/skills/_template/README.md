# Creating Custom Skills

This template helps you create custom Claude Agent Skills for the Multi-Agent Reasoning System.

## Quick Start

1. **Copy the template**:
   ```bash
   cp -r .claude/skills/_template .claude/skills/my-skill
   ```

2. **Edit SKILL.md**:
   - Update YAML frontmatter
   - Write skill instructions
   - Add examples

3. **Use the skill**:
   - Auto-discovered by SDK
   - Invoke via Skill tool in agent prompts

## SKILL.md Format

### Required Frontmatter

```yaml
---
name: my-skill
description: Brief description of what this skill does
version: 1.0.0
author: Your Name
tags: [category, keywords]
---
```

### Content

After the frontmatter, write your skill instructions in Markdown. Include:

- What the skill does
- How to use it
- Examples and patterns
- Edge cases and limitations
- Expected output format

## Optional: References Directory

Create a `references/` subdirectory for supporting materials:

```
.claude/skills/my-skill/
├── SKILL.md
└── references/
    ├── example1.md
    ├── example2.py
    └── patterns.json
```

## Skill Discovery

Skills are auto-discovered from:
- `.claude/skills/*/SKILL.md` (project skills)
- `~/.claude/skills/*/SKILL.md` (user skills)

Agents invoke skills via the Skill tool:
```
Use the "my-skill" skill to help with this task.
```

## Best Practices

1. **Single purpose**: Each skill should do one thing well
2. **Clear instructions**: Be explicit about what the skill does
3. **Examples matter**: Show concrete examples
4. **Version control**: Tag your skills with versions
5. **Test iteratively**: Validate skills work as expected

## Examples

See the built-in skills for reference:
- `multi-agent-reasoning/` - Core reasoning patterns
- `code-generation/` - Code creation patterns
- `web-research/` - Research and fact-finding
- `document-creation/` - Document generation

## Need Help?

- See FR-027: Skill Authoring Template
- Check existing skills in `.claude/skills/`
- Review the Agent Skills standard: https://agentskills.io
