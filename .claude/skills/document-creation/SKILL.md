---
name: document-creation
description: Generate professional documents in various formats (DOCX, PDF, XLSX, PPTX)
version: 1.0.0
author: Multi-Agent System
category: documentation
tags: [documents, formatting, docx, pdf, xlsx, pptx, reports]
prerequisites: [basic-writing]
capabilities: [document-structure, professional-formatting, multi-format-output]
output_format: formatted_document
---

# Document Creation Skill

You are an expert in **creating professional documents** in multiple formats with proper structure, formatting, and style.

## Supported Formats

| Format | Best For | Key Features |
|--------|----------|--------------|
| **DOCX** | Editable documents | Rich text, styles, headers/footers |
| **PDF** | Final versions | Preserved formatting, cross-platform |
| **XLSX** | Data/spreadsheets | Formulas, charts, pivot tables |
| **PPTX** | Presentations | Slides, animations, speaker notes |
| **Markdown** | Technical docs | Code blocks, tables, cross-references |

## Document Structure

### Standard Sections

1. **Title Page**
   - Document title
   - Author/organization
   - Date
   - Version number

2. **Table of Contents**
   - Auto-generated from headings
   - Page numbers
   - Hyperlinked navigation

3. **Executive Summary**
   - Brief overview (1 page max)
   - Key findings
   - Recommendations

4. **Introduction**
   - Purpose and scope
   - Background context
   - Methodology (if applicable)

5. **Main Content**
   - Organized with clear headings
   - Tables, charts, diagrams
   - Code examples (for technical docs)

6. **Conclusion**
   - Summary of key points
   - Next steps
   - Appendices (if needed)

## Best Practices

### Typography

- Use one font family throughout
- Headings: 16-24pt, bold
- Body: 10-12pt, regular
- Line spacing: 1.15-1.5
- Margins: 1 inch minimum

### Visual Elements

- **Tables**: Clear headers, alternating row colors
- **Figures**: Numbered with captions below
- **Code blocks**: Monospace font, syntax highlighting
- **Diagrams**: Clear labels, consistent styling

### Cross-References

```markdown
See Section 3.2 for implementation details.
Refer to Figure 4.1 for the system architecture.
As shown in Table 2.3, performance improved by 40%.
```

## Format-Specific Guidelines

### DOCX (Microsoft Word)

- Use styles (Heading 1, Heading 2, Normal)
- Insert page breaks between sections
- Add header with document title
- Add footer with page numbers
- Insert table of contents from references

### PDF

- Set page size (Letter, A4)
- Embed fonts for portability
- Add bookmarks for navigation
- Optimize images for web/email
- Enable accessibility tags

### XLSX (Excel)

- Use first row as headers
- Freeze header row
- Format numbers appropriately
- Add data validation for inputs
- Create summary sheet with formulas

### PPTX (PowerPoint)

- One main idea per slide
- 6x6 rule: max 6 bullet points, 6 words each
- High contrast for readability
- Use slide master for consistency
- Add speaker notes

## Templates

### Technical Report Template

```markdown
# [Title]

**Author:** [Name]
**Date:** [Date]
**Version:** [1.0]

## Executive Summary
[Brief overview]

## 1. Introduction
### 1.1 Purpose
### 1.2 Scope
### 1.3 Background

## 2. Methodology
[Approach used]

## 3. Findings
### 3.1 [Key Finding 1]
### 3.2 [Key Finding 2]

## 4. Recommendations
1. [Recommendation 1]
2. [Recommendation 2]

## 5. Conclusion
[Summary]

## Appendix
- A: [Additional detail]
- B: [Supporting data]
```

### Presentation Template

```markdown
Slide 1: Title Slide
- Title
- Subtitle
- Author
- Date

Slide 2: Agenda
- Overview
- Topics to cover
- Timeline

Slide 3-N: Content Slides
- Main point (heading)
- 3-5 supporting bullets
- Diagram/chart if applicable

Final Slide: Summary
- Key takeaways
- Next steps
- Contact info
```

## Language & Style

### Tone

- Professional and objective
- Active voice preferred
- Present tense for facts
- Past tense for methods/results

### Common Mistakes

- ❌ Overuse of jargon
- ❌ Inconsistent terminology
- ❌ Wall of text (no breaks)
- ❌ Unclear section boundaries
- ❌ Missing page numbers

### Writing Tips

- ✅ Use short sentences (15-20 words)
- ✅ One concept per paragraph
- ✅ Transition words between sections
- ✅ Concrete examples
- ✅ Visual aids for complex concepts

## When to Use This Skill

Use **document-creation** when:
- Creating reports or proposals
- Generating documentation
- Building presentations
- Exporting data to spreadsheets
- Formatting technical specifications

## Output Format

Return documents with:
1. Proper file extension (.docx, .pdf, etc.)
2. Structured sections with headings
3. Professional formatting
4. Tables and figures numbered
5. Cross-references where appropriate
6. Metadata (title, author, date)
