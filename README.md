# AI Systems Study Pack

This folder is organized as a clean study pack for low-level AI systems practice and learning.

## Structure

- STUDY_GUIDE.md: Roadmap only (no embedded implementation code)
- IMPLEMENTATION_CHECKLIST.md: What to implement and verify
- PRACTICE_QUESTIONS.md: Questions grouped by phase and difficulty
- ops/: Python implementation modules
- scratch/: Standalone exploratory scripts preserved for reference
- tests/: Minimal correctness tests

## Validation

- Syntax check:
  - python -m compileall ops tests
- Run tests:
  - python -m pytest -q tests

## Notes

- Keep implementation code in ops/*.py only.
- Keep exploratory one-off scripts in scratch/.
- Keep guide files Markdown-only.
