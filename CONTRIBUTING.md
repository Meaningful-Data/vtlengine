# Contributing to vtlengine

Thanks for your interest in contributing! This guide covers the basics to get you productive quickly.

## Getting Started
- Use Python 3.9+.
- Install dependencies with poetry:
  ```bash
  poetry install
  ```
- Activate the virtual environment:
  ```bash
  poetry shell
  ```

## Development Workflow
1) Branch from `main` and use a descriptive branch name.
2) Make changes with tests and type hints in mind.
3) Run mandatory quality checks (must pass before any commit/PR):
   ```bash
   poetry run ruff format src/
   poetry run ruff check --fix src/
   poetry run mypy src/
   ```
4) Run the full test suite (must be green before finishing an issue/PR):
   ```bash
   poetry run pytest tests/
   ```
5) Keep diffs small and focused; include relevant test updates/data fixtures where needed.

## Project Conventions
- VTL grammar lives in `src/vtlengine/AST/Grammar/`; regenerate via `antlr4 -Dlanguage=Python3 -visitor Vtl.g4` (ANTLR 4.9.x). Do not hand-edit generated `lexer.py`, `parser.py`, `tokens.py`.
- Operators follow the `validate/compute` pattern in `src/vtlengine/Operators/` with strict type checks before execution.
- Identifiers cannot be nullable; measures may be. Role definitions are enforced in `Model`.
- For SDMX integrations, use `run_sdmx` with proper `VtlDataflowMapping` when multiple datasets are present.

## Submitting Changes
- Open a Pull Request against `main` with a clear description of the change and testing performed.
- Reference related issues and include screenshots/logs if relevant.
- Be responsive to review feedback; keep discussions respectful (see Code of Conduct).

## Questions?
Open a GitHub issue or discuss in the PR. Security issues: follow the SECURITY.md instructions.
