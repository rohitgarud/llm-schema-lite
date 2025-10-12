# Contributing to schema-lite

## Development Setup

1. Fork and clone the repository
2. Install dependencies: `make setup`
3. Create a branch: `git checkout -b feature/my-feature`
4. Make changes and add tests
5. Run tests: `make test`
6. Run linters: `make lint`
7. Commit with conventional commits: `git commit -m "feat: add new feature"`
8. Push and create PR

## Commit Convention

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

## Code Style

- Use `ruff format` for formatting
- Pass `ruff check` and `mypy` checks
- Add type hints to all functions
- Write tests for new features
