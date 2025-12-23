# Contributing to Jupiter

Thank you for your interest in contributing to Jupiter!

## How to Contribute

### Reporting Bugs

1. Check that the bug hasn't been reported before
2. Create an issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (macOS/Linux, chip, RAM)

### Proposing Features

1. Open an issue describing the feature
2. Discuss the design before implementing
3. Reference the issue in your PR

### Pull Requests

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Add tests if applicable
5. Make sure tests pass: `pytest`
6. Format the code: `black . && ruff check --fix .`
7. Commit: `git commit -m "feat: description"`
8. Push: `git push origin feature/my-feature`
9. Open a Pull Request

### Code Style

- We use [Black](https://black.readthedocs.io/) for formatting
- We use [Ruff](https://docs.astral.sh/ruff/) for linting
- Maximum 100 characters per line
- Docstrings in English

### Commits

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new functionality
- `fix:` bug fix
- `docs:` documentation
- `refactor:` refactoring
- `test:` tests
- `chore:` maintenance

## Contribution Areas

### High Priority

- [ ] Automated tests
- [ ] PyTorch model implementation
- [ ] More example domains
- [ ] Performance optimizations

### Ideas

- Support for more tokenizer backends
- Real-time metrics visualization
- Weights & Biases integration
- GGUF export for llama.cpp

## Local Development

```bash
# Clone
git clone https://github.com/raym33/jupiter.git
cd jupiter

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
ruff check --fix .
```

## Questions

Open an issue with the `question` label.

Thank you for contributing!
