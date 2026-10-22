# Contributing to PredictGuard AI

Thank you for your interest in contributing to **PredictGuard AI**! This document outlines our development process, coding standards, and contribution guidelines.

---

## 1. Code of Conduct

We are committed to providing a welcoming, inclusive, and professional environment for all contributors. Please be respectful and constructive in all issues and pull requests.

---

## 2. Local Development Setup

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `python -m venv`
- Git

### Initializing Environment

```bash
# Clone the repository
git clone https://github.com/your-username/PredictGuard-AI.git
cd PredictGuard-AI

# Create and activate virtual environment with Python 3.11
uv venv .venv --python 3.11

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install runtime and development dependencies
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

---

## 3. Development Workflow

### Running Locally
```bash
uvicorn app.main:app --reload --port 8000
```
Visit `http://localhost:8000` to access the operations console and `http://localhost:8000/docs` for the interactive Swagger documentation.

### Code Style & Formatting
We enforce strict linting and formatting standards using **Ruff** and type-checking with **Mypy**:

```bash
# Lint checks
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Format checks
ruff format --check .

# Static type checking
mypy app
```

### Pre-commit Hooks
Install pre-commit hooks to automatically check your code prior to committing:
```bash
pip install pre-commit
pre-commit install
```

---

## 4. Running Tests

Always ensure the entire test suite passes before submitting a pull request:

```bash
# Run pytest with code coverage
pytest tests --cov=app --cov-report=term-missing

# Run legacy integration test suite
python test_api_endpoints.py
```

Coverage must remain at or above **70%**.

---

## 5. Pull Request Guidelines

1. **Branch Naming**: Use descriptive branch names:
   - `feature/description`
   - `fix/bug-name`
   - `docs/topic`
2. **Commit Messages**: Write clear, imperative commit messages (e.g., `Add tool wear regression endpoint`).
3. **Tests**: Every new feature or bug fix must be covered with tests in `tests/unit` or `tests/integration`.
4. **CI Checks**: Ensure all GitHub Actions workflow checks pass cleanly.
