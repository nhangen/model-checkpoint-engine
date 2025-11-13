# Pre-Commit Hook Setup Guide

## Overview

This project uses **pre-commit** hooks to maintain code quality, consistency, and security. Pre-commit runs automated checks before each commit to catch issues early in the development process.

## What Gets Checked

Our pre-commit configuration includes:

### 1. Code Formatting
- **Black**: Automatically formats Python code to PEP 8 standards
- **isort**: Organizes and sorts import statements

### 2. Code Quality & Linting
- **Flake8**: Checks for style violations, code smells, and potential bugs
  - Includes plugins: docstrings, bugbear, comprehensions
- **PyDocStyle**: Validates docstring conventions (Google style)

### 3. Type Checking
- **MyPy**: Static type checking for type hints and annotations

### 4. Security
- **Bandit**: Scans for common security issues in Python code
- **Safety**: Checks dependencies for known security vulnerabilities
- **Private Key Detection**: Prevents committing sensitive keys

### 5. File Quality
- **Trailing Whitespace**: Removes trailing whitespace
- **End of File Fixer**: Ensures files end with a newline
- **Mixed Line Endings**: Normalizes to LF line endings
- **Large File Check**: Prevents committing files larger than 1MB
- **YAML/JSON/TOML Validation**: Checks syntax of config files

### 6. Python-Specific
- **AST Validation**: Ensures valid Python syntax
- **Debug Statement Detection**: Catches leftover debug statements
- **Test Naming**: Validates test file naming conventions
- **Merge Conflict Detection**: Catches unresolved merge conflicts

## Installation

### Step 1: Install Development Dependencies

```bash
# Install all dev dependencies including pre-commit
pip install -e ".[dev]"
```

Or install just pre-commit:

```bash
pip install pre-commit
```

### Step 2: Install Git Hooks

```bash
# Install the pre-commit hooks into your .git/hooks/
pre-commit install
```

This creates a `.git/hooks/pre-commit` script that runs automatically before each commit.

### Step 3: (Optional) Install Commit Message Hook

```bash
# Also check commit messages
pre-commit install --hook-type commit-msg
```

## Usage

### Automatic Running (Recommended)

Once installed, pre-commit runs automatically on staged files when you run `git commit`:

```bash
git add <files>
git commit -m "Your commit message"
# Pre-commit hooks run automatically here
```

If any hook fails:
1. The commit is **blocked**
2. Some hooks auto-fix issues (Black, isort, etc.)
3. Review the changes and re-stage: `git add <files>`
4. Commit again: `git commit -m "Your message"`

### Manual Running

Run on all files (useful for first-time setup):

```bash
pre-commit run --all-files
```

Run on specific files:

```bash
pre-commit run --files model_checkpoint/core/*.py
```

Run a specific hook:

```bash
pre-commit run black --all-files
pre-commit run flake8 --all-files
pre-commit run mypy --all-files
```

### Bypassing Hooks (Not Recommended)

In rare cases where you need to bypass hooks:

```bash
git commit --no-verify -m "Emergency fix"
```

⚠️ **Warning**: Only use `--no-verify` for emergencies. CI/CD will still run these checks.

## Configuration Files

The pre-commit system uses several configuration files:

| File | Purpose |
|------|---------|
| `.pre-commit-config.yaml` | Main pre-commit hook configuration |
| `pyproject.toml` | Black, isort, pytest, coverage, bandit config |
| `.flake8` | Flake8 linting rules |
| `mypy.ini` | MyPy type checking configuration |

## Updating Hooks

Keep hooks up-to-date with the latest versions:

```bash
# Update to latest hook versions
pre-commit autoupdate

# Review changes in .pre-commit-config.yaml
git diff .pre-commit-config.yaml

# Test the updates
pre-commit run --all-files
```

## Troubleshooting

### Issue: Hooks are too slow

**Solution**: Pre-commit caches results. First run is slower, subsequent runs are fast.

```bash
# Clear cache if needed
pre-commit clean
```

### Issue: MyPy errors on third-party imports

**Solution**: MyPy is configured with `ignore_missing_imports = True` globally. If needed, add specific ignores in `mypy.ini`:

```ini
[mypy-some_package.*]
ignore_missing_imports = True
```

### Issue: Flake8 conflicts with Black

**Solution**: Our `.flake8` config already ignores E203, W503, and E501 which conflict with Black. This is pre-configured.

### Issue: Hook installation fails

**Solution**: Ensure you have Git and Python 3.8+ installed:

```bash
git --version
python --version
pre-commit --version
```

### Issue: Want to skip a specific hook temporarily

```bash
# Set environment variable
SKIP=flake8 git commit -m "message"

# Or skip multiple hooks
SKIP=flake8,mypy git commit -m "message"
```

## Development Workflow

### Recommended Workflow

1. **Make changes** to your code
2. **Run formatters** (optional, pre-commit will do this):
   ```bash
   black model_checkpoint/
   isort model_checkpoint/
   ```
3. **Stage changes**:
   ```bash
   git add <files>
   ```
4. **Commit** (hooks run automatically):
   ```bash
   git commit -m "Description of changes"
   ```
5. **If hooks fail**, review and fix:
   - Auto-fixes are already applied (re-stage them)
   - Manual fixes required for linting errors
   - Re-commit after fixes

### First-Time Setup Workflow

When setting up pre-commit on an existing codebase:

```bash
# 1. Install pre-commit
pip install -e ".[dev]"
pre-commit install

# 2. Run on all files to fix formatting
pre-commit run black --all-files
pre-commit run isort --all-files

# 3. Stage the formatting changes
git add -A

# 4. Run all hooks to see remaining issues
pre-commit run --all-files

# 5. Fix any remaining linting/type errors manually

# 6. Commit the pre-commit setup
git add .pre-commit-config.yaml pyproject.toml .flake8 mypy.ini setup.py
git commit -m "Add pre-commit hook configuration"
```

## CI/CD Integration

Pre-commit hooks also run in CI/CD environments. The `.pre-commit-config.yaml` includes CI configuration:

```yaml
ci:
  autofix_commit_msg: '[pre-commit.ci] auto fixes from pre-commit hooks'
  autofix_prs: true
  autoupdate_schedule: weekly
```

### GitHub Actions Example

```yaml
# .github/workflows/pre-commit.yml
name: Pre-commit

on: [push, pull_request]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - uses: pre-commit/action@v3.0.0
```

## Customization

### Adding a New Hook

Edit `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/example/pre-commit-hook
    rev: v1.0.0
    hooks:
      - id: example-hook
        name: Example Hook Description
        args: [--option=value]
```

Then update:

```bash
pre-commit install
pre-commit run --all-files
```

### Excluding Files

Add to `.pre-commit-config.yaml`:

```yaml
exclude: ^(path/to/exclude/|another/path/)
```

Or per-hook:

```yaml
- id: black
  exclude: ^legacy/
```

### Adjusting Severity

Modify settings in configuration files:
- **Black**: `pyproject.toml` → `[tool.black]`
- **Flake8**: `.flake8` → `[flake8]`
- **MyPy**: `mypy.ini` → `[mypy]`
- **isort**: `pyproject.toml` → `[tool.isort]`

## Best Practices

1. ✅ **Run pre-commit regularly**: Don't disable hooks unless absolutely necessary
2. ✅ **Keep hooks updated**: Run `pre-commit autoupdate` monthly
3. ✅ **Fix issues promptly**: Don't accumulate linting debt
4. ✅ **Use auto-formatters**: Let Black and isort handle formatting
5. ✅ **Review auto-fixes**: Always review what hooks changed before committing
6. ✅ **Document exceptions**: If you add ignore rules, document why
7. ❌ **Don't use --no-verify** except for emergencies
8. ❌ **Don't disable security hooks** (bandit, safety, private key detection)

## Benefits

- **Consistency**: Everyone follows the same code style
- **Quality**: Catches bugs and code smells before they're committed
- **Security**: Detects security issues and prevents leaking secrets
- **Efficiency**: Automated checks save code review time
- **Documentation**: Enforces docstring standards
- **Type Safety**: MyPy catches type-related bugs early

## Additional Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [isort Documentation](https://pycqa.github.io/isort/)

## Support

If you encounter issues with pre-commit hooks:

1. Check this documentation
2. Review error messages carefully
3. Try `pre-commit clean` to clear caches
4. Update pre-commit: `pip install --upgrade pre-commit`
5. Check hook versions: `pre-commit autoupdate`

For project-specific questions, open an issue in the repository.
