# Contributing

See [CONTRIBUTING.md](https://github.com/strands-labs/robots/blob/main/CONTRIBUTING.md) for full guidelines.

## Quick Version

```bash
git clone https://github.com/strands-labs/robots
cd robots
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

1. Create a branch from `main`
2. Make your change
3. `python -m pytest`
4. `ruff format . && ruff check .`
5. Open a pull request

## Report Issues

- [Bug Reports](https://github.com/strands-labs/robots/issues/new?template=bug_report.yml)
- [Feature Requests](https://github.com/strands-labs/robots/issues/new?template=feature_request.yml)
