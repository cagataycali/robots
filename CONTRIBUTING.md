# Contributing Guidelines

Thank you for your interest in contributing to Strands Robots! Whether it's a bug report, new feature, or documentation improvement, we value your help.

## Reporting Bugs/Feature Requests

Use [Bug Reports](../../issues/new?template=bug_report.yml) or [Feature Requests](../../issues/new?template=feature_request.yml) to file issues.

Please include:
* Steps to reproduce
* Robot hardware / simulation backend used
* Python version and OS
* Any relevant logs or error messages

## Finding Contributions

Check [Ready for Contribution](../../issues?q=is%3Aissue%20state%3Aopen%20label%3A%22ready%20for%20contribution%22) issues. Comment on an issue before starting significant work.

## Development Setup

```bash
git clone https://github.com/strands-labs/robots
cd robots
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## Making Changes

1. Create a branch from `main`
2. Make your change — keep it focused
3. Run tests: `python -m pytest`
4. Format: `ruff format .`
5. Lint: `ruff check .`
6. Commit with a clear message following [Conventional Commits](https://www.conventionalcommits.org)
7. Open a pull request

## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).

## Security Issue Notifications

If you discover a potential security issue, notify AWS/Amazon Security via the [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

## Licensing

See the [LICENSE](./LICENSE) file. We will ask you to confirm the licensing of your contribution.
