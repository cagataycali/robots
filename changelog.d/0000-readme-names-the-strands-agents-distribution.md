### Docs: the README says `from strands import Agent` comes from `strands-agents`

The Install section now names the distribution behind the quickstart's
`from strands import Agent` and warns that the PyPI package called `strands` is
an unrelated physics solver, so a reader who hits `No module named 'strands'`
does not reach for `pip install strands`.
