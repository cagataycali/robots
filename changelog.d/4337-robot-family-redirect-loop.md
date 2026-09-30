### Docs: the Aerial, Bimanual and Mobile family pages no longer redirect to themselves

Three `redirect_maps` rows pointed `robots/<family>.md` at
`robots/<family>/index.md`. Under `use_directory_urls` both render to the same
`robots/<family>/index.html`, and `mkdocs-redirects` writes its stub after the
page, so the published family page was a stub whose destination was `./`.
Clicking Aerial, Bimanual or Mobile in the Robots sidebar reloaded forever.
The three rows are removed and the redirect grader now compares rendered URLs
rather than source file names, so a redirect that lands on its own URL fails
the suite.
