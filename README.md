# Unknown-entity message: the listing is not gated on the requested name's type

`capture.py` runs unchanged in two trees (upstream/main and the change) and dumps
`main.json` / `branch.json`; `compose.py` builds the figure and asserts every
number it renders, that the two dumps came from different trees, that the
str-name messages are byte-identical, and that the two headless MuJoCo renders
agree to within 1/255.
