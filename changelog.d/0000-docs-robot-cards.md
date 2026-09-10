### Changed: robot family pages are generated cards, not hand tables

`docs/hooks/robot_cards.py` turns `{{robot_cards:<category>}}` into one card
per registered robot (sim render when `docs/assets/sim_render_<name>.png`
exists, captioned "sim render"; `Robot("<name>")`; joints; sim / real badges;
aliases). The five family pages lose ~330 lines of tables that drifted from
`robots.json`; four table-grading tests are replaced by one hook grader.
