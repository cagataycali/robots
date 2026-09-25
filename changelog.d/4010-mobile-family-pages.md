### Changed: the mobile catalog is three pages, one per registry family

`docs/robots/mobile.md` documented three registry categories - `mobile` (10
robots), `mobile_manip` (6) and `aerial` (2) - and four hardware bring-ups in
3,518 words, while `docs/robots/index.md` already offered three cards that all
pointed at it. Each family now has its own page: `mobile.md` keeps the Go2 and
EarthRover native drivers (1,450 words), the new `mobile-manip.md` holds the
Yahboom ROSMASTER M3 Pro in sim, over its ROS 2 graph and on the twin (1,394),
and the new `aerial.md` holds the Crazyflie CRTP flight path (787). The index
cards, the counts table and the nav land on the family they name, and
`test_docs_pages_are_within_the_word_budget.py` no longer exempts
`robots/mobile.md` from the 1,500-word budget.
