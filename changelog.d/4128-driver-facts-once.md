### Docs: each native driver's facts are published once, on the drivers page

The 21 robot pages with a native driver stop restating that driver's `port=`, SDK, kwargs, action keys and write checks; each links to one generated section per driver on `learn/hardware/drivers.md` (`{{driver_facts}}`, rendered from the same `DRIVERS` table in `docs/hooks/robot_pages.py`). Site total 51,184 -> 48,896 words; the site ceiling is lowered to match.
