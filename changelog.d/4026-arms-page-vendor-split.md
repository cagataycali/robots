### Docs: the arms page keeps the catalog, each vendor's bring-up gets its own page

`docs/robots/arms.md` had grown to 2,381 words carrying three subjects: the
generated catalog with its compatibility table, a UR e-Series bring-up over
RTDE, and the Feetech SO arm bring-up - reading an arm without moving it,
the travel `lerobot-calibrate` records, a threaded `run_policy`, and the
`transport="twin"` verbs.

Split at the vendor boundary, the way `hardware/reachy-mini.md` already holds
one robot's bring-up. `robots/arms.md` (549 words) is the family page: the
catalog, which arms have which real path, and the Franka bullet. The UR
driver's two write gates, its vector-width refusal and `stop_task()` are now
`docs/hardware/universal-robots.md`; the SO arm bring-up is
`docs/hardware/so-arms.md`. Every page is inside the 1,500-word budget, the
`twin-transport.md` links follow the section they name, and both new pages
have a nav row.
