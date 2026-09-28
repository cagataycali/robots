### Fixed: the dashboard's sim e-stop no longer reports that it locked the fleet

`POST /api/safety/estop` freezes this dashboard's simulation sessions only, but
its lockout read "an e-stop from loopback locked the fleet" while mesh peers
kept accepting tasks. The reason now names what it locked, and
`GET /api/safety` answers two named scopes: `lockout` (with `scope`) for the
simulation and `fleet` for the mesh verdict. The fleet stop remains
`POST /api/mesh/safety/estop`.
