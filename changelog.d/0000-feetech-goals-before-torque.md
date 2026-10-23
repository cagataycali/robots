### Fixed: a Feetech arm computes every goal before torque comes on

`Robot.act` on a Feetech body now computes (and validates) every joint goal before enabling torque, so a
non-finite target is refused with the arm still limp. Before, torque was enabled first and only then were the
goals computed, so a NaN in the command stiffened the arm against a stale `Goal_Position` before raising.
