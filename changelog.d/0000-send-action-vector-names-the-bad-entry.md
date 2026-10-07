### Fixed: `send_action` names the vector entry that is not a number

An ordered action vector with a non-numeric entry (a string, a nested list,
bytes, `None`) is now refused as
`action vector entry 3 ('wrist_flex') must be a scalar number (one value per
actuator/joint), got str.` - the index, the action key and the type, in the same
words the mapping branch uses - instead of the raw `float()` exception text. A
vector of the wrong length is now reported as a length mismatch even when one of
its entries is also non-numeric.
