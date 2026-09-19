### Tests: one table owns the shared-domain reader scan

Each `Trainer._*_problems` gate documents a biconditional -- a backend that reads
the spec field must route it through the gate, and one that gates the field must
read it -- and 27 per-field `tests/training/test_*_domain.py` files each carried a
private copy of the scan that grades it. The scan is now one table keyed on
(gate, fields, readers) in `tests/training/test_every_shared_domain_has_one_owner.py`;
the per-field files keep the domain of their field and the premise behind it.

The consolidated rule is stronger than the copies it replaces: it grades every
operand of a comparison on the syntax tree, so the chained form the substring
scans could not see is reported; the comparisons and conversions a backend
legitimately makes are declared per row; the reverse half of the biconditional,
pinned for 5 of 27 fields, now holds for all 24 rows; and a gate on `Trainer`
that no row and no other-shaped guard names is reported.
