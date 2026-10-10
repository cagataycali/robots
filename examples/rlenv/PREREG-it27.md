## it27 PRE-REGISTRATION (written 2026-10-10 06:5xZ, BEFORE koch's pair is recorded or scored)

it26's rule ("goal-decoupling is bought with expert competence") is only a two-point reading, and two
points fit any monotone story. koch is the third point and its sigma=0 cell already exists, so the
prediction can be written down before the number exists:

  koch push sigma=0 goal-aware 0.5750, goal-blind 0.0000, headroom 0.5750 (n=40, band 1.0)
  koch push sigma=0.05 pass-A failure rate running ~0.66 at the time of writing => expert ~0.34

Relative competence lost to the noise, per arm:
  so100  0.4333 -> 0.4667   = none (expert unmoved)            pair goal AUC 0.6929
  koch   0.5750 -> ~0.34    = ~0.41 of its success rate        pair goal AUC = ?
  so101  0.8000 -> 0.2833   = ~0.65 of its success rate        pair goal AUC 0.4990

PREDICTION (primary): koch's within-condition goal-only AUC lands STRICTLY BETWEEN so101's 0.4990 and
so100's 0.6929, i.e. in [0.52, 0.67], and its state AUC stays inside its permuted null.
FALSIFIED IF: koch's goal AUC is >= 0.6929 or <= 0.4990 (i.e. the ordering by competence lost does not
hold), in which case the it26 rule is withdrawn as a two-point coincidence and the lane reports the
honest alternative -- that the amount of decoupling is arm-specific and not predictable from the
competence cost at all.
PUBLISHING GATE is unchanged and independent of the prediction: goal AUC < 0.75 AND state AUC inside its
permuted null AND all data gates (goals bit-exact with the seeded noise, load-back, sidecar, single
class, zero seed overlap). A falsified prediction does NOT block publishing; it changes the card text.
