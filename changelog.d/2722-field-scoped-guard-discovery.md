### Fixed

- **training**: the guard that holds every field-scoped shared-domain guard to
  the "a reader scan must see both forms of a read" rule now discovers all
  thirteen of them rather than five. It keyed discovery on the *name* of the
  helper a guard uses to list the backend modules, and the guards spell that
  helper two ways (`_trainer_modules` and `_training_modules`), so the eight
  guards using the other spelling sat outside the sweep - silently, because it
  reported a clean tree over the five it could see and pinned that count as its
  own non-vacuity assertion. All eight were the by-name-only scans the shared
  rule exists to replace, so each certified a clean `readers == {...}` sweep
  while a backend that forwards its gated field through a table and skips the
  gate went unreported. Discovery is now keyed on the two properties that make a
  guard gradeable (one reader helper, a scope rooted at the backend tree), the
  eight reader scans route through the shared rule, and the forwarding-provider
  assertions derive their own scope from what that provider actually forwards so
  they no longer assume every gated field is one of them.
