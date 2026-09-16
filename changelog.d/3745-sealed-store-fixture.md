### Fixed: the pre-sign-in cell grades a dashboard that really has passkeys

`_seal_with_two_passkeys` wrote a store with no `jwt_secret`. Since a store that
cannot serve as one is moved aside for a working default, that file was rescued
rather than read: the cell asserting the enrolled passkeys are not published ran
against a dashboard with none enrolled, so its two withholding assertions held
vacuously while `setup_required` reported `true`. The fixture now writes the
secret every real store carries, and the cell asserts there is something to
withhold before checking that it is withheld.
