### Fixed

- **mesh**: every resource the AWS IoT bootstrap ensures is now recorded in
  `BootstrappedAccount.created` / `.skipped`. Those two lists are the whole record an
  operator has of what a provisioning run touched, and the closing log line counts them, but
  three of the thirteen helpers that mutate the account ensured a resource without recording
  it. `_grant_iot_invoke_lambda` recorded neither the `lambda:InvokeFunction` grant it creates
  for the E-stop fan-out rule nor the one it finds already present, while the other
  `add_permission` helper forty lines away recorded both - so the one resource absent from the
  audit trail was a permission grant. `_ensure_iot_action_role` and `_ensure_provisioning_role`
  recorded the role they create but not the existing role they reuse, unlike the two sibling
  role helpers. Measured with the real helpers driven against fake clients: a fresh account
  creates thirteen resources and reported twelve, a fully provisioned one reused eleven and
  reported ten, and a resumed run - roles and Lambdas left by an earlier attempt, rules and
  template lost - reused ten and reported seven. Both permission statements now record under a
  single-sourced ledger name, because a resource-policy statement has no ARN of its own to
  return; the values are unchanged, so existing assertions on the provisioning-hook entry hold.
  No resource, ARN, IAM policy or API call changes - only what the run reports about itself.
  A new structural guard derives the helper set from the module, so a helper added later is
  held to the same contract on arrival.
