### Docs: every Hub id the docs name is a repository that exists

`remote.md` named `lerobot/act_so101`, a checkpoint that does not exist on the Hub; it now names `robotfuel/act_so101_t16b`. A grader keeps every quoted checkpoint id on the site in a verified table (sha and date from `HfApi().model_info`), re-checked live with `STRANDS_DOCS_HUB_LIVE=1`. (#4158)
