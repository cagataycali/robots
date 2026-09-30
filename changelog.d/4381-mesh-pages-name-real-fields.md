### Docs: the mesh pages name the presence fields, audit rows and lockout reply the code produces

`fleet.md` names `robot_id` and `age` (not `robot` and `last_seen`); `safety-and-estop.md` names `command_rejected_lockout` for a command under lockout, quotes the real `{"type": "error", "error": "command rejected"}` reply and lists `ping` among the verbs a locked peer still answers. A grader reads each name from the code. (#4174)
