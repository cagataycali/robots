### Fixed: the README quickstart puts the red cube in the scene it asks the agent to pick up

`Robot("so100")` opens an empty table, so the headline `Agent(tools=[robot])("pick up the red cube")` asked for an object that was not there. The snippet now adds a 5 cm red cube 20 cm in front of the arm and a `front` camera whose view of it is not blocked by the arm. `tests/test_readme_quickstart_scene_holds_what_the_prompt_names.py` runs the README block with a recording `Agent` and checks the prompt's object exists and the camera's line of sight reaches it.
