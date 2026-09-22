### Docs: the world-building guide is six pages, each under the word budget

`docs/simulation/world-building.md` carried 17 level-2 sections in one
5,893-word scroll. It is now a hub (setup entry points, strategies, cameras,
multi-robot) plus `simulation/spawning.md`, `simulation/terrain.md`,
`simulation/objects.md`, `simulation/meshes-and-materials.md` and
`simulation/scene-editing.md`. The text moved verbatim, every page is under the
1,500-word per-page budget, and the page's exemption in
`tests/test_docs_pages_are_within_the_word_budget.py` is dropped so it cannot
grow back unnoticed.
