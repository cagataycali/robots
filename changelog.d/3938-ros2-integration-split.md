### Docs: the ROS 2 integration guide is six pages, each under the word budget

`docs/ros2-integration.md` carried ten level-2 sections in one 5,241-word
scroll. It is now a hub (surfaces table, requirements, `use_ros` actions,
examples, the live demo) plus `ros2/safety.md`, `ros2/ackermann.md`,
`ros2/sim-bridge.md`, `ros2/hardware-bridge.md` and `ros2/mesh-bridge.md`. The
text moved verbatim, every page is under the 1,500-word per-page budget, the
`safety-critical-command-surfaces-need-operator-approval` anchor other pages
link keeps its slug on the safety page, and the page's exemption in
`tests/test_docs_pages_are_within_the_word_budget.py` is dropped so it cannot
grow back unnoticed.
