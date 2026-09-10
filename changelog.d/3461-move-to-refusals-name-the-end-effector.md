### Fixed: move_to refusals name the end-effector frame they tried to place

`move_to` picks its end-effector automatically (a TCP-like site, else a hand
body, else the chain's leaf), and on a humanoid that last rung lands on a wrist
link. The success text named that frame; the three refusals
("unreachable", "POSE not achievable", "did not reach") did not, and the
remedy spoke of "the arm's position servos" while driving all of a g1's 29.
Every refusal now names the frame (`EE (body 'g1/left_wrist_roll_link')`),
so a caller can see which hand moved and correct.
