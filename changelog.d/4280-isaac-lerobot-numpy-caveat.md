### Docs: the lerobot and Isaac Sim numpy conflict is named, with the combination that works

`lerobot` 0.6 needs `numpy<2.3` and Isaac Sim pins `numpy==2.3.1`. The install hints, docs and notebook 05 now say to install `lerobot` after Isaac Sim and let pip downgrade numpy to 2.2.6, which is verified to work with Isaac Sim 6.1.
