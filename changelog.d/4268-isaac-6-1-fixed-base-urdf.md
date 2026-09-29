### Fixed: fixed-base URDF robots stay welded on Isaac Sim 6.1

Isaac Sim 6.1's URDF importer roots the articulation on the base link's rigid body, which PhysX treats as a floating base, so a fixed-base arm lifted 2 cm and drifted. The backend now moves the articulation root onto the fixed joint that welds the robot to the world; 6.0.x is unaffected.
