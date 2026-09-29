### Fixed: MJCF robots on Isaac keep their position-servo gains

Isaac's MJCF importer dropped every `dampratio` position servo (all of MuJoCo Menagerie), so converted robots had zero-stiffness drives and hung limp. The backend now authors each servo's compiled gains as PhysX drives in PhysX's per-degree units; on so100 the tracking error matches MuJoCo's.
