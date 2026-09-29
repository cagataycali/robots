### Fixed: Isaac Sim 6.1 robots load with their physics

Isaac Sim 6.1's MJCF/URDF importers put every physics schema behind a `Physics` variantSet that nothing selected, so every `add_robot` of a converted robot failed on 6.1 (`'NoneType' object has no attribute 'is_homogeneous'`). The Isaac backend now selects the PhysX variant when the converter authored one (a no-op on 6.0.x), and the USD conversion cache key includes the importer version so 6.0 and 6.1 processes sharing a cache no longer trade conversions.
