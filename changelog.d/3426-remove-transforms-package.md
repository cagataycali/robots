### Removed:

`strands_robots.transforms` (augmentation transforms, `import_transform_class`, the `MockTransform` provider), its `docs/data/transforms.md` page and the `07_data_augmentation` notebook. Nothing else in the package imported it; LeRobot's own `image_transforms` on `LeRobotDataset` is the supported way to augment a dataset.
