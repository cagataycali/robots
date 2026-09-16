### Fixed: an unreadable connection no longer deletes the device facts that *were* read

`status` on a real `Robot` reads `is_connected` from the driver, and every
shipped lerobot arm folds its cameras into it (`self.bus.is_connected and
all(cam.is_connected ...)`, 8 drivers in lerobot 0.6.2). So a camera unplugged
mid-run - the failure `_device_facts` already tolerates *per camera* - raised
through the aggregate, where it was read unguarded and took the whole probe
with it: the tool printed the bare task state again (no port, no cameras, the
defect the device lines exist to fix) and `get_status()` answered
`is_connected: False` with `task_status: "error"` for an arm that was connected
and driving a task.

`is_connected` is now three-state like every other fact in that probe - `None`
means it was not read - so the port and the healthy cameras still print, the
camera that could not answer is named as such rather than as "not connected",
and the per-camera attribution survives the failure it exists to attribute.
