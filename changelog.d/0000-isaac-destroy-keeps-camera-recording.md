### Fixed: destroy() no longer discards a held Isaac camera recording in silence

`IsaacSimulation.stop_cameras_recording` leaves a raw-camera recording
registered when the clip encoder is absent - the frames are host-side NumPy
arrays still in `_cams_rec_state["buffers"]`, and the refusal promises they are
recoverable by installing the encoder and calling the verb again. `destroy()`
then set `_cams_rec_state = None` outright, discarding exactly those frames
under `status="success"` - the silent loss the refusal had ruled out:

```python
sim.start_cameras_recording(cameras=["front"], fps=20, name="wedge")
# ... frames captured into the host-side buffers ...
sim.destroy()
# before: buffers dropped to None; no clip, no warning
# after:  encoder present -> clip encoded before teardown;
#         encoder absent  -> WARNING naming "wedge" and per-camera frame counts
```

The buffers are host arrays that survive the stage teardown and encode fine
(unlike the old comment claimed). `destroy()` now honours the same rule the
flush does: if the encoder is installed it encodes every camera's buffered
frames to their registered path before the stage is torn down; if it is absent
it names the recording and its per-camera frame counts in a WARNING - never a
silent `None`. This mirrors the MuJoCo recorder, whose recording already
survives `destroy()`.
