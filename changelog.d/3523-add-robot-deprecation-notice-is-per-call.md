### Fixed: the add_robot deprecation notice reaches only the call that earned it

`MuJoCoSimEngine.add_robot` carried the "resolved via deprecated
name-as-registry-key fallback" notice on the engine instead of in the call. It
was armed during model resolution and disarmed only by the success return, so
any error exit between the two -- a mesh asset the downloader cannot resolve, an
injection the recompiler refuses, an unexpected compile crash -- left it armed
for the NEXT `add_robot` to print.

The result was a false deprecation warning against a caller who had done exactly
what the message advises: `add_robot(name="arm_two", data_config="so100")`
returned success with `Warning: Hint: add_robot(name='so100') resolved via
deprecated name-as-registry-key fallback`, naming the robot from the earlier
failed call rather than the one just added.

The notice is a call-local now, so it cannot outlive its call. The deprecated
form that succeeds still warns, unchanged.
