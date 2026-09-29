### Fixed: an Isaac add_robot load failure keeps its traceback and exception type

A robot that failed to load on the Isaac backend was logged as a one-line message and reported with only the exception's text, so a failure deep in Kit could not be located. The failure is now logged with its traceback, and the error envelope names the exception type.
