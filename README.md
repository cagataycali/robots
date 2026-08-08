# Artifact: IotMqttTransport connect_timeout domain

`iot_connect_timeout_domain.png` is composed from two measurement runs of the
same script, one in a checkout of `main` and one on the branch, each recording
which tree it imported. `compose_figure.py` re-derives every cell from those two
JSON dumps and asserts each claim before saving.

* `capture_connecting_broker.py` - broker reports CONNACK 50 ms after `start()`.
* `capture_silent_broker.py` - broker never reports CONNACK.
