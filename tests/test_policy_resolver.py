"""Tests for strands_robots.policy_resolver — resolve_policy()."""

from strands_robots.policy_resolver import resolve_policy


class TestResolveShorthand:
    def test_mock(self):
        provider, kwargs = resolve_policy("mock")
        assert provider == "mock"

    def test_random(self):
        provider, kwargs = resolve_policy("random")
        assert provider == "mock"

    def test_groot(self):
        provider, kwargs = resolve_policy("groot")
        assert provider == "groot"

    def test_dreamgen(self):
        provider, kwargs = resolve_policy("dreamgen")
        assert provider == "dreamgen"

    def test_cosmos(self):
        provider, kwargs = resolve_policy("cosmos")
        assert provider == "cosmos_predict"


class TestResolveHuggingFace:
    def test_lerobot_model(self):
        provider, kwargs = resolve_policy("lerobot/act_aloha_sim_transfer_cube_human")
        assert provider == "lerobot_local"
        assert kwargs["pretrained_name_or_path"] == "lerobot/act_aloha_sim_transfer_cube_human"

    def test_openvla_model_falls_back_to_lerobot(self):
        """openvla org was removed — unknown orgs now default to lerobot_local."""
        provider, kwargs = resolve_policy("openvla/openvla-7b")
        assert provider == "lerobot_local"
        assert kwargs["pretrained_name_or_path"] == "openvla/openvla-7b"

    def test_nvidia_groot(self):
        provider, kwargs = resolve_policy("nvidia/gr00t-something")
        assert provider == "groot"

    def test_unknown_org_defaults_lerobot(self):
        provider, kwargs = resolve_policy("someorg/somemodel")
        assert provider == "lerobot_local"


class TestResolveServerAddresses:
    def test_host_port(self):
        provider, kwargs = resolve_policy("localhost:8080")
        assert provider == "lerobot_async"
        assert kwargs["server_address"] == "localhost:8080"

    def test_websocket(self):
        provider, kwargs = resolve_policy("ws://gpu-server:9000")
        assert provider == "dreamzero"
        assert kwargs["host"] == "gpu-server"
        assert kwargs["port"] == 9000

    def test_grpc(self):
        provider, kwargs = resolve_policy("grpc://my-server:50051")
        assert provider == "lerobot_async"
        assert kwargs["server_address"] == "my-server:50051"

    def test_zmq(self):
        provider, kwargs = resolve_policy("zmq://host:5555")
        assert provider == "groot"
        assert kwargs["host"] == "host"
        assert kwargs["port"] == 5555


class TestResolveExtraKwargs:
    def test_extra_kwargs_passed_through(self):
        provider, kwargs = resolve_policy("mock", foo="bar")
        assert kwargs.get("foo") == "bar"

    def test_hf_extra_kwargs(self):
        provider, kwargs = resolve_policy("lerobot/act_test", device="cuda")
        assert kwargs["device"] == "cuda"
        assert kwargs["pretrained_name_or_path"] == "lerobot/act_test"
