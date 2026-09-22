"""The RunPod helper spends money, so its decisions are tested without the network."""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "runpod_gpu.py"
spec = importlib.util.spec_from_file_location("runpod_gpu", SCRIPT)
runpod_gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runpod_gpu)

SECRET = "rpa_TESTKEYTHATMUSTNEVERBEPRINTED"
RUNNING = {
    "id": "abc123",
    "desiredStatus": "RUNNING",
    "publicIp": "203.0.113.7",
    "portMappings": {"22": 10341},
    "costPerHr": 0.34,
    "machine": {"gpuTypeId": "NVIDIA GeForce RTX 4090"},
}


class FakeApi:
    """Records every call and answers from a script, so no test touches the network."""

    def __init__(self, answers: dict[tuple[str, str], object], balance: float = 25.0) -> None:
        self.answers = answers
        self.balance = balance
        self.calls: list[tuple[str, str, object]] = []

    def __call__(self, method: str, url: str, headers: dict[str, str], body: object):
        assert headers["Authorization"] == f"Bearer {SECRET}"
        # The API sits behind Cloudflare, which refuses the default urllib agent.
        assert headers["User-Agent"].startswith("exu-runpod/")
        self.calls.append((method, url, body))
        if url == runpod_gpu.GRAPHQL:
            return 200, {
                "data": {"myself": {"clientBalance": self.balance, "currentSpendPerHr": 0}}
            }
        path = url.removeprefix(runpod_gpu.API)
        answer = self.answers[(method, path)]
        if isinstance(answer, Exception):
            raise answer
        return answer


def client(tmp_path, answers, balance: float = 25.0):
    api = FakeApi(answers, balance)
    return runpod_gpu.RunPod(SECRET, transport=api, state=tmp_path / "pod.json"), api


def test_the_key_comes_from_the_environment_before_the_file(tmp_path, monkeypatch) -> None:
    key_file = tmp_path / "runpod.key"
    key_file.write_text("rpa_from_file\n", encoding="utf-8")
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_from_env")
    assert runpod_gpu.load_key(key_file) == "rpa_from_env"
    monkeypatch.delenv("RUNPOD_API_KEY")
    assert runpod_gpu.load_key(key_file) == "rpa_from_file"


def test_a_missing_key_is_a_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="RUNPOD_API_KEY"):
        runpod_gpu.load_key(tmp_path / "nope.key")


def test_the_create_request_asks_for_what_the_lockfile_needs() -> None:
    payload = runpod_gpu.create_payload("exu", ["NVIDIA GeForce RTX 4090"], "COMMUNITY", 30)

    assert payload["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
    assert payload["gpuTypePriority"] == "custom", "the list is an order of preference"
    assert payload["gpuCount"] == 1
    assert payload["cloudType"] == "COMMUNITY"
    assert "22/tcp" in payload["ports"]
    assert payload["supportPublicIp"] is True
    # torch in uv.lock is a CUDA 13 build: an older driver would not see the GPU.
    assert payload["allowedCudaVersions"] == ["13.0"]
    assert payload["interruptible"] is False
    # The image is pulled on billed time, so a slow host is an expensive host.
    assert payload["minDownloadMbps"] == runpod_gpu.MIN_DOWNLOAD_MBPS >= 300


def test_up_refuses_to_rent_without_credit(tmp_path) -> None:
    pods, api = client(tmp_path, {}, balance=0.0)

    with pytest.raises(SystemExit, match="balance"):
        pods.up("exu", ["NVIDIA GeForce RTX 4090"], "COMMUNITY", 30)

    assert ("POST", f"{runpod_gpu.API}/pods") not in [(m, url) for m, url, _body in api.calls]


def test_up_refuses_a_second_pod_while_one_is_recorded(tmp_path) -> None:
    pods, api = client(tmp_path, {("GET", "/pods/abc123"): (200, RUNNING)})
    pods.state.write_text(json.dumps({"id": "abc123"}), encoding="utf-8")

    with pytest.raises(SystemExit, match="abc123"):
        pods.up("exu", ["NVIDIA GeForce RTX 4090"], "COMMUNITY", 30)

    assert ("POST", f"{runpod_gpu.API}/pods") not in [(m, url) for m, url, _body in api.calls]


def test_up_records_the_pod_so_it_can_be_found_and_terminated(tmp_path) -> None:
    pods, _api = client(tmp_path, {("POST", "/pods"): (201, {"id": "abc123", "costPerHr": 0.34})})

    pod = pods.up("exu", ["NVIDIA GeForce RTX 4090"], "COMMUNITY", 30)

    assert pod["id"] == "abc123"
    assert json.loads(pods.state.read_text(encoding="utf-8"))["id"] == "abc123"


def test_up_can_rent_a_different_image(tmp_path) -> None:
    pods, api = client(tmp_path, {("POST", "/pods"): (201, {"id": "abc123"})})

    pods.up("exu", ["NVIDIA GeForce RTX 4090"], "COMMUNITY", 30, "runpod/base:small")

    sent = [body for method, url, body in api.calls if (method, url[-5:]) == ("POST", "/pods")]
    assert sent[0]["imageName"] == "runpod/base:small"


def test_waiting_for_ssh_survives_a_slow_image_pull(tmp_path) -> None:
    pulling = {**RUNNING, "publicIp": "", "portMappings": None}
    answers = iter([pulling, pulling, RUNNING])
    pods, _api = client(tmp_path, {})
    pods.state.write_text(json.dumps({"id": "abc123"}), encoding="utf-8")
    pods.pod = lambda: next(answers)
    naps: list[float] = []

    found = pods.wait_for_ssh(5, pause=naps.append, reachable=lambda _endpoint: True)

    assert found == ("203.0.113.7", 10341)
    assert len(naps) == 2


def test_waiting_does_not_trust_a_port_nothing_answers_on(tmp_path) -> None:
    # Found on a real pod: right after `start` the API still named the port of the
    # previous run, `wait` returned it, and ssh got "Connection refused".
    stale = {**RUNNING, "portMappings": {"22": 41296}}
    fresh = {**RUNNING, "portMappings": {"22": 41262}}
    answers = iter([stale, stale, fresh])
    pods, _api = client(tmp_path, {})
    pods.pod = lambda: next(answers)

    found = pods.wait_for_ssh(
        5, pause=lambda _seconds: None, reachable=lambda endpoint: endpoint[1] == 41262
    )

    assert found == ("203.0.113.7", 41262)


def test_a_closed_port_is_reported_as_closed() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert runpod_gpu.port_open(("127.0.0.1", port))
    assert not runpod_gpu.port_open(("127.0.0.1", port))


def test_waiting_gives_up_and_says_the_pod_still_bills(tmp_path) -> None:
    pods, _api = client(tmp_path, {})
    pods.pod = lambda: {**RUNNING, "publicIp": "", "portMappings": None}

    with pytest.raises(SystemExit, match="still billing"):
        pods.wait_for_ssh(0, pause=lambda _seconds: None)


def test_waiting_stops_at_once_when_the_pod_dies(tmp_path) -> None:
    # A pod the host evicted never gets an endpoint: do not sit out the whole timeout.
    pods, _api = client(tmp_path, {})
    pods.pod = lambda: {"id": "abc123", "desiredStatus": "EXITED", "publicIp": ""}

    with pytest.raises(SystemExit, match="EXITED"):
        pods.wait_for_ssh(60, pause=lambda _seconds: pytest.fail("should not wait"))


def test_a_data_folder_is_sent_file_by_file(tmp_path) -> None:
    # rsync --files-from silently skips the contents of a folder it is given.
    (tmp_path / "tools" / "probes").mkdir(parents=True)
    (tmp_path / "tools" / "experiment.sh").write_text("", encoding="utf-8")
    (tmp_path / "tools" / "probes" / "probe.py").write_text("", encoding="utf-8")
    single = tmp_path / "data.jsonl"
    single.write_text("", encoding="utf-8")

    found = runpod_gpu.expand([str(tmp_path / "tools"), str(single)])

    assert [Path(path).name for path in found] == ["experiment.sh", "probe.py", "data.jsonl"]


def test_a_data_path_that_does_not_exist_is_refused(tmp_path) -> None:
    with pytest.raises(SystemExit, match="does not exist"):
        runpod_gpu.expand([str(tmp_path / "typo.jsonl")])


def test_down_terminates_and_forgets_the_pod(tmp_path) -> None:
    pods, api = client(tmp_path, {("DELETE", "/pods/abc123"): (204, None)})
    pods.state.write_text(json.dumps({"id": "abc123"}), encoding="utf-8")

    pods.down()

    assert ("DELETE", f"{runpod_gpu.API}/pods/abc123", None) in api.calls
    assert not pods.state.exists()


def test_down_keeps_the_record_when_the_api_refuses(tmp_path) -> None:
    # Forgetting a pod that is still billing is the expensive mistake.
    pods, _api = client(tmp_path, {("DELETE", "/pods/abc123"): (500, {"error": "boom"})})
    pods.state.write_text(json.dumps({"id": "abc123"}), encoding="utf-8")

    with pytest.raises(SystemExit, match="500"):
        pods.down()

    assert pods.state.exists()


def test_an_api_error_never_prints_the_key(tmp_path) -> None:
    pods, _api = client(tmp_path, {("GET", "/pods"): (401, {"error": f"bad key {SECRET}"})})

    with pytest.raises(SystemExit) as failure:
        pods.list()

    assert SECRET not in str(failure.value)


def test_the_ssh_endpoint_needs_both_the_address_and_the_port() -> None:
    assert runpod_gpu.ssh_endpoint(RUNNING) == ("203.0.113.7", 10341)
    assert runpod_gpu.ssh_endpoint({**RUNNING, "publicIp": ""}) is None
    assert runpod_gpu.ssh_endpoint({**RUNNING, "portMappings": {}}) is None
    assert runpod_gpu.ssh_endpoint({**RUNNING, "portMappings": None}) is None


def test_sync_sends_tracked_files_and_named_data_but_never_secrets_or_checkpoints() -> None:
    command = runpod_gpu.rsync_push(("203.0.113.7", 10341), Path("~/.ssh/id_rsa"), "files.txt")

    joined = " ".join(command)
    assert "--files-from=files.txt" in joined
    assert "-p 10341" in joined
    assert command[-1] == f"root@203.0.113.7:{runpod_gpu.REMOTE_DIR}/"


def test_the_pod_gets_rsync_before_the_first_transfer() -> None:
    # Found on a real pod: `bash: rsync: command not found`, and the sync died.
    prepare = runpod_gpu.prepare_remote()

    assert f"mkdir -p {runpod_gpu.REMOTE_DIR}" in prepare
    assert "command -v rsync" in prepare
    assert prepare.index("command -v rsync") < prepare.index("apt-get install")


def test_setup_refuses_to_start_a_second_install() -> None:
    # Found on a real pod: three interrupted setups left three `uv sync` running at
    # once, the environment took ten billed minutes and was still not ready.
    setup = runpod_gpu.setup_remote()

    assert setup.index("pgrep") < setup.index("uv sync --frozen")
    assert "exit 3" in setup
    # Six hosts in a row once passed nvidia-smi and failed CUDA init; the job then
    # crawled on the CPU at full price. Setup has to fail there, not print False.
    assert "assert torch.cuda.is_available()" in setup


def test_setup_is_told_apart_from_its_own_guard(tmp_path) -> None:
    # The guard greps for `uv sync --frozen`; it must not match the guard's own shell.
    done = subprocess.run(["bash", "-c", "pgrep -x -f 'uv sync --frozen'; echo rc=$?"],
                          capture_output=True, text=True)  # fmt: skip
    assert "rc=1" in done.stdout, "nothing is installing here, so the guard must pass"


def test_the_file_list_leaves_out_weights_and_keys(tmp_path) -> None:
    listed = [
        "src/exu/model.py",
        "artifacts/old/model.safetensors",
        ".env",
        "id_rsa",
        "a/runpod.key",
    ]

    kept = runpod_gpu.safe_to_send(listed)

    assert kept == ["src/exu/model.py"]


def test_the_remote_job_stops_the_pod_when_it_ends_and_has_a_time_limit() -> None:
    script = runpod_gpu.remote_job("bash run.sh", max_hours=6, stop_after=True)

    assert "timeout 6h" in script
    assert "bash run.sh" in script
    assert "runpodctl stop pod" in script
    assert "nohup" in script


def test_the_pod_outlives_the_job_long_enough_to_fetch_the_results() -> None:
    # A stopped pod has no ssh. Stopping the instant the job ends would strand the results.
    script = runpod_gpu.remote_job("bash run.sh", max_hours=6, stop_after=True, grace_minutes=20)

    # A marker left by an earlier job is cleared first, or `logs` would report it.
    assert script.index("rm -f run.done") < script.index("bash run.sh")
    assert script.index("bash run.sh") < script.index("> run.done") < script.index("sleep 1200")
    assert script.index("sleep 1200") < script.index("runpodctl stop pod")


def test_offers_keep_only_what_is_in_stock_cheapest_first() -> None:
    answer = {
        "data": {
            "gpuTypes": [
                {"id": "NVIDIA L40S", "memoryInGb": 48, "lowestPrice": {
                    "stockStatus": "Low", "uninterruptablePrice": 0.79, "minVcpu": 24,
                    "minMemory": 251}},
                {"id": "NVIDIA RTX A5000", "memoryInGb": 24, "lowestPrice": {
                    "stockStatus": None, "uninterruptablePrice": None, "minVcpu": None,
                    "minMemory": None}},
                {"id": "NVIDIA GeForce RTX 5090", "memoryInGb": 32, "lowestPrice": {
                    "stockStatus": "Low", "uninterruptablePrice": 0.69, "minVcpu": 12,
                    "minMemory": 46}},
                {"id": "NVIDIA A40", "memoryInGb": 48, "lowestPrice": None},
            ]
        }
    }  # fmt: skip

    rows = runpod_gpu.offer_rows(answer)

    assert [row[1] for row in rows] == ["NVIDIA GeForce RTX 5090", "NVIDIA L40S"]
    assert rows[0] == (0.69, "NVIDIA GeForce RTX 5090", 32, "Low", 12, 46)


def test_the_offers_query_carries_the_speed_floor_and_the_cloud() -> None:
    query = runpod_gpu.offers_query(400, secure=True)["query"]

    assert "minDownload:400" in query
    assert "secureCloud:true" in query


def test_start_wakes_the_recorded_pod(tmp_path) -> None:
    pods, api = client(tmp_path, {
        ("GET", "/pods/abc123"): (200, {**RUNNING, "desiredStatus": "EXITED"}),
        ("POST", "/pods/abc123/start"): (200, {"id": "abc123"}),
    })  # fmt: skip
    pods.state.write_text(json.dumps({"id": "abc123"}), encoding="utf-8")

    pods.start()

    assert ("POST", f"{runpod_gpu.API}/pods/abc123/start", None) in api.calls


def test_starting_a_job_returns_at_once(tmp_path) -> None:
    # Found on a real pod: the launch hung for as long as the job ran. Run the same
    # shell locally, with a job that outlives the launch.

    script = runpod_gpu.remote_job("sleep 5", max_hours=1, stop_after=False)
    local = script.replace(f"cd {runpod_gpu.REMOTE_DIR}", f"cd {tmp_path}")
    started = time.monotonic()

    done = subprocess.run(["bash", "-c", local], capture_output=True, timeout=4)

    assert done.returncode == 0
    assert time.monotonic() - started < 3
    assert not (tmp_path / "run.done").exists(), "the job is still running"


def _launch(tmp_path, command: str, **options) -> subprocess.CompletedProcess[bytes]:
    script = runpod_gpu.remote_job(command, max_hours=1, **options)
    local = script.replace(f"cd {runpod_gpu.REMOTE_DIR}", f"cd {tmp_path}")
    return subprocess.run(["bash", "-c", local], capture_output=True, timeout=10)


def _wait_for(path: Path, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.05)


def test_a_second_job_cancels_the_stop_the_first_one_left_pending(tmp_path) -> None:
    # The first job is over and sits in its grace period. Left alone, its stop would
    # fire in the middle of the second job. Real shell, with a stop that leaves a mark.
    first = _launch(tmp_path, "true", stop_after=True, grace_minutes=0.02,
                    stop_command="touch stopped")  # fmt: skip
    assert first.returncode == 0
    _wait_for(tmp_path / "run.done")

    second = _launch(tmp_path, "sleep 2", stop_after=False)

    assert second.returncode == 0
    time.sleep(1.6)  # past the 1.2 s grace period of the first job
    assert not (tmp_path / "stopped").exists(), "the first job's stop fired during the second"
    assert (tmp_path / "run.previous.log").exists()


def test_the_pending_stop_still_fires_when_nothing_replaces_the_job(tmp_path) -> None:
    done = _launch(tmp_path, "true", stop_after=True, grace_minutes=0.01,
                   stop_command="touch stopped")  # fmt: skip

    assert done.returncode == 0
    _wait_for(tmp_path / "stopped")


def test_a_running_job_refuses_a_second_one(tmp_path) -> None:
    assert _launch(tmp_path, "echo first; sleep 2", stop_after=False).returncode == 0
    _wait_for(tmp_path / "run.pid")

    refused = _launch(tmp_path, "echo second", stop_after=False)

    assert refused.returncode == 3
    assert b"still running" in refused.stderr
    assert "second" not in (tmp_path / "run.log").read_text(encoding="utf-8")
    assert not (tmp_path / "run.previous.log").exists(), "the running job kept its log"


def test_the_pod_finds_its_own_id_before_stopping_itself() -> None:
    # Found on a real pod: over ssh RUNPOD_POD_ID is empty and runpodctl has no config,
    # so a bare `runpodctl stop pod "$RUNPOD_POD_ID"` fails and the pod bills on.
    stop = runpod_gpu.STOP_SELF

    assert stop.index("source /etc/rp_environment") < stop.index("runpodctl config")
    assert stop.index("runpodctl config") < stop.index("runpodctl stop pod")
    assert "&&" not in stop, "config exits 1 after saving, the stop must still run"
    assert runpod_gpu.STOP_SELF in runpod_gpu.remote_job("x", max_hours=1, stop_after=True)


def test_the_remote_job_can_leave_the_pod_running() -> None:
    assert "runpodctl" not in runpod_gpu.remote_job("bash run.sh", max_hours=6, stop_after=False)


def test_fetch_leaves_the_weights_behind_unless_asked() -> None:
    light = " ".join(
        runpod_gpu.rsync_pull(("h", 1), Path("k"), "artifacts/exp", "out", weights=False)
    )
    heavy = " ".join(
        runpod_gpu.rsync_pull(("h", 1), Path("k"), "artifacts/exp", "out", weights=True)
    )

    assert "--exclude=*.safetensors" in light
    assert "--exclude=*.safetensors" not in heavy
