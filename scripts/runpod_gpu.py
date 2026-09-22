#!/usr/bin/env python3
"""Rent a GPU on RunPod for a training run, and give it back.

    uv run python scripts/runpod_gpu.py status
    uv run python scripts/runpod_gpu.py offers              # stock and price per GPU
    uv run python scripts/runpod_gpu.py up                  # rent, wait for ssh
    uv run python scripts/runpod_gpu.py wait                # keep waiting after a timeout
    uv run python scripts/runpod_gpu.py sync --data artifacts/data/goemotions.jsonl
    uv run python scripts/runpod_gpu.py setup               # uv and the locked env
    uv run python scripts/runpod_gpu.py run "bash path/to/experiment.sh"
    uv run python scripts/runpod_gpu.py logs
    uv run python scripts/runpod_gpu.py fetch artifacts/exp # reports back, no weights
    uv run python scripts/runpod_gpu.py start               # wake a stopped pod to fetch
    uv run python scripts/runpod_gpu.py down                # terminate, stop paying

The key is read from RUNPOD_API_KEY, or from ~/.config/exu/runpod.key (mode 600). It
is never written anywhere and never printed, errors included. Your ssh public key
has to be registered in the RunPod account: the official images install it.

Money guards, because a forgotten pod bills until someone notices:

- `up` refuses without credit, and refuses a second pod while one is recorded;
- the pod is recorded in artifacts/runpod/<name>.json (git-ignored), so `down` always
  knows what to terminate, and the record is only dropped once the API confirms.
  `--pod NAME` (default `pod`) addresses one of several pods, each with its record;
- `setup` refuses to run while an install is already running on the pod: a second
  one competes with the first and neither finishes for a long time;
- `run` wraps the job in `timeout`, and stops the pod a grace period after the job
  ends, so there is time to `fetch` and `down`. Fetch inside that window: a stopped
  pod keeps its volume and bills storage only, but it has no ssh, and `start` fails
  when the host has rented the GPU to someone else, which took under a minute once.

torch in uv.lock is a CUDA 13 build, so the pod is only placed on hosts whose driver
supports CUDA 13. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

API = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
KEY_FILE = Path("~/.config/exu/runpod.key").expanduser()
STATE_DIR = Path("artifacts/runpod")
STATE = STATE_DIR / "pod.json"
REMOTE_DIR = "/workspace/exu"
# The image of the most used official template: hosts already have it, and a pod was
# up in 18 seconds. A tag republished that day took 14 billed minutes to pull. Its own
# torch does not matter, `uv sync` installs the locked one: only the driver has to match.
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
CUDA_VERSIONS = ["13.0"]
# Cheapest first among cards with bfloat16 and room for a base encoder.
GPUS = ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090"]
# A Community host at 183 Mbps took over 12 billed minutes to pull the default image.
MIN_DOWNLOAD_MBPS = 400
# An ssh session does not inherit the container environment: the pod id and the key
# scoped to this pod are only in /etc/rp_environment, and runpodctl starts unconfigured.
# `config` exits 1 after saving (that key may not touch the account's ssh keys), hence `;`.
STOP_SELF = (
    "source /etc/rp_environment; "
    'runpodctl config --apiKey "$RUNPOD_API_KEY" > /dev/null 2>&1; '
    'runpodctl stop pod "$RUNPOD_POD_ID"'
)
NEVER_SEND = (".safetensors", ".key", ".pem", ".env")
NEVER_SEND_NAMES = {".env", "id_rsa", "id_ed25519", "runpod.key"}

Transport = Callable[[str, str, dict[str, str], Any], tuple[int, Any]]


def load_key(key_file: Path = KEY_FILE) -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key and key_file.is_file():
        key = key_file.read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit(f"no API key: set RUNPOD_API_KEY or write it to {key_file} (mode 600)")
    return key


def http(method: str, url: str, headers: dict[str, str], body: Any) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw, status = response.read(), response.status
    except urllib.error.HTTPError as error:
        raw, status = error.read(), error.code
    try:
        return status, json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return status, raw.decode(errors="replace")


def create_payload(
    name: str,
    gpus: Sequence[str],
    cloud: str,
    volume_gb: int,
    image: str = IMAGE,
    min_download_mbps: int = MIN_DOWNLOAD_MBPS,
) -> dict[str, Any]:
    return {
        "name": name,
        "imageName": image,
        "gpuTypeIds": list(gpus),
        # The default, "availability", ignores the order and may pick the dearest.
        "gpuTypePriority": "custom",
        "gpuCount": 1,
        "cloudType": cloud,
        "allowedCudaVersions": CUDA_VERSIONS,
        "minDownloadMbps": min_download_mbps,
        "containerDiskInGb": 30,
        "volumeInGb": volume_gb,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "interruptible": False,
    }


def ssh_endpoint(pod: dict[str, Any]) -> tuple[str, int] | None:
    address = pod.get("publicIp")
    port = (pod.get("portMappings") or {}).get("22")
    return (address, int(port)) if address and port else None


def expand(paths: Sequence[str]) -> list[str]:
    """rsync --files-from does not descend into a folder, so name every file in it."""
    files: list[str] = []
    for path in paths:
        entry = Path(path)
        if entry.is_dir():
            files += sorted(str(found) for found in entry.rglob("*") if found.is_file())
        elif entry.is_file():
            files.append(path)
        else:
            raise SystemExit(f"--data {path} does not exist")
    return files


def port_open(endpoint: tuple[str, int]) -> bool:
    try:
        with socket.create_connection(endpoint, timeout=5):
            return True
    except OSError:
        return False


def safe_to_send(paths: Sequence[str]) -> list[str]:
    return [
        path
        for path in paths
        if not path.endswith(NEVER_SEND) and Path(path).name not in NEVER_SEND_NAMES
    ]


def _ssh_options(endpoint: tuple[str, int], identity: Path) -> str:
    return (
        f"ssh -p {endpoint[1]} -i {identity} -o StrictHostKeyChecking=accept-new "
        "-o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    )


def setup_remote() -> str:
    """Install uv and the locked environment, unless an install is already running."""
    return (
        "if pgrep -x -f 'uv sync --frozen' > /dev/null; then "
        'echo "an install is already running on the pod: wait for it" >&2; exit 3; fi; '
        "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; "
        f'export PATH="$HOME/.local/bin:$PATH"; cd {REMOTE_DIR} && uv sync --frozen && '
        'uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available()); '
        "assert torch.cuda.is_available(), 'CUDA does not initialise on this host: down and rent again'; "
        'print(torch.cuda.get_device_name(0))"'
    )


def prepare_remote() -> str:
    """The official PyTorch image ships without rsync, and both directions need it there."""
    return (
        f"mkdir -p {REMOTE_DIR} && (command -v rsync >/dev/null || "
        "(apt-get update -qq && apt-get install -y -qq rsync >/dev/null))"
    )


def rsync_push(endpoint: tuple[str, int], identity: Path, files_from: str) -> list[str]:
    return [
        "rsync", "-az", "--files-from=" + files_from, "-e", _ssh_options(endpoint, identity),
        ".", f"root@{endpoint[0]}:{REMOTE_DIR}/",
    ]  # fmt: skip


def rsync_pull(
    endpoint: tuple[str, int], identity: Path, remote: str, local: str, *, weights: bool
) -> list[str]:
    command = ["rsync", "-az", "-e", _ssh_options(endpoint, identity)]
    if not weights:
        command.append("--exclude=*.safetensors")
    return [*command, f"root@{endpoint[0]}:{REMOTE_DIR}/{remote.rstrip('/')}/", f"{local}/"]


def remote_job(
    command: str,
    *,
    max_hours: int,
    stop_after: bool,
    grace_minutes: float = 15,
    stop_command: str = STOP_SELF,
) -> str:
    """Shell that runs `command` detached on the pod, time-limited, then stops the pod.

    `run.done` holds the exit code once the job ends. The pod only stops after the
    grace period: a stopped pod has no ssh, so this is the window to fetch results.

    `run.pid` is the wrapper. A job that is still running refuses a second one, which
    would take over its log. A finished job still holds a pending stop, and that has
    to die before the next job starts, or the pod stops in the middle of it.
    """
    job = (
        "echo $$ > run.pid; rm -f run.done; "
        f"timeout {max_hours}h bash -lc {shlex.quote(command)}; echo $? > run.done"
    )
    if stop_after:
        job += f"; sleep {grace_minutes * 60:g}; {stop_command}"
    guard = (
        'if [ -f run.pid ] && kill -0 "$(cat run.pid)" 2>/dev/null; then '
        'if [ -f run.done ]; then kill "$(cat run.pid)"; '
        'else echo "a job is still running on the pod: see logs" >&2; exit 3; fi; fi; '
        "[ -f run.log ] && mv -f run.log run.previous.log"
    )
    # The parentheses matter: `cd && nohup ... &` backgrounds the whole list in a subshell
    # that keeps the ssh channel open, and the local command hangs until the job ends.
    detached = f"nohup bash -c {shlex.quote(job)} > run.log 2>&1 < /dev/null &"
    return f"cd {REMOTE_DIR} && {{ {guard}; ({detached}); }}"


def offers_query(min_download_mbps: int, secure: bool) -> dict[str, str]:
    floor = f"gpuCount:1, secureCloud:{str(secure).lower()}, minDownload:{int(min_download_mbps)}"
    return {
        "query": "query { gpuTypes { id memoryInGb lowestPrice(input:{" + floor + "}) "
        "{ stockStatus uninterruptablePrice minVcpu minMemory } } }"
    }


def offer_rows(answer: dict[str, Any]) -> list[tuple[float, str, int, str, int, int]]:
    """In-stock GPUs, cheapest first. The API reports out-of-stock ones with nulls."""
    rows = []
    for gpu in answer["data"]["gpuTypes"]:
        offer = gpu.get("lowestPrice") or {}
        if offer.get("stockStatus") and offer.get("uninterruptablePrice"):
            rows.append((float(offer["uninterruptablePrice"]), gpu["id"], int(gpu["memoryInGb"]),
                         offer["stockStatus"], int(offer.get("minVcpu") or 0),
                         int(offer.get("minMemory") or 0)))  # fmt: skip
    return sorted(rows)


class RunPod:
    def __init__(self, key: str, *, transport: Transport = http, state: Path = STATE) -> None:
        self._key = key
        self._transport = transport
        self.state = state

    def _call(self, method: str, path: str, body: Any = None, *, url: str | None = None) -> Any:
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            # Cloudflare answers 403 (error 1010) to the default Python-urllib agent.
            "User-Agent": "exu-runpod/0.1",
        }
        status, answer = self._transport(method, url or API + path, headers, body)
        if status >= 400:
            detail = json.dumps(answer)[:300].replace(self._key, "<key>")
            raise SystemExit(f"RunPod answered {status} to {method} {path}: {detail}")
        return answer

    def balance(self) -> float:
        query = {"query": "query { myself { clientBalance currentSpendPerHr } }"}
        answer = self._call("POST", "/graphql", query, url=GRAPHQL)
        return float(answer["data"]["myself"]["clientBalance"])

    def offers(self, min_download_mbps: int, secure: bool) -> list[tuple[Any, ...]]:
        query = offers_query(min_download_mbps, secure)
        return offer_rows(self._call("POST", "/graphql", query, url=GRAPHQL))

    def list(self) -> list[dict[str, Any]]:
        return self._call("GET", "/pods") or []

    def recorded(self) -> str | None:
        if not self.state.is_file():
            return None
        return json.loads(self.state.read_text(encoding="utf-8")).get("id")

    def pod(self) -> dict[str, Any]:
        identifier = self.recorded()
        if identifier is None:
            raise SystemExit(f"no pod recorded in {self.state}: run `up` first")
        return self._call("GET", f"/pods/{identifier}")

    def up(
        self,
        name: str,
        gpus: Sequence[str],
        cloud: str,
        volume_gb: int,
        image: str = IMAGE,
        min_download_mbps: int = MIN_DOWNLOAD_MBPS,
    ) -> dict[str, Any]:
        if self.recorded() is not None:
            raise SystemExit(f"pod {self.recorded()} is already recorded: `down` it first")
        if self.balance() <= 0:
            raise SystemExit("the account balance is zero: add credit before renting a GPU")
        payload = create_payload(name, gpus, cloud, volume_gb, image, min_download_mbps)
        pod = self._call("POST", "/pods", payload)
        self.state.parent.mkdir(parents=True, exist_ok=True)
        self.state.write_text(json.dumps({"id": pod["id"]}), encoding="utf-8")
        return pod

    def wait_for_ssh(
        self,
        minutes: int,
        *,
        pause: Callable[[float], None] = time.sleep,
        reachable: Callable[[tuple[str, int]], bool] = port_open,
    ) -> tuple[str, int]:
        # A host that never saw the image pulls all of it first, 10 GB for the default.
        # After `start` the API keeps naming the port of the previous run for some
        # seconds, so the endpoint only counts once something answers on it.
        deadline = time.monotonic() + minutes * 60
        while True:
            pod = self.pod()
            endpoint = ssh_endpoint(pod)
            if endpoint and reachable(endpoint):
                return endpoint
            if pod.get("desiredStatus") != "RUNNING":
                raise SystemExit(f"the pod went to {pod.get('desiredStatus')} before ssh came up")
            if time.monotonic() >= deadline:
                raise SystemExit(f"no ssh endpoint after {minutes} minutes. The pod is still "
                                 "billing: run `wait` to keep waiting, or `down`")  # fmt: skip
            pause(15)

    def stop(self) -> None:
        self._call("POST", f"/pods/{self.pod()['id']}/stop")

    def start(self) -> None:
        # Fails when the host rented the GPU to someone else in the meantime.
        self._call("POST", f"/pods/{self.pod()['id']}/start")

    def down(self) -> None:
        identifier = self.recorded()
        if identifier is None:
            raise SystemExit(f"no pod recorded in {self.state}")
        self._call("DELETE", f"/pods/{identifier}")
        self.state.unlink()


def _ssh(endpoint: tuple[str, int], identity: Path, command: str) -> int:
    return subprocess.call([*shlex.split(_ssh_options(endpoint, identity)),
                            f"root@{endpoint[0]}", command])  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rent a GPU on RunPod and give it back.")
    parser.add_argument("--identity", type=Path, default=Path("~/.ssh/id_rsa").expanduser())
    parser.add_argument("--pod", default="pod",
                        help="which pod this command acts on; each has its own record")  # fmt: skip
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("status", help="balance, recorded pod, every pod on the account")
    offers = actions.add_parser("offers", help="GPUs in stock, with a download-speed floor")
    offers.add_argument("--min-download-mbps", type=int, default=MIN_DOWNLOAD_MBPS)
    up = actions.add_parser("up", help="rent a pod and wait for ssh")
    up.add_argument("--name", default="exu")
    up.add_argument("--gpu", action="append", help="GPU type id, repeat in order of preference")
    up.add_argument("--cloud", choices=("COMMUNITY", "SECURE"), default="COMMUNITY")
    up.add_argument("--volume-gb", type=int, default=40)
    up.add_argument("--image", default=IMAGE)
    up.add_argument("--min-download-mbps", type=int, default=MIN_DOWNLOAD_MBPS)
    up.add_argument("--wait-minutes", type=int, default=25)
    wait = actions.add_parser("wait", help="wait for ssh on the recorded pod")
    wait.add_argument("--wait-minutes", type=int, default=25)
    sync = actions.add_parser("sync", help="send the working tree and the named data files")
    sync.add_argument("--data", action="append", default=[],
                      help="git-ignored file or folder to send too")  # fmt: skip
    actions.add_parser("setup", help="install uv and the locked environment on the pod")
    run = actions.add_parser("run", help="start a detached job on the pod")
    run.add_argument("command")
    run.add_argument("--max-hours", type=int, default=8)
    run.add_argument("--keep-running", action="store_true", help="do not stop the pod at the end")
    run.add_argument("--grace-minutes", type=float, default=15,
                     help="how long the pod stays up after the job, to fetch results")  # fmt: skip
    actions.add_parser("logs", help="tail the job log")
    actions.add_parser("ssh", help="print the ssh command")
    fetch = actions.add_parser("fetch", help="bring a folder back, without weights by default")
    fetch.add_argument("remote")
    fetch.add_argument("--to", default=None)
    fetch.add_argument("--weights", action="store_true")
    actions.add_parser("stop", help="stop the pod: the volume stays, the GPU stops billing")
    actions.add_parser("start", help="start a stopped pod again, then `wait`")
    actions.add_parser("down", help="terminate the pod and drop the record")
    args = parser.parse_args(argv)

    pods = RunPod(load_key(), state=STATE_DIR / f"{args.pod}.json")
    if args.action == "status":
        recorded = {path.stem: json.loads(path.read_text(encoding="utf-8")).get("id")
                    for path in sorted(STATE_DIR.glob("*.json"))}  # fmt: skip
        print(f"balance: ${pods.balance():.2f} | recorded: {recorded or None}")
        for pod in pods.list():
            print(f"  {pod['id']}  {pod.get('desiredStatus')}  ${pod.get('costPerHr')}/h  "
                  f"{(pod.get('machine') or {}).get('gpuTypeId')}  {pod.get('name')}")  # fmt: skip
        return 0
    if args.action == "up":
        pod = pods.up(args.name, args.gpu or GPUS, args.cloud, args.volume_gb, args.image,
                      args.min_download_mbps)  # fmt: skip
        print(f"pod {pod['id']} at ${pod.get('costPerHr')}/h, waiting for ssh...", flush=True)
    if args.action in ("up", "wait"):
        address, port = pods.wait_for_ssh(args.wait_minutes)
        print(f"ready: ssh -p {port} -i {args.identity} root@{address}")
        return 0
    if args.action == "offers":
        for secure in (False, True):
            print(f"{'secure' if secure else 'community'} cloud, "
                  f"download >= {args.min_download_mbps} Mbps (CUDA version not filtered):")  # fmt: skip
            for price, gpu, memory, stock, vcpu, ram in pods.offers(args.min_download_mbps, secure):
                print(f"  ${price:<5} {gpu:<34} {memory:>3} GB  stock {stock:<7} "
                      f"{vcpu} vCPU, {ram} GB RAM")  # fmt: skip
        return 0
    if args.action == "stop":
        pods.stop()
        return 0
    if args.action == "start":
        pods.start()
        return 0
    if args.action == "down":
        pods.down()
        print("terminated")
        return 0

    endpoint = ssh_endpoint(pods.pod())
    if endpoint is None:
        raise SystemExit("the pod has no ssh endpoint yet")
    if args.action == "ssh":
        print(f"ssh -p {endpoint[1]} -i {args.identity} root@{endpoint[0]}")
        return 0
    if args.action == "sync":
        listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"],
                                check=True, capture_output=True, text=True).stdout.split("\n")  # fmt: skip
        files = safe_to_send([path for path in [*listed, *expand(args.data)] if path])
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as manifest:
            manifest.write("\n".join(files) + "\n")
            manifest.flush()
            if _ssh(endpoint, args.identity, prepare_remote()) != 0:
                raise SystemExit("could not prepare the pod: no folder, or no rsync")
            return subprocess.call(rsync_push(endpoint, args.identity, manifest.name))
    if args.action == "setup":
        return _ssh(endpoint, args.identity, setup_remote())
    if args.action == "run":
        command = f'export PATH="$HOME/.local/bin:$PATH"; {args.command}'
        script = remote_job(command, max_hours=args.max_hours, stop_after=not args.keep_running,
                            grace_minutes=args.grace_minutes)  # fmt: skip
        return _ssh(endpoint, args.identity, script)
    if args.action == "logs":
        show = f'cd {REMOTE_DIR} && tail -n 40 run.log; [ -f run.done ] && echo "exit code: $(cat run.done)"'
        return _ssh(endpoint, args.identity, show)
    if args.action == "fetch":
        local = args.to or args.remote
        Path(local).mkdir(parents=True, exist_ok=True)
        return subprocess.call(rsync_pull(endpoint, args.identity, args.remote, local,
                                          weights=args.weights))  # fmt: skip
    return 2


if __name__ == "__main__":
    sys.exit(main())
