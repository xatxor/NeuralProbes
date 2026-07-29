#! /usr/bin/env python

"""Run the agent's tool calls inside a locked-down container.

One container per episode: it starts when the episode starts, every tool call is a `docker exec`
into it, and it is torn down at the end. State therefore persists across tool calls the way it would
on a real machine -- the agent writes a file with one call and runs pytest over it with the next.

The container has no network, a read-only root filesystem, and a single writable bind mount holding
the episode directory. Running this file directly verifies each of those claims against a live
container rather than asserting them.
"""

import argparse
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("sandbox")

# Backend selection. Docker stays the default because the V100 box is shared with other people and
# that is the only irreversible risk available to us. DataSphere sets this to "subprocess": there,
# docker, docker-in-docker and unshare are all blocked -- measured, not assumed -- and the job
# container is itself ephemeral and single-tenant, so it is already the isolation boundary.
BACKEND = os.environ.get("AGENTIC_SANDBOX", "docker")

IMAGE = "neuralprobes-agentic:1"
# Locked by decision. These fix how hard workload 1's timing requirement is, so they must be
# identical for every episode or its branch rate stops meaning anything.
CPUS = "2"
MEMORY = "2g"
PIDS = 256
TIMEOUT = 30


@dataclass
class Result:
    """What one tool call produced."""

    code: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def output(self) -> str:
        """Both streams as the agent will see them, stderr last."""
        parts = [self.stdout.rstrip(), self.stderr.rstrip()]
        return "\n".join(part for part in parts if part)


class Sandbox:
    """A per-episode container over one writable directory."""

    def __init__(
        self,
        workdir: Path,
        image: str = IMAGE,
        cpus: str = CPUS,
        memory: str = MEMORY,
        pids: int = PIDS,
        timeout: int = TIMEOUT,
    ) -> None:
        """
        :param workdir: host directory bind-mounted at /work; created if absent and never deleted.
        :param image: the built sandbox image.
        :param cpus: `--cpus` value, as a string so fractional values pass through unchanged.
        :param memory: `--memory` value.
        :param pids: `--pids-limit` value, a fork-bomb backstop.
        :param timeout: default wall-clock seconds for one tool call.
        """
        self.workdir = workdir
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self.pids = pids
        self.timeout = timeout
        self.container: str | None = None

    def start(self) -> None:
        """Launch the container and leave it idling until tool calls arrive."""
        if self.container:
            raise RuntimeError("sandbox already started")
        self.workdir.mkdir(parents=True, exist_ok=True)
        if BACKEND == "subprocess":
            # Nothing to launch. On DataSphere there is no container to start: the job's own
            # container is the boundary, and it is torn down when the job ends.
            log.info(f"subprocess backend over {self.workdir}")
            return
        argv = [
            "docker", "run", "--detach",
            "--network", "none",
            "--read-only",
            # pytest and the interpreter need somewhere writable that is not the bind mount.
            "--tmpfs", "/tmp:rw,size=256m,exec",
            "--volume", f"{self.workdir.resolve()}:/work:rw",
            # Files land owned by the host account, so they can be read and removed without sudo.
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--cpus", self.cpus,
            "--memory", self.memory,
            "--pids-limit", str(self.pids),
            "--workdir", "/work",
            self.image,
            "sleep", "infinity",
        ]
        done = subprocess.run(argv, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"docker run failed: {done.stderr.strip()}")
        self.container = done.stdout.strip()
        log.info(f"container {self.container[:12]} over {self.workdir}")

    def run(self, argv: list[str], timeout: int | None = None, cwd: str | None = None) -> Result:
        """Execute one command inside the container.

        :param argv: the command and its arguments, passed without a shell.
        :param timeout: wall-clock seconds, defaulting to the sandbox's own.
        :param cwd: directory inside the container to run in, defaulting to /work.

        :return: exit code, both output streams, and whether the timeout fired.
        """
        limit = self.timeout if timeout is None else timeout

        if BACKEND == "subprocess":
            # /work exists only inside a container, so paths the workload uses are remapped onto the
            # host episode directory. The timeout is enforced here rather than in-container because
            # subprocess.run kills the process it started -- the leak that bit us with `docker exec`
            # does not apply when we own the child directly.
            where = self.workdir.resolve()
            if cwd:
                where = where / cwd.removeprefix("/work").lstrip("/")
            try:
                done = subprocess.run(
                    argv, capture_output=True, text=True, timeout=limit, cwd=str(where)
                )
            except subprocess.TimeoutExpired:
                return Result(code=124, stdout="", stderr=f"timed out after {limit}s", timed_out=True)
            except FileNotFoundError as problem:
                return Result(code=127, stdout="", stderr=str(problem), timed_out=False)
            return Result(code=done.returncode, stdout=done.stdout, stderr=done.stderr, timed_out=False)

        if not self.container:
            raise RuntimeError("sandbox not started")
        place = ["--workdir", cwd] if cwd else []
        # The timeout has to run *inside* the container. Killing the `docker exec` client leaves the
        # process running -- measured: a 60 s sleep survived its client being killed at 2 s, and would
        # have kept eating the container's CPU budget for the rest of the episode. The outer timeout
        # below is only a backstop for a wedged daemon.
        wrapped = ["timeout", "--kill-after=2", str(limit), *argv]
        try:
            done = subprocess.run(
                ["docker", "exec", *place, self.container, *wrapped],
                capture_output=True,
                text=True,
                timeout=limit + 10,
            )
        except subprocess.TimeoutExpired:
            return Result(code=124, stdout="", stderr=f"docker exec wedged after {limit + 10}s", timed_out=True)
        if done.returncode in (124, 137):
            return Result(code=done.returncode, stdout=done.stdout, stderr=f"timed out after {limit}s", timed_out=True)
        return Result(code=done.returncode, stdout=done.stdout, stderr=done.stderr, timed_out=False)

    def stop(self) -> None:
        """Remove the container. The bind-mounted directory is left in place as evidence."""
        if BACKEND == "subprocess" or not self.container:
            return
        subprocess.run(["docker", "rm", "--force", self.container], capture_output=True, text=True)
        log.info(f"container {self.container[:12]} removed; {self.workdir} kept")
        self.container = None

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("episodes/sandbox-check"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        log.info(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            failures.append(name)

    with Sandbox(args.workdir) as box:
        version = box.run(["python", "--version"])
        check("python present", version.code == 0, version.output)

        pytest_version = box.run(["python", "-m", "pytest", "--version"])
        check("pytest present", pytest_version.code == 0, pytest_version.output)

        # A real failing test, so we know the agent will actually see red output.
        written = box.run([
            "python", "-c",
            "open('/work/test_demo.py','w').write('def test_pass():\\n    assert 1 == 1\\n"
            "def test_fail():\\n    assert 2 == 3\\n')",
        ])
        check("write into /work", written.code == 0, written.output or "(no output)")

        tests = box.run(["python", "-m", "pytest", "-q", "test_demo.py"])
        check(
            "pytest runs and reports a failure",
            tests.code != 0 and "1 failed" in tests.output and "1 passed" in tests.output,
            tests.output.strip().splitlines()[-1] if tests.output else "(no output)",
        )

        owned = (args.workdir / "test_demo.py").stat()
        check(
            "files owned by the host account",
            owned.st_uid == os.getuid(),
            f"uid {owned.st_uid} (host {os.getuid()})",
        )

        readonly = box.run(["python", "-c", "open('/usr/local/blocked','w')"])
        check("root filesystem read-only", readonly.code != 0, readonly.stderr.strip().splitlines()[-1] if readonly.stderr else "(no error)")

        tmp = box.run(["python", "-c", "open('/tmp/allowed','w').write('x')"])
        check("/tmp writable", tmp.code == 0, tmp.output or "(no output)")

        # No DNS and no route: both should fail fast rather than hang, hence the short timeout.
        network = box.run(
            ["python", "-c", "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)"],
            timeout=15,
        )
        check("network disabled", network.code != 0, network.stderr.strip().splitlines()[-1] if network.stderr else "(no error)")

        limits = box.run(["python", "-c", "import os; print(os.cpu_count(), open('/sys/fs/cgroup/memory.max').read().strip())"])
        check("cgroup limits visible", limits.code == 0, limits.output)

        slept = box.run(["python", "-c", "import time; time.sleep(60)"], timeout=3)
        check("timeout fires", slept.timed_out, slept.stderr)

        # The point of the in-container wrapper: nothing may survive its own timeout.
        survivors = box.run(["ps", "-eo", "args", "--no-headers"])
        check(
            "timed-out process actually died",
            "time.sleep(60)" not in survivors.stdout,
            "no survivors" if "time.sleep(60)" not in survivors.stdout else survivors.stdout.strip(),
        )

    if failures:
        raise SystemExit(f"sandbox checks failed: {', '.join(failures)}")
    log.info("every sandbox check passed")


if __name__ == "__main__":
    main()
