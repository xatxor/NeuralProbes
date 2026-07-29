# Sandbox the agent executes code in. Built once, then run with networking disabled, so everything
# the agent can import has to be present at build time.
FROM python:3.12

# Pinned so a workload's difficulty cannot drift between runs when pytest changes its behaviour.
RUN pip install --no-cache-dir pytest==9.1.1

# Containers run with --user matching the host account and a read-only root filesystem, so HOME has
# to point somewhere writable; /tmp is mounted as a tmpfs at run time.
ENV HOME=/tmp
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /work
