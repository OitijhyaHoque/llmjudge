"""Running the judge from a Colab cell, with the results kept on Drive.

A Colab runtime is temporary: the tab is closed, the session is recycled, the GPU quota
runs out. Drive is what survives. So the judge runs on `/content` -- local disk, where
append and fsync mean what they say -- and the results are copied up to Drive every
couple of minutes and once more at the end. Writing straight to the Drive FUSE mount is
not reliable under append, which is why the results are not simply written there.

    from llmjudge.colab import run

    run(items="drive:genmd-judge/items.jsonl",
        out="drive:genmd-judge/results/pilot",
        run_tag="pilot-01", prompt="c3", limit=100)

Nothing is pasted in: `drive:` is `MyDrive/`, and the URL, key and model name come from
the serving cell, which exported them. Re-running the cell resumes -- whatever the last
session left on Drive is copied back down first, and a row already answered is never
sent again.

There are no bundles and no zips. The items file is the only thing a colleague puts on
Drive, and the code comes from a tag:

    pip install "git+https://$GH_TOKEN@github.com/<org>/llmjudge.git@v0.1.0"
"""

from __future__ import annotations

import os
import shutil
import threading

from .run import judge, sha

DRIVE_ROOT = "/content/drive"
MYDRIVE = os.path.join(DRIVE_ROOT, "MyDrive")
LOCAL_ROOT = "/content"
BACKUP_EVERY = 100                     # seconds between copies to Drive

# Where the model is, in the order the names are looked at. `notebooks/serve_vllm.py`
# exports the LLMJUDGE_* names, so a cell that has just started a server needs to say
# nothing at all. The MEDGEMMA_* names are what the older notebooks exported and they
# still work.
ENV_BASE_URL = ("LLMJUDGE_BASE_URL", "MEDGEMMA_BASE_URL")
ENV_API_KEY = ("LLMJUDGE_API_KEY", "MEDGEMMA_API_KEY")
ENV_MODEL = ("LLMJUDGE_MODEL", "MEDGEMMA_MODEL")

# The judge's own codes are 0 ok, 2 refused, 3 incomplete. This one is: the rows were
# judged, but they are not all on Drive, so the run is not safe to walk away from.
EXIT_MIRROR = 4


# --------------------------------------------------------------------------------------
# Drive


def mount(root: str = DRIVE_ROOT) -> bool:
    """Mount Drive if this is Colab and it is not mounted yet. -> is MyDrive there.

    Off Colab this does nothing and returns False, so every function below can be run
    and tested on an ordinary machine."""
    if os.path.isdir(os.path.join(root, "MyDrive")):
        return True
    try:
        from google.colab import drive
    except ImportError:
        return False
    drive.mount(root)
    return os.path.isdir(os.path.join(root, "MyDrive"))


def resolve(path: str) -> str:
    """`drive:genmd-judge/x` -> `/content/drive/MyDrive/genmd-judge/x`.

    Any other path is used as it is written, so the same cell works off Colab."""
    if path.startswith("drive:"):
        return os.path.join(MYDRIVE, path[len("drive:"):].lstrip("/"))
    return os.path.abspath(os.path.expanduser(path))


def restore(remote: str, local: str) -> int:
    """Bring the last session's files down from Drive. -> how many were copied.

    Only files that are not there locally: a file that is already local belongs to this
    session and is ahead of the copy on Drive."""
    n = 0
    for root, _, names in os.walk(remote):
        target = os.path.join(local, os.path.relpath(root, remote))
        os.makedirs(target, exist_ok=True)
        for name in sorted(names):
            src, dst = os.path.join(root, name), os.path.join(target, name)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                n += 1
    return n


def sync(local: str, remote: str) -> list[tuple[str, str]]:
    """Copy `local` to `remote`. -> [(file, error)] for whatever could not be copied.

    A `.jsonl` file is copied by its new tail only. `results.jsonl` is append-only and
    can hold 50,000 rows, and re-uploading all of it every hundred seconds would spend
    the whole run pushing bytes Drive already has. Everything else is small and is
    copied whole when it changes.
    """
    problems: list[tuple[str, str]] = []
    for root, _, names in os.walk(local):
        sub = os.path.relpath(root, local)
        target = remote if sub == "." else os.path.join(remote, sub)
        try:
            os.makedirs(target, exist_ok=True)
        except OSError as e:
            problems.append((sub, str(e)))
            continue
        for name in sorted(names):
            src, dst = os.path.join(root, name), os.path.join(target, name)
            if not os.path.isfile(src) or name == "judge.pid":
                continue               # a pid from a dead runtime means nothing on Drive
            try:
                if name.endswith(".jsonl"):
                    have = os.path.getsize(dst) if os.path.exists(dst) else 0
                    if os.path.getsize(src) > have:
                        with open(src, "rb") as a, open(dst, "ab") as b:
                            a.seek(have)
                            shutil.copyfileobj(a, b)
                            b.flush()
                            os.fsync(b.fileno())
                elif (not os.path.exists(dst)
                      or os.path.getmtime(src) > os.path.getmtime(dst)):
                    shutil.copy2(src, dst)
            except OSError as e:
                problems.append((os.path.join(sub, name) if sub != "." else name, str(e)))
    return problems


def lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))


# --------------------------------------------------------------------------------------
# where the model is


def from_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def from_secrets(name: str) -> str:
    """A Colab Secret (the key icon in the sidebar), or "".

    Every failure is the same answer -- not Colab, no such secret, notebook access not
    granted -- and each raises a different Colab-internal exception, so they are caught
    together."""
    try:
        from google.colab import userdata
        return (userdata.get(name) or "").strip()
    except Exception:                                      # noqa: BLE001 - see docstring
        return ""


def endpoint(options: dict) -> None:
    """Fill `base_url`, `model` and `api_key` into `options`, in place.

    What the cell passed wins, then what the serving cell exported into the environment,
    then Colab Secrets. Nothing is guessed: a run with no endpoint is refused here rather
    than a minute later inside the judge.
    """
    if options.get("endpoints"):
        return                                    # a file of several servers; not our job
    url = (options.get("base_url") or from_env(ENV_BASE_URL)
           or from_secrets("LLMJUDGE_BASE_URL"))
    model = options.get("model") or from_env(ENV_MODEL) or from_secrets("LLMJUDGE_MODEL")
    key = options.get("api_key") or from_env(ENV_API_KEY) or from_secrets("LLMJUDGE_API_KEY")
    if not url or not model:
        raise RuntimeError(
            "no endpoint. Run the serving cell first, or pass base_url= and model= (or "
            f"endpoints= for a file of several). Looked at {', '.join(ENV_BASE_URL)} and "
            f"{', '.join(ENV_MODEL)} in the environment, and at Colab Secrets.")
    options["base_url"], options["model"] = url, model
    if key:
        options["api_key"] = key                  # a server on localhost may want none
    if "timeout" not in options and ("127.0.0.1" in url or "localhost" in url):
        # The 95 s default is set by Cloudflare, which cuts a request whose origin has
        # been silent for about 100 s. There is no tunnel in front of localhost, so a
        # reasoning prompt may take as long as it takes.
        options["timeout"] = 600


# --------------------------------------------------------------------------------------
# the cell


def run(items: str, out: str, run_tag: str, system: str | None = None,
        user: str | None = None, backup_every: float = BACKUP_EVERY, **options) -> int:
    """Judge `items` into `out`, keeping `out` on Drive. -> the exit code.

    `items` and `out` may start with `drive:`, which is `MyDrive/`. Everything else is a
    `llmjudge.judge()` option: `prompt=`, or `system=` and `user=` with your own text;
    `guided=`, `max_tokens=`, `limit=`, `retry_errors=`, and so on.

    Exit codes are the judge's -- 0 every planned row recorded, 2 refused, 3 stopped
    incomplete -- plus 4: the rows were judged but they are not all on Drive.
    """
    mounted = mount()
    items_path, drive_out = resolve(items), resolve(out)
    on_drive = drive_out == MYDRIVE or drive_out.startswith(MYDRIVE + os.sep)
    if on_drive and not mounted:
        raise RuntimeError(f"{out} is on Drive, but Drive is not mounted and this is not "
                           f"a Colab runtime")
    if not os.path.isfile(items_path):
        raise FileNotFoundError(f"no items file {items_path}")

    if on_drive:
        # Two results directories on Drive can share a basename, and they would then
        # share one directory on /content and pollute each other's resume. The hash of
        # the Drive path keeps them apart and is stable across sessions, so re-running
        # the same cell lands in the same local directory.
        name = os.path.basename(drive_out.rstrip(os.sep))
        local_out = os.path.join(LOCAL_ROOT, f"{name}-{sha(drive_out)[:8]}")
        os.makedirs(drive_out, exist_ok=True)
        os.makedirs(local_out, exist_ok=True)
        restored = restore(drive_out, local_out)
        print(f"results  {drive_out}"
              + (f"   (restored {restored} file(s) to resume)" if restored else ""))
    else:
        local_out = drive_out                     # not on Drive: nothing to mirror
        os.makedirs(local_out, exist_ok=True)

    options = dict(options)
    endpoint(options)

    stop = threading.Event()
    if on_drive:
        def mirror() -> None:
            while not stop.wait(backup_every):
                report(sync(local_out, drive_out))

        threading.Thread(target=mirror, name="llmjudge-drive", daemon=True).start()

    try:
        code = judge(items=items_path, out=local_out, run_tag=run_tag,
                     system=system, user=user, **options)
    finally:
        stop.set()
    if not on_drive:
        return code

    # The mirror is verified, not assumed. A sync that failed mid-run printed its error
    # and carried on, which is right while there is time to recover and wrong at the end:
    # a run whose answers are only on a runtime that is about to be recycled has to say so.
    problems = report(sync(local_out, drive_out))
    here = lines(os.path.join(local_out, "results.jsonl"))
    there = lines(os.path.join(drive_out, "results.jsonl"))
    print(f"\nresults on Drive: {drive_out}   ({there:,} rows)")
    if problems or here != there:
        print(f"MIRROR INCOMPLETE: results.jsonl has {here:,} lines on {local_out} and "
              f"{there:,} on Drive. The answers are still on this runtime. Copy them off "
              f"before it is recycled.", flush=True)
        return EXIT_MIRROR
    print("Re-run this cell to resume. 0 missing in the summary means the pool is done.")
    return code


def report(problems: list[tuple[str, str]]) -> list[tuple[str, str]]:
    for name, error in problems:
        print(f"[drive] {name}: {error}", flush=True)
    return problems
