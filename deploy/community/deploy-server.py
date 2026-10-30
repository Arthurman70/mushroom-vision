"""Run only on the existing VPS, after the local release has passed review/tests."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import urllib.request

BASE = Path("/root/gleaner-mushroom-lab")
CADDY = Path("/root/tradechaser/Caddyfile")
PROXY = "tradechaser-caddy-1"
SERVICE = "gleaner-mushroom-lab"
ORIGIN = "https://thegreatlakesgleaner.press"
HEALTH = "/mushroom-lab/api/health"
BEGIN, END = "\t# BEGIN MYCOSCOUT COMMUNITY", "\t# END MYCOSCOUT COMMUNITY"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(args, *, input=None, capture=False, timeout=180):
    return subprocess.run(args, input=input, check=True, stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None, timeout=timeout).stdout


def caddy(data, action):
    run(["docker", "exec", "-i", PROXY, "caddy", action, "--config", "/dev/stdin", "--adapter", "caddyfile"], input=data, timeout=30)


def patched_config(original, fragment):
    text = original.decode()
    if text.count(BEGIN) != text.count(END) or text.count(BEGIN) > 1:
        raise ValueError("Ambiguous existing community route markers")
    if BEGIN in text:
        start, stop = text.index(BEGIN), text.index(END) + len(END)
        return (text[:start] + fragment.rstrip() + text[stop:]).encode()
    match = re.search(r"^thegreatlakesgleaner\.press, www\.thegreatlakesgleaner\.press \{\r?$", text, re.M)
    if not match:
        raise ValueError("Expected live Gleaner host block not found")
    if "/mushroom-lab" in text:
        raise ValueError("An unmanaged mushroom-lab route already exists")
    return (text[:match.end()] + "\n" + fragment.rstrip() + text[match.end():]).encode()


def write_config(data, expected):
    if digest(CADDY.read_bytes()) != expected:
        raise RuntimeError("Caddy changed concurrently; preserving the newer configuration")
    # Preserve the inode: Caddy's Docker mount binds this individual host file.
    with CADDY.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def compose(release, *args):
    release = Path(release).resolve(strict=True)
    if release.parent != BASE / "releases" or not re.fullmatch(r"[a-z0-9-]{6,80}", release.name):
        raise ValueError("Release must be a direct child of the managed releases directory")
    env = dict(os.environ, RELEASE_ID=release.name)
    subprocess.run(["docker", "compose", "--project-name", SERVICE, "-f", str(release / "compose.yaml"), *args], env=env, check=True, timeout=180)


def wait_ready(frames):
    code = "fetch('http://127.0.0.1:8791" + HEALTH + "').then(async r=>{const j=await r.json();if(!r.ok||!j.ok||j.frames!==" + str(frames) + ")process.exit(1)}).catch(()=>process.exit(1))"
    for _ in range(15):
        check = subprocess.run(["docker", "exec", SERVICE, "node", "-e", code], capture_output=True, timeout=10)
        if check.returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError("Contributor service did not become ready")


def verify_release(release):
    manifest = json.loads((release / "release-manifest.json").read_text())
    if manifest.get("schema") != "mycoscout.community-release.v1":
        raise ValueError("Invalid release manifest")
    for name, expected in manifest["files"].items():
        path = (release / name).resolve(strict=True)
        if not path.is_relative_to(release) or path.is_symlink() or not path.is_file():
            raise ValueError("Release path escaped or is not regular")
        data = path.read_bytes()
        if len(data) != expected["bytes"] or digest(data) != expected["sha256"]:
            raise ValueError(f"Release integrity mismatch: {name}")
    return manifest


def rollback(backup):
    backup = Path(backup).resolve(strict=True)
    if backup.parent != BASE / "backups":
        raise ValueError("Backup outside managed directory")
    record = json.loads((backup / "deployment.json").read_text())
    previous = record.get("previousRelease")
    if previous:
        compose(previous, "up", "-d", "--no-build")
    else:
        # Remove only this service; retain its persistent database and releases.
        found = subprocess.run(["docker", "inspect", SERVICE], capture_output=True)
        if found.returncode == 0:
            info = json.loads(found.stdout)[0]
            if info["Config"].get("Labels", {}).get("com.docker.compose.project") != SERVICE:
                raise ValueError("Refusing to stop an unmanaged container")
            run(["docker", "stop", SERVICE], timeout=30)
            run(["docker", "rm", SERVICE], timeout=30)
    before = (backup / "Caddyfile").read_bytes()
    current = digest(CADDY.read_bytes())
    if current == record["candidateCaddySha256"]:
        caddy(before, "validate")
        write_config(before, current)
        caddy(before, "reload")
    elif current != record["originalCaddySha256"]:
        raise RuntimeError("Caddy changed after this deployment; manual route-only reconciliation is required")
    if record.get("previousState"):
        (BASE / "current.json").write_text(json.dumps(record["previousState"], indent=2))
    elif (BASE / "current.json").exists():
        (BASE / "current.json").unlink()
    print(json.dumps({"rolledBack": True, "backup": str(backup), "reviewsPreserved": True}))


def publish(release):
    release = release.resolve(strict=True)
    if release.parent != BASE / "releases" or not re.fullmatch(r"[a-z0-9-]{6,80}", release.name):
        raise ValueError("Unmanaged release directory")
    manifest = verify_release(release)
    if shutil.disk_usage(BASE).free < 1024 ** 3:
        raise RuntimeError("At least 1 GiB free space is required")
    previous = json.loads((BASE / "current.json").read_text()) if (BASE / "current.json").exists() else None
    if not previous and subprocess.run(["docker", "inspect", SERVICE], capture_output=True).returncode == 0:
        raise ValueError("Existing service lacks managed deployment state")
    before = CADDY.read_bytes()
    candidate = patched_config(before, (release / "caddy-route.txt").read_text())
    caddy(candidate, "validate")
    run(["docker", "build", "-t", SERVICE + ":" + release.name, "-f", str(release / "Dockerfile"), str(release)], timeout=300)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = BASE / "backups" / (stamp + "-" + release.name)
    backup.mkdir(parents=True, exist_ok=False)
    os.chmod(backup, 0o700)
    (backup / "Caddyfile").write_bytes(before)
    record = {"release": str(release), "previousRelease": previous.get("release") if previous else None,
              "previousState": previous, "originalCaddySha256": digest(before), "candidateCaddySha256": digest(candidate)}
    (backup / "deployment.json").write_text(json.dumps(record, indent=2))
    data = BASE / "data"
    data.mkdir(exist_ok=True)
    os.chown(data, 1000, 1000)
    os.chmod(data, 0o700)
    db = data / "community.sqlite"
    if db.exists():
        with sqlite3.connect("file:" + str(db) + "?mode=ro", uri=True) as source, sqlite3.connect(backup / "community.sqlite") as destination:
            source.backup(destination)
    try:
        compose(release, "up", "-d", "--no-build")
        wait_ready(manifest["frames"])
        write_config(candidate, digest(before))
        caddy(candidate, "reload")
        with urllib.request.urlopen(ORIGIN + HEALTH, timeout=20) as response:
            health = json.load(response)
        if not health.get("ok") or health.get("frames") != manifest["frames"]:
            raise RuntimeError("Public health verification failed")
        state = {"release": str(release), "backup": str(backup), "frames": manifest["frames"], "publishedAt": stamp,
                 "url": ORIGIN + "/mushroom-lab/", "caddySha256": digest(candidate)}
        (BASE / "current.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(state, indent=2))
    except BaseException:
        rollback(backup)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["publish", "rollback", "status"])
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()
    if args.mode == "publish":
        publish(Path(__file__).resolve().parent)
    elif args.mode == "rollback":
        if not args.backup:
            parser.error("--backup is required")
        rollback(args.backup)
    else:
        path = BASE / "current.json"
        print(path.read_text() if path.exists() else json.dumps({"deployed": False}))
