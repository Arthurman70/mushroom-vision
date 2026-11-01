"""Explicit recovery of one interrupted V5 run; never changes frozen inputs.

Saved evidence is independently revalidated and copied into a new directory.
Only missing frozen jobs run. Total attempted work is unknown because the failed
process may have finished unrecorded in-flight work before shutting down.
"""
from __future__ import annotations
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.evaluate_full_pixel_v5 import (
    ROOT, OUT, aggregate, digest, evaluate_job, new_json, now, read, rel, verify, worker_init,
)
from experiments.verify_full_pixel_v5 import read_hashed_json, validate_prediction_record

RECOVERY = OUT / "recovery-01"


def atomic_json_retry(path, value, attempts=30, delay=.2):
    """Windows readers can briefly deny replacement while still permitting reads."""
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    for attempt in range(attempts):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt + 1 == attempts: raise
            time.sleep(delay)


def copy_bound(source, destination, expected):
    raw = Path(source).read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != expected: raise ValueError("Recovery input changed")
    with Path(destination).open("xb") as stream: stream.write(raw)
    if digest(destination) != expected: raise ValueError("Recovery copy differs")


def prepare_recovery(source=OUT, destination=RECOVERY):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists(): raise FileExistsError("Recovery output must be a new directory")
    if not destination.is_relative_to(ROOT): raise ValueError("Recovery output outside project")
    protocol = read(source / "protocol.json"); report = read(source / "report.json")
    protocol_hash, failed_hash = digest(source / "protocol.json"), digest(source / "report.json")
    if (report.get("status") != "failed" or report.get("error", {}).get("type") != "PermissionError" or
        report.get("protocolSha256") != protocol_hash or read(source / "protocol-sha256.json")["sha256"] != protocol_hash):
        raise ValueError("Expected this explicitly interrupted permission-error run")
    verify(protocol)
    jobs = {(j["frameId"], j["conditionId"]): j for j in protocol["jobs"]}
    saved = {(r["frameId"], r["conditionId"]): r for r in report["records"]}
    if len(saved) != len(report["records"]) or not set(saved) <= set(jobs): raise ValueError("Invalid saved identity set")
    ledger = []
    for identity, record in saved.items():
        path = ROOT / record["predictionPath"]
        if path.parent != source / "predictions": raise ValueError("Unexpected saved prediction location")
        prediction = read_hashed_json(path, record["predictionSha256"])
        validate_prediction_record(jobs[identity], record, prediction, protocol["expectedModel"])
        ledger.append({"frameId": identity[0], "conditionId": identity[1], "sourcePath": rel(path),
            "sha256": record["predictionSha256"], "recordOrder": record["order"]})
    if {ROOT / item["sourcePath"] for item in ledger} != set((source / "predictions").glob("*.json")):
        raise ValueError("Unaccounted saved prediction file; do not silently discard it")
    missing = [j for j in protocol["jobs"] if (j["frameId"], j["conditionId"]) not in saved]
    if len(saved) != 92 or len(missing) != 9: raise ValueError("This recovery was authorized for92saved/9missing scenes")
    if digest(source / "report.json") != failed_hash: raise ValueError("Failed report changed while validating")
    destination.mkdir(); (destination / "predictions").mkdir()
    copy_bound(source / "protocol.json", destination / "protocol.json", protocol_hash)
    copy_bound(source / "protocol-sha256.json", destination / "protocol-sha256.json", digest(source / "protocol-sha256.json"))
    copy_bound(source / "report.json", destination / "failed-report.json", failed_hash)
    for item in ledger:
        copy_bound(ROOT / item["sourcePath"], destination / "predictions" / Path(item["sourcePath"]).name, item["sha256"])
    plan = {"schema": "mushroom-vision.full-pixel-recovery-plan.v1", "createdAt": now(),
        "protocolSha256": protocol_hash, "failedReportSha256": failed_hash,
        "originalFailedReportPath": rel(source / "report.json"), "recoveryWrapperSha256": digest(Path(__file__)),
        "sourcePredictions": ledger, "missingJobs": [{"frameId": j["frameId"], "conditionId": j["conditionId"], "order": j["order"]} for j in missing],
        "reusedScenes": 92, "newScenes": 9, "uniquePlannedModelForwards": 505, "newRecoveryModelForwards": 45,
        "totalAttemptedModelForwards": None, "workers": 2, "torchThreadsPerWorker": 4,
        "softwareVersions": protocol["softwareVersions"], "settings": protocol["settings"], "expectedModel": protocol["expectedModel"],
        "cause": "Windows denied an atomic report replacement. The next error-report write succeeded, consistent with a transient file lock; the holding process is not known.",
        "disclosure": "Original failed report and92saved predictions are preserved unchanged. Only9missing frozen jobs are rerun. The failed executor may have completed unrecorded in-flight work, so total attempted forwards are unknown.505means unique persisted planned forwards, not a clean uninterrupted505forward run."}
    new_json(destination / "recovery-plan.json", plan)
    print(json.dumps({"preparedRecovery": rel(destination), "reusedScenes": 92, "missingScenes": 9,
        "planSha256": digest(destination / "recovery-plan.json"), "attemptedTotal": None}), flush=True)
    return plan


def run_recovery(destination=RECOVERY):
    destination = Path(destination).resolve(); plan = read(destination / "recovery-plan.json")
    plan_hash = digest(destination / "recovery-plan.json")
    if digest(Path(__file__)) != plan["recoveryWrapperSha256"]: raise ValueError("Prepared recovery wrapper changed")
    protocol = read(destination / "protocol.json"); verify(protocol)
    if digest(destination / "protocol.json") != plan["protocolSha256"]: raise ValueError("Protocol changed")
    original = read(destination / "failed-report.json")
    if digest(destination / "failed-report.json") != plan["failedReportSha256"]: raise ValueError("Archived failure changed")
    if (destination / "report.json").exists(): raise FileExistsError("No silent retry of an existing recovery")
    records = copy.deepcopy(original["records"])
    for record in records:
        path = destination / "predictions" / Path(record["predictionPath"]).name
        if digest(path) != record["predictionSha256"]: raise ValueError("Saved copy changed")
        record["predictionPath"] = rel(path)
    report = {k: copy.deepcopy(v) for k, v in original.items() if k not in ("error", "failedAt", "records")}
    report.update(status="recovering", records=records, recoveryStartedAt=now(),
        recovery={"planSha256": plan_hash, "failedReportSha256": plan["failedReportSha256"],
            "reusedScenes": 92, "newScenes": 9, "uniquePersistedModelForwards": 505,
            "totalAttemptedModelForwards": None, "disclosure": plan["disclosure"]})
    atomic_json_retry(destination / "report.json", report)
    new_keys = {(j["frameId"], j["conditionId"]) for j in plan["missingJobs"]}
    jobs = [j for j in protocol["jobs"] if (j["frameId"], j["conditionId"]) in new_keys]
    if len(jobs) != 9: raise ValueError("Recovery job set changed")
    started = time.perf_counter(); pending = []
    executor = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"), initializer=worker_init)
    try:
        pending = [executor.submit(evaluate_job, job) for job in jobs]
        for future in as_completed(pending):
            record, prediction = future.result()
            validate_prediction_record(next(j for j in jobs if j["order"] == record["order"]), record, prediction, protocol["expectedModel"])
            path = destination / "predictions" / f"{record['frameId']}--{record['conditionId']}.json"
            new_json(path, prediction); record.update(predictionPath=rel(path), predictionSha256=digest(path))
            report["records"].append(record); report["records"].sort(key=lambda r: r["order"])
            report["progress"] = {"completed": len(report["records"]) * 5, "total": 505, "scenesCompleted": len(report["records"]), "scenesTotal": 101}
            report["cohorts"] = aggregate(report["records"])
            report.update(recoveryWallSeconds=time.perf_counter() - started,
                elapsedWallSeconds=(datetime.now(timezone.utc) - datetime.fromisoformat(report["startedAt"])).total_seconds(),
                summedSceneInferenceSeconds=sum(r["sceneWallSeconds"] for r in report["records"]), model=prediction["model"])
            atomic_json_retry(destination / "report.json", report)
            print(json.dumps({"recoveryCompletedScenes": len(report["records"]) - 92, "persistedUniqueForwards": report["progress"]["completed"],
                "frameId": record["frameId"], "rawHits": record["methods"]["raw"]["metrics"]["hits"],
                "tileHits": record["methods"]["tiles"]["metrics"]["hits"], "recoverySeconds": report["recoveryWallSeconds"]}), flush=True)
        verify(protocol)
        if digest(destination / "recovery-plan.json") != plan_hash or len(report["records"]) != 101: raise ValueError("Recovery integrity failed")
        receipt = {"schema": "mushroom-vision.full-pixel-recovery-receipt.v1", "status": "completed", "completedAt": now(),
            "protocolSha256": plan["protocolSha256"], "planSha256": plan_hash, "failedReportSha256": plan["failedReportSha256"],
            "recoveryWrapperSha256": plan["recoveryWrapperSha256"], "reusedScenes": 92, "newScenes": 9,
            "uniquePersistedModelForwards": 505, "newRecoveryModelForwards": 45, "totalAttemptedModelForwards": None,
            "predictionHashes": {Path(r["predictionPath"]).name: r["predictionSha256"] for r in report["records"]},
            "disclosure": plan["disclosure"], "settings": protocol["settings"], "softwareVersions": protocol["softwareVersions"],
            "recoveryWallSeconds": time.perf_counter() - started, "originalReportedWallSeconds": original["elapsedWallSeconds"],
            "elapsedWallSecondsIncludesInterruption": True, "trainingPerformed": False, "weightsChanged": False}
        new_json(destination / "recovery-receipt.json", receipt)
        report["recovery"]["receiptSha256"] = digest(destination / "recovery-receipt.json")
        report.update(status="completed", completedAt=now(), inputIntegrityVerified=True)
        atomic_json_retry(destination / "report.json", report)
    except Exception as exc:
        for future in pending: future.cancel()
        report.update(status="failed", failedAt=now(), error={"type": type(exc).__name__, "message": str(exc)})
        atomic_json_retry(destination / "report.json", report); raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    print(json.dumps({"recovered": True, "output": rel(destination), "uniquePersistedModelForwards": 505,
        "totalAttemptedModelForwards": None}), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args(); {"prepare": prepare_recovery, "run": run_recovery}[args.action]()
