"""Exercise the real local checkpoint through its deployed newline JSON protocol."""
import base64
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    dataset=json.loads((ROOT/"data/public-fungi-v3/manifest.json").read_text(encoding="utf-8"))
    selected=[]
    for class_id in ("black_trumpets","true_morels","hericium","bolete_research"):
        row=next(r for r in dataset["images"] if r["split"]=="validation" and r["classId"]==class_id)
        selected.append(row)
    requests=[]
    for row in selected:
        payload=(ROOT/"data/public-fungi-v3"/row["path"]).read_bytes()
        requests.append({"id":row["id"],"imageBase64":base64.b64encode(payload).decode("ascii")})
    requests.insert(1,{"id":"invalid-image","imageBase64":"this is not base64"})
    requests.insert(2,{"id":["bad","identifier"],"imageBase64":""})
    started=time.perf_counter()
    result=subprocess.run([sys.executable,"-m","vision.multiclass","worker","--checkpoint","vision/checkpoints/groups-v1.json"],
        cwd=ROOT,input="\n".join(json.dumps(r) for r in requests)+"\n",text=True,capture_output=True,timeout=90,check=True)
    responses=[json.loads(line) for line in result.stdout.splitlines()]
    assert len(responses)==len(requests)
    assert responses[1]["status"]=="error" and responses[2]["status"]=="error" and responses[2]["id"] is None
    summaries=[]
    for request,response in zip(requests,responses):
        if response["status"]=="error": continue
        assert response["id"]==request["id"]
        guess=response["result"]
        assert len(guess["guesses"])==3 and guess["humanVerificationRequired"] and not guess["speciesConfirmed"]
        assert not guess["flightGateEnabled"] and not guess["salesApproval"]
        assert all(0<=g["modelScore"]<=1 for g in guess["guesses"])
        summaries.append({"id":request["id"],"topGuess":guess["guesses"][0],"abstained":guess["abstained"],"processingMilliseconds":guess["processingMilliseconds"]})
    report={"schema":"mycoscout.group-worker-smoke.v1","realModel":True,"requests":len(requests),"successfulImages":len(summaries),
        "invalidInputsRejected":2,"workerSurvivedInvalidInputs":True,"elapsedSeconds":time.perf_counter()-started,
        "examples":summaries,"note":"Four validation examples verify runtime protocol only; not an accuracy estimate."}
    output=ROOT/"experiments/results/groups-v1/worker-smoke.json"
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
