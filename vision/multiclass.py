"""Local frozen-feature mushroom-group guesser. All results require human review."""
from __future__ import annotations
import argparse
import base64
import binascii
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from .transfer import ROOT, EXPECTED_WEIGHT_SHA, read_json, write_json, checksum, photo_path, load_backbone, extract_features

SCHEMA = "mycoscout.group-classifier.v1"
DATA_SCHEMA = "mycoscout.group-photo-dataset.v3"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_REQUEST_LINE = 11_200_000
LIMITATIONS = [
    "Weak observation taxon labels are not independently verified species or image-content labels.",
    "Research taxon classes are broader than Michigan certification targets and exclusions.",
    "Entoloma abortivum form and chaga growth form are not verified by these labels.",
    "This model classifies a photo or crop; it does not locate mushrooms or measure range.",
    "No woodland-background or drone-camera operational test exists; unfamiliar backgrounds may receive confident guesses.",
    "Temperature calibration and abstention are empirical validation fits, not correctness guarantees.",
    "No edibility, sales approval, or automatic flight decision is supplied.",
    "Preprocessing uses a central crop and may omit off-center evidence.",
    "Inherited V2 holdout images are reused evaluation, not a newly untouched dataset.",
]


def validate_manifest(path):
    data = read_json(path)
    if data.get("schema") != DATA_SCHEMA or data.get("status") != "frozen":
        raise ValueError("A frozen V3 group dataset is required")
    if checksum(path) != read_json(Path(path).with_name("manifest-sha256.json"))["sha256"]:
        raise ValueError("Dataset freeze checksum mismatch")
    classes = {c["classId"] for c in data["classes"]}
    groups, ids, hashes = defaultdict(set), set(), set()
    for row in data["images"]:
        if row["classId"] not in classes or row["id"] in ids or row["sha256"] in hashes:
            raise ValueError("Invalid label, duplicate ID or exact duplicate")
        ids.add(row["id"]); hashes.add(row["sha256"])
        if row.get("synthetic") is not False or row.get("split") not in {"train","validation","test"}:
            raise ValueError("Only original photos with frozen splits are accepted")
        if row.get("labelSource")=="human_review":
            rights=row.get("rights",{})
            if rights.get("trainingOptIn") is not True or not rights.get("license") or not rights.get("attestedBy"):
                raise ValueError("Human-reviewed photo lacks training permission")
            if row.get("review",{}).get("action") not in {"confirm_group","correct_species"} or not row.get("review",{}).get("reviewerName"):
                raise ValueError("Human-reviewed photo lacks explicit review")
        elif row.get("photoLicense",{}).get("code") not in {"cc0","cc-by","cc-by-sa"}:
            raise ValueError("Unsupported photo license")
        for field in ("groupId","splitGroupId","observerId","observationId","photoId","sha256","normalizedPixelSha256"):
            groups[(field,str(row[field]))].add(row["split"])
        if checksum(photo_path(path,row)) != row["sha256"]:
            raise ValueError("Photo checksum changed")
    if any(len(partitions)>1 for partitions in groups.values()):
        raise ValueError("Observer/content leakage across dataset partitions")
    return data


def softmax(logits):
    values = np.asarray(logits,dtype=np.float64)
    values = np.exp(values-values.max(axis=-1,keepdims=True))
    return values/values.sum(axis=-1,keepdims=True)


def nll(scores, labels):
    return float(-np.log(np.clip(scores[np.arange(len(labels)),labels],1e-12,1)).mean())


def ece(scores, labels):
    confidence=scores.max(axis=1); correct=scores.argmax(axis=1)==labels
    bins=[]; total=0.0
    for lower,upper in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(confidence>=lower)&(confidence<upper if upper<1 else confidence<=upper)
        count=int(mask.sum())
        accuracy=float(correct[mask].mean()) if count else None
        average=float(confidence[mask].mean()) if count else None
        if count: total+=count/len(labels)*abs(accuracy-average)
        bins.append({"lower":float(lower),"upper":float(upper),"count":count,"accuracy":accuracy,"meanScore":average})
    return {"expectedCalibrationError":float(total),"bins":bins}


def evaluate(scores, labels, classes, accept=None):
    pred=scores.argmax(axis=1); count=len(labels)
    top3=np.argsort(-scores,axis=1)[:,:3]
    per=[]
    matrix=np.zeros((len(classes),len(classes)),dtype=int)
    for actual,guess in zip(labels,pred): matrix[actual,guess]+=1
    for index,info in enumerate(classes):
        tp=int(matrix[index,index]); fp=int(matrix[:,index].sum()-tp); fn=int(matrix[index,:].sum()-tp)
        precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
        per.append({"classId":info["classId"],"groupId":info["groupId"],"support":int(matrix[index,:].sum()),
                    "truePositive":tp,"falsePositive":fp,"falseNegative":fn,"precision":precision,"recall":recall,
                    "f1":2*precision*recall/(precision+recall) if precision+recall else 0.0})
    present=[p for p in per if p["support"]]
    coarse=np.array([c["groupId"] for c in classes])
    result={"photos":count,"top1Accuracy":float(np.mean(pred==labels)),"top3Accuracy":float(np.mean(np.any(top3==labels[:,None],axis=1))),
            "coarseGroupAccuracy":float(np.mean(coarse[pred]==coarse[labels])),"macroF1":float(np.mean([p["f1"] for p in present])),
            "negativeLogLikelihood":nll(scores,labels),"calibration":ece(scores,labels),"perClass":per,"confusionMatrix":matrix.tolist()}
    if accept is not None:
        accepted=int(accept.sum())
        result["reviewPolicy"]={"acceptedGuesses":accepted,"abstained":count-accepted,"coverage":accepted/count,
            "acceptedAccuracy":float(np.mean(pred[accept]==labels[accept])) if accepted else None,
            "acceptedErrors":int(np.sum(pred[accept]!=labels[accept])),
            "note":"Accepted still requires human verification; this is not a safety decision."}
    return result


def fit_head(features, labels, supported, l2, class_count):
    import torch
    torch.manual_seed(37)
    x=torch.tensor(features,dtype=torch.float64); y=torch.tensor(labels,dtype=torch.int64)
    counts=np.bincount(labels,minlength=class_count)
    weight=np.zeros(class_count); weight[supported]=len(labels)/(len(supported)*counts[supported])
    weights=torch.tensor(weight,dtype=torch.float64)
    head=torch.nn.Linear(features.shape[1],class_count,dtype=torch.float64)
    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    unsupported=[i for i in range(class_count) if i not in supported]
    optimizer=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=100,tolerance_grad=1e-7,line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        logits=head(x)
        if unsupported: logits[:,unsupported]=-1e9
        loss=torch.nn.functional.cross_entropy(logits,y,weight=weights)+l2*head.weight.square().sum()/class_count
        loss.backward(); return loss
    optimizer.step(closure)
    return head.weight.detach().numpy(),head.bias.detach().numpy()


def logits_for(features, weight, bias, supported):
    logits=features@weight.T+bias
    unsupported=[i for i in range(len(bias)) if i not in supported]
    if unsupported: logits[:,unsupported]=-1e9
    return logits


def accept_mask(scores, classes, threshold, margin):
    order=np.argsort(-scores,axis=1)
    result=(scores[np.arange(len(scores)),order[:,0]]>=threshold)&(
        scores[np.arange(len(scores)),order[:,0]]-scores[np.arange(len(scores)),order[:,1]]>=margin)
    for i,top in enumerate(order[:,0]):
        if classes[top]["classId"]=="other_fungi" or not classes[top].get("supported",True): result[i]=False
    return result


def choose_policy(scores, labels, classes, target_accuracy=.8):
    if not .5<=target_accuracy<=1: raise ValueError("Target validation accuracy must be between 0.5 and 1")
    trials=[]
    for threshold in np.arange(.25,.951,.05):
        for margin in (.0,.05,.1,.2,.3):
            mask=accept_mask(scores,classes,float(threshold),margin)
            n=int(mask.sum()); correct=int(np.sum(scores.argmax(axis=1)[mask]==labels[mask]))
            trials.append({"threshold":round(float(threshold),2),"margin":margin,"accepted":n,
                           "accuracy":correct/n if n else None,"coverage":n/len(labels)})
    eligible=[t for t in trials if t["accepted"]>=20 and t["accuracy"]>=target_accuracy]
    if not eligible:
        return {"threshold":1.01,"margin":1.01,"targetValidationAccuracy":target_accuracy,"targetMet":False,"selected":None,"trials":trials}
    chosen=max(eligible,key=lambda t:(t["accepted"],t["accuracy"],t["threshold"],t["margin"]))
    return {"threshold":chosen["threshold"],"margin":chosen["margin"],"targetValidationAccuracy":target_accuracy,"targetMet":True,"selected":chosen,"trials":trials}


def train(manifest, output, checkpoint, weights, min_train_photos=25, min_validation_photos=5, target_validation_accuracy=.8):
    import torch
    torch.set_num_threads(4)
    if Path(checkpoint).exists() or (Path(output)/"study.json").exists():
        raise ValueError("Checkpoint or experiment is already frozen; choose new --output and --checkpoint version paths")
    if not 10<=min_train_photos<=500 or not 3<=min_validation_photos<=200 or not .5<=target_validation_accuracy<=1:
        raise ValueError("Support bounds: train 10–500, validation 3–200; target accuracy 0.5–1")
    started=time.perf_counter(); data=validate_manifest(manifest)
    classes=[dict(c) for c in data["classes"]]; names=[c["classId"] for c in classes]
    rows=data["images"]
    original=np.array([names.index(r["classId"]) for r in rows],dtype=np.int64)
    split=np.array([r["split"] for r in rows]); trainmask=split=="train"; valmask=split=="validation"; testmask=split=="test"
    for info in classes:
        support={}
        for partition in ("train","validation","test"):
            selected=[r for r in rows if r["split"]==partition and r["classId"]==info["classId"]]
            support[partition+"Photos"]=len(selected)
            support[partition+"Observers"]=len({r["observerId"] for r in selected if r.get("labelSource")!="human_review"})
            support[partition+"LocalSourceGroups"]=len({r["groupId"] for r in selected if r.get("labelSource")=="human_review"})
            support[partition+"SourceGroups"]=len({r["groupId"] for r in selected})
        info["support"]=support
        # Do not consult test metrics/counts when deciding whether a class can be emitted.
        info["supported"]=(support["trainPhotos"]>=min_train_photos and support["validationPhotos"]>=min_validation_photos and support["trainSourceGroups"]>=10 and support["validationSourceGroups"]>=3)
    other=names.index("other_fungi")
    if not classes[other]["supported"]: raise ValueError("Insufficient other-fungi comparison coverage")
    supported=[i for i,c in enumerate(classes) if c["supported"]]
    labels=original.copy(); labels[~np.isin(labels,supported)]=other
    arrays, extraction=extract_features(data,manifest,weights,Path(output)/"features",threads=4)
    x=arrays["resnet18"]
    mean=x[trainmask].mean(axis=0); scale=x[trainmask].std(axis=0); scale[scale<1e-6]=1
    x=(x-mean)/scale
    trials=[]; fitted=[]
    for l2 in (.01,.1,1.0):
        weight,bias=fit_head(x[trainmask],labels[trainmask],supported,l2,len(classes))
        logits=logits_for(x[valmask],weight,bias,supported); scores=softmax(logits)
        metrics=evaluate(scores,original[valmask],classes)
        trials.append({"l2":l2,"validationMacroF1":metrics["macroF1"],"validationNll":metrics["negativeLogLikelihood"]})
        fitted.append((weight,bias,logits))
        print(json.dumps({"event":"head-fit",**trials[-1]}),flush=True)
    chosen=max(range(len(trials)),key=lambda i:(trials[i]["validationMacroF1"],-trials[i]["validationNll"]))
    weight,bias,val_logits=fitted[chosen]
    # The same validation set calibrates and selects the review policy; test is read only after these choices freeze.
    temperature_grid=np.geomspace(.25,4,81)
    temperature=float(min(temperature_grid,key=lambda t:nll(softmax(val_logits/t),original[valmask])))
    val_scores=softmax(val_logits/temperature)
    policy=choose_policy(val_scores,original[valmask],classes,target_validation_accuracy)
    scores=softmax(logits_for(x,weight,bias,supported)/temperature)
    accepted=accept_mask(scores,classes,policy["threshold"],policy["margin"])
    evidence="groups-v1-"+checksum(manifest)[:12]
    metrics={partition:evaluate(scores[mask],original[mask],classes,accepted[mask]) for partition,mask in (("validation",valmask),("test",testmask))}
    for cohort,key in (("inherited_v2","testInheritedV2"),("new_v3","testNewV3"),("reviewed_local","testReviewedLocal")):
        mask=testmask&np.array([r.get("cohort")==cohort for r in rows])
        if mask.any(): metrics[key]=evaluate(scores[mask],original[mask],classes,accepted[mask])
    calibration={"method":"validation-temperature","temperature":temperature,"empirical":True,"guaranteed":False,
                 "validationUsedForModelSelectionAndPolicy":True,
                 "validationBefore":ece(softmax(val_logits),original[valmask]),"validationAfter":ece(val_scores,original[valmask])}
    payload={"schema":SCHEMA,"evidenceVersion":evidence,"classes":classes,"architecture":"frozen official ResNet18 + linear softmax head",
        "backbone":{"path":"models/newfeaturecache/resnet18-f37072fd.pth","sha256":EXPECTED_WEIGHT_SHA,
                    "source":"https://download.pytorch.org/models/resnet18-f37072fd.pth","preprocessing":"resize shorter side 256, center crop 224, ImageNet normalization"},
        "mean":mean.tolist(),"scale":scale.tolist(),"weight":weight.tolist(),"bias":bias.tolist(),"supportedIndices":supported,
        "scoreCalibration":calibration,"abstention":{k:v for k,v in policy.items() if k!="trials"},
        "training":{"manifestSha256":checksum(manifest),"photos":len(rows),"trainingPhotos":int(trainmask.sum()),"validationPhotos":int(valmask.sum()),"testPhotos":int(testmask.sum()),
                    "selectedL2":trials[chosen]["l2"],"selection":"maximum validation macro-F1, then minimum validation NLL",
                    "classSupportPolicy":f"train>={min_train_photos} photos/10 source groups and validation>={min_validation_photos} photos/3 source groups; public observer and local session groups are separate units, not proof of photographer independence; unsupported labels map to other_fungi in fitting only",
                    "parameters":{"minTrainPhotos":min_train_photos,"minValidationPhotos":min_validation_photos,"targetValidationAccuracy":target_validation_accuracy},
                    "source":"mixed public iNaturalist and opted-in reviewed local images" if data.get("reviewImport") else "https://www.inaturalist.org",
                    "sourceCounts":dict(Counter("reviewed_local" if r.get("labelSource")=="human_review" else "public_inaturalist" for r in rows)),
                    "licenses":dict(Counter(r["photoLicense"]["code"] for r in rows)),"parentDataset":data["parentDataset"]},
        "metrics":metrics,"humanVerificationRequired":True,"speciesConfirmed":False,"flightGateEnabled":False,"limitations":LIMITATIONS}
    write_json(checkpoint,payload)
    study={"schema":"mycoscout.group-study.v1","evidenceVersion":evidence,"manifestSha256":checksum(manifest),"classes":classes,
        "selectionTrials":trials,"scoreCalibration":calibration,"abstention":policy,"metrics":metrics,"extraction":extraction,
        "elapsedSeconds":time.perf_counter()-started,"limitations":LIMITATIONS}
    write_json(Path(output)/"study.json",study)
    predictions=[]
    for i,row in enumerate(rows):
        if row["split"]=="train": continue
        order=np.argsort(-scores[i])[:3]
        predictions.append({"id":row["id"],"split":row["split"],"inherited":row["inherited"],"actualWeakClass":row["classId"],
                            "abstained":not bool(accepted[i]),"guesses":[{"classId":classes[k]["classId"],"modelScore":float(scores[i,k])} for k in order]})
    write_json(Path(output)/"predictions.json",{"evidenceVersion":evidence,"predictions":predictions})
    print(json.dumps({"event":"trained","checkpoint":str(checkpoint),"testTop1":metrics["test"]["top1Accuracy"],"testReview":metrics["test"]["reviewPolicy"]}),flush=True)
    return study


def load_checkpoint(path):
    data=read_json(path)
    if data.get("schema")!=SCHEMA or data.get("flightGateEnabled") is not False:
        raise ValueError("Unsupported or unsafe checkpoint metadata")
    classes=data.get("classes",[]); count=len(classes)
    for key,shape in (("mean",(512,)),("scale",(512,)),("weight",(count,512)),("bias",(count,))):
        value=np.asarray(data[key],dtype=np.float64)
        if value.shape!=shape or not np.isfinite(value).all(): raise ValueError("Invalid checkpoint numerical array")
    if np.any(np.asarray(data["scale"])<=0): raise ValueError("Invalid feature scale")
    if data["backbone"]["sha256"]!=EXPECTED_WEIGHT_SHA: raise ValueError("Unapproved backbone hash")
    if not 0<float(data["scoreCalibration"]["temperature"])<=10: raise ValueError("Invalid temperature")
    if len({c["classId"] for c in classes})!=count: raise ValueError("Duplicate class IDs")
    supported=data.get("supportedIndices",[])
    if len(supported)<3 or len(set(supported))!=len(supported) or any(type(i) is not int or not 0<=i<count for i in supported):
        raise ValueError("Invalid supported class indices")
    return data


class Predictor:
    def __init__(self,checkpoint):
        import torch
        torch.set_num_threads(4)
        self.data=load_checkpoint(checkpoint)
        # Fixed repository-local backbone: checkpoint cannot select arbitrary files or network destinations.
        self.model,self.transform=load_backbone(ROOT/"models/newfeaturecache/resnet18-f37072fd.pth")

    def predict(self,image):
        import torch
        started=time.perf_counter()
        with torch.inference_mode(): feature=self.model(self.transform(image).unsqueeze(0)).numpy()
        data=self.data
        feature=(feature-np.asarray(data["mean"]))/np.asarray(data["scale"])
        logits=logits_for(feature,np.asarray(data["weight"]),np.asarray(data["bias"]),data["supportedIndices"])
        scores=softmax(logits/data["scoreCalibration"]["temperature"])
        classes=data["classes"]; order=np.argsort(-scores[0])[:3]
        threshold=data["abstention"]["threshold"]; margin=data["abstention"]["margin"]
        accepted=bool(accept_mask(scores,classes,threshold,margin)[0]); reasons=[]
        if classes[order[0]]["classId"]=="other_fungi": reasons.append("other_or_unsupported_fungus")
        if scores[0,order[0]]<threshold: reasons.append("insufficient_model_score")
        if scores[0,order[0]]-scores[0,order[1]]<margin: reasons.append("competing_guesses_too_close")
        return {"schema":"mycoscout.group-guess.v1","evidenceVersion":data["evidenceVersion"],
            "guesses":[{"classId":classes[k]["classId"],"groupId":classes[k]["groupId"],"label":classes[k]["label"],"modelScore":float(scores[0,k])} for k in order],
            "abstained":not accepted,"abstentionReasons":reasons,"suggestedGroupId":classes[order[0]]["groupId"] if accepted else "unknown",
            "scoreCalibration":{"method":"validation-temperature","empirical":True,"guaranteed":False},
            "reviewPolicy":{"scoreThreshold":threshold,"marginThreshold":margin,"targetValidationAccuracy":data["abstention"]["targetValidationAccuracy"],
                            "validationTargetMet":data["abstention"]["targetMet"]},
            "humanVerificationRequired":True,"speciesConfirmed":False,"edibility":"not assessed","salesApproval":False,"flightGateEnabled":False,
            "processingMilliseconds":round((time.perf_counter()-started)*1000,2)}


def decode_image(encoded, mime_type=None):
    if not isinstance(encoded,str) or len(encoded)>((MAX_IMAGE_BYTES+2)//3)*4: raise ValueError("Image must be base64 of at most 8 MiB")
    try: payload=base64.b64decode(encoded,validate=True)
    except (ValueError,binascii.Error) as exc: raise ValueError("Invalid base64 image") from exc
    if not payload or len(payload)>MAX_IMAGE_BYTES: raise ValueError("Image must be 1 byte to 8 MiB")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.width*source.height>20_000_000 or min(source.size)<16: raise ValueError("Image must be 16px minimum and at most 20 MP")
            formats={"JPEG":"image/jpeg","PNG":"image/png","WEBP":"image/webp"}
            if source.format not in formats or mime_type is not None and mime_type!=formats[source.format]: raise ValueError("Unsupported image format or MIME mismatch")
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError,Image.DecompressionBombError) as exc: raise ValueError("Invalid image") from exc


def worker(checkpoint):
    predictor=Predictor(checkpoint)
    print("MycoScout group worker ready (local inference; human verification required)",file=sys.stderr,flush=True)
    while True:
        line=sys.stdin.buffer.readline(MAX_REQUEST_LINE)
        if not line: break
        request_id=None
        try:
            if len(line)>=MAX_REQUEST_LINE or not line.endswith(b"\n"): raise ValueError("Request line exceeds limit or is unterminated")
            request=json.loads(line)
            if not isinstance(request,dict): raise ValueError("Request must be an object")
            supplied_id=request.get("id")
            if not isinstance(supplied_id,str) or len(supplied_id)>128: raise ValueError("Request ID must be a string of at most 128 characters")
            request_id=supplied_id
            image=decode_image(request.get("imageBase64"),request.get("mimeType"))
            result={"id":request_id,"status":"ok","result":predictor.predict(image)}
        except (ValueError,KeyError,TypeError,OSError) as exc:
            result={"id":request_id,"status":"error","error":{"code":"INVALID_REQUEST","message":str(exc)[:300]}}
        print(json.dumps(result,allow_nan=False),flush=True)
        if len(line)>=MAX_REQUEST_LINE: break


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    trainer=sub.add_parser("train"); trainer.add_argument("--manifest",type=Path,default=ROOT/"data/public-fungi-v3/manifest.json")
    trainer.add_argument("--output",type=Path,default=ROOT/"experiments/results/groups-v1")
    trainer.add_argument("--checkpoint",type=Path,default=ROOT/"vision/checkpoints/groups-v1.json")
    trainer.add_argument("--weights",type=Path,default=ROOT/"models/newfeaturecache/resnet18-f37072fd.pth")
    trainer.add_argument("--min-train-photos",type=int,default=25)
    trainer.add_argument("--min-validation-photos",type=int,default=5)
    trainer.add_argument("--target-validation-accuracy",type=float,default=.8)
    service=sub.add_parser("worker"); service.add_argument("--checkpoint",type=Path,default=ROOT/"vision/checkpoints/groups-v1.json")
    infer=sub.add_parser("predict"); infer.add_argument("image",type=Path); infer.add_argument("--checkpoint",type=Path,default=ROOT/"vision/checkpoints/groups-v1.json")
    args=parser.parse_args()
    if args.command=="train": train(args.manifest,args.output,args.checkpoint,args.weights,args.min_train_photos,args.min_validation_photos,args.target_validation_accuracy)
    elif args.command=="worker": worker(args.checkpoint)
    else:
        image=decode_image(base64.b64encode(args.image.read_bytes()).decode("ascii"))
        print(json.dumps(Predictor(args.checkpoint).predict(image),indent=2))


if __name__=="__main__": main()
