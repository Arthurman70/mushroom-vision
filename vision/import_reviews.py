"""Create a new frozen corpus version from explicitly reviewed, opted-in photos.

Never mutates a benchmark or model. Near-duplicate matching is conservative;
quarantines require human curation rather than silently changing prior labels.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.parse
from PIL import Image
from .acquire_public import now, digest
from .multiclass import validate_manifest, decode_image, DATA_SCHEMA, MAX_IMAGE_BYTES
from .transfer import read_json, write_json, checksum, photo_path


def image_fingerprints(image):
    pixel=digest(image.resize((64,64),Image.Resampling.BILINEAR).tobytes())
    thumb=list(image.convert("L").resize((9,8),Image.Resampling.BILINEAR).tobytes())
    difference=sum((thumb[y*9+x]>thumb[y*9+x+1])<<(y*8+x) for y in range(8) for x in range(8))
    return pixel,f"{difference:016x}"


def stage_records(export,image_root,classes):
    if export.get("schema")!="mycoscout.reviewed-training-export.v1" or export.get("labelsFrom")!="explicit_human_reviews_only":
        raise ValueError("Expected an explicit-human-review export")
    if not isinstance(export.get("records"),list) or len(export["records"])>5000:
        raise ValueError("Review import supports at most 5000 records per release")
    image_root=Path(image_root).resolve(); staged=[]; rejected=[]
    for record in export["records"]:
        identity=record.get("findId")
        try:
            if not isinstance(identity,str) or not identity or len(identity)>200: raise ValueError("invalid find ID")
            labels=record.get("labels",{}); rights=record.get("rights",{}); review=record.get("review",{})
            cid=labels.get("classId")
            if labels.get("labelSource")!="human_review" or cid not in classes: raise ValueError("missing explicit supported class label; species text and AI guesses are not labels")
            if labels.get("groupId")!=classes[cid]["groupId"]: raise ValueError("class and coarse group disagree")
            if rights.get("trainingOptIn") is not True or not isinstance(rights.get("license"),str) or not rights["license"].strip() or not rights.get("attestedBy"):
                raise ValueError("local-training rights and opt-in are required")
            if review.get("action") not in {"confirm_group","correct_species"} or not review.get("reviewerName"):
                raise ValueError("explicit human review required")
            if record.get("partition") not in {"train","validation","test"} or not record.get("sourceGroupId"):
                raise ValueError("partition and source group are required")
            image_info=record.get("image",{}); relative=image_info.get("relativePath")
            if not isinstance(relative,str) or not relative.startswith("photos/"): raise ValueError("photo path must be relative to finds image root")
            path=(image_root/relative).resolve()
            if not path.is_relative_to(image_root/"photos"): raise ValueError("photo path escapes finds photo directory")
            if path.stat().st_size>MAX_IMAGE_BYTES: raise ValueError("photo exceeds worker size limit")
            payload=path.read_bytes(); sha=digest(payload)
            if sha!=image_info.get("sha256"): raise ValueError("photo hash differs from export")
            image=decode_image(base64.b64encode(payload).decode("ascii"),image_info.get("mimeType"))
            pixel,perceptual=image_fingerprints(image)
            provenance=record.get("provenance",{})
            tokens={"source:"+str(record["sourceGroupId"])}
            for field in ("sessionId","groupId"):
                if provenance.get(field): tokens.add(field+":"+str(provenance[field]))
            metadata=provenance.get("sourceMetadata") or {}
            # Declared source identifiers only connect groups conservatively; never supply labels.
            if not isinstance(metadata,dict): raise ValueError("source metadata must be an object")
            parent_hash=metadata.get("parentImageSha256")
            if parent_hash is not None:
                if not isinstance(parent_hash,str) or not re.fullmatch(r"[0-9a-fA-F]{64}",parent_hash):
                    raise ValueError("parentImageSha256 must be a SHA-256 hex digest")
                parent_hash=parent_hash.lower()
                # Also links siblings whose parent is not in the corpus, and a parent exported alongside its crops.
                tokens.add("sha:"+parent_hash)
            source_ids={}
            for output_key,keys in (("observerId",("sourceObserverId","observerId")),("observationId",("sourceObservationId","observationId")),("photoId",("sourcePhotoId","photoId"))):
                for key in keys:
                    value=metadata.get(key)
                    if value is not None and str(value).isdigit(): source_ids[output_key]=str(value); break
            url=urllib.parse.urlparse(str(metadata.get("sourceUrl","")))
            if url.scheme=="https" and url.hostname in {"inaturalist.org","www.inaturalist.org"}:
                parts=url.path.strip("/").split("/")
                if len(parts)==2 and parts[0]=="observations" and parts[1].isdigit(): source_ids["observationId"]=parts[1]
            for key,value in source_ids.items(): tokens.add("declared-inat-"+key+":"+value)
            tokens.add("sha:"+sha); tokens.add("pixel:"+pixel)
            staged.append({"record":record,"sourcePath":path,"sha256":sha,"pixel":pixel,"dhash":perceptual,"tokens":tokens,
                           "width":image.width,"height":image.height,"bytes":len(payload),"classId":cid,
                           "evaluationOnly":provenance.get("evaluationOnly") is True,"declaredSourceIds":source_ids,
                           "parentImageSha256":parent_hash})
        except (ValueError,TypeError,KeyError,OSError) as exc:
            rejected.append({"findId":identity,"reason":str(exc)})
    return staged,rejected


def reconcile(staged,base_rows):
    parents=list(range(len(staged)))
    def find(i):
        while parents[i]!=i: parents[i]=parents[parents[i]]; i=parents[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b: parents[max(a,b)]=min(a,b)
    tokens={}; known_observers=defaultdict(list)
    for i,row in enumerate(staged):
        for token in row["tokens"]:
            if token in tokens: union(i,tokens[token])
            tokens[token]=i
        row["knownMatches"]=[]; row["priorGroupMatches"]=[]
        for original in base_rows:
            distance=(int(row["dhash"],16)^int(original["perceptualDifferenceHash"],16)).bit_count()
            is_duplicate=row["sha256"]==original["sha256"] or row["pixel"]==original["normalizedPixelSha256"] or distance<=2
            same_source=any(str(original.get(field))==value for field,value in row.get("declaredSourceIds",{}).items())
            same_parent=row.get("parentImageSha256") is not None and row["parentImageSha256"] in {original["sha256"],original.get("originalSha256")}
            if is_duplicate:
                row["knownMatches"].append(original)
            if is_duplicate or same_source or same_parent:
                row["priorGroupMatches"].append(original)
                known_observers[original["splitGroupId"]].append(i)
        for j,previous in enumerate(staged[:i]):
            if (int(row["dhash"],16)^int(previous["dhash"],16)).bit_count()<=2: union(i,j)
    for indices in known_observers.values():
        for i in indices[1:]: union(indices[0],i)
    components=defaultdict(list)
    for i,row in enumerate(staged): components[find(i)].append(row)
    accepted=[]; excluded=[]; linked=[]
    for component in components.values():
        prior={m["split"] for row in component for m in row["priorGroupMatches"]}
        declared={row["record"]["partition"] for row in component}
        evaluation=any(row["evaluationOnly"] for row in component)
        conflicts=[]
        if len(prior)>1: conflicts.append("component connects frozen corpus partitions")
        if len(declared)>1: conflicts.append("review source component has conflicting declared partitions")
        target=next(iter(prior)) if len(prior)==1 else next(iter(declared)) if len(declared)==1 else None
        ordering={"train":0,"validation":1,"test":2}
        if target is not None and len(declared)==1 and ordering[target]<ordering[next(iter(declared))]:
            conflicts.append("would downgrade a declared held-out review partition")
        if evaluation and target!="test": conflicts.append("evaluation-only photo conflicts with non-test partition")
        if any(m["classId"]!=row["classId"] for row in component for m in row["knownMatches"]):
            conflicts.append("human label conflicts with known or near-duplicate corpus photo; frozen labels were not overwritten")
        for i,row in enumerate(component):
            for other in component[:i]:
                if (int(row["dhash"],16)^int(other["dhash"],16)).bit_count()<=2 and row["classId"]!=other["classId"]:
                    conflicts.append("near-duplicate review labels conflict")
        if conflicts:
            excluded.extend({"findId":r["record"]["findId"],"reason":"; ".join(sorted(set(conflicts))),"status":"quarantine"} for r in component)
            continue
        component_id="review-group-"+digest("\n".join(sorted(t for r in component for t in r["tokens"])).encode())[:24]
        prior_components={m["splitGroupId"] for r in component for m in r["priorGroupMatches"]}
        # Inherited partition stays fixed. Re-encodings are not additional training examples.
        seen_hashes=set(); seen_pixels=set()
        for row in component:
            if row["knownMatches"]:
                linked.append({"findId":row["record"]["findId"],"status":"known-photo-excluded","inheritedPartition":target,
                               "matchedIds":[m["id"] for m in row["knownMatches"]],"declaredPartition":row["record"]["partition"]})
                continue
            if row["sha256"] in seen_hashes or row["pixel"] in seen_pixels:
                excluded.append({"findId":row["record"]["findId"],"reason":"duplicate reviewed image","status":"excluded"}); continue
            seen_hashes.add(row["sha256"]); seen_pixels.add(row["pixel"])
            row["split"]=target; row["componentId"]=component_id
            row["linkedPriorComponents"]=sorted(prior_components)
            accepted.append(row)
    return accepted,excluded,linked


def import_reviews(base_manifest,export_path,image_root,output):
    base_manifest=Path(base_manifest).resolve(); output=Path(output).resolve(); source_root=base_manifest.parent
    if output==source_root or output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError("Imported version must be separate from frozen corpus")
    if (output/"manifest.json").exists(): raise ValueError("Output version is already frozen")
    data=validate_manifest(base_manifest); export=read_json(export_path)
    if data.get("reviewImport"):
        raise ValueError("Rebuild from the original frozen corpus and latest complete export so revoked permissions and corrected reviews are not retained")
    classes={c["classId"]:c for c in data["classes"]}
    staged,rejected=stage_records(export,image_root,classes)
    accepted,excluded,linked=reconcile(staged,data["images"])
    rows=json.loads(json.dumps(data["images"]))
    (output/"images").mkdir(parents=True,exist_ok=True)
    for row in rows:
        destination=(output/row["path"]).resolve()
        if not destination.is_relative_to(output): raise ValueError("Invalid inherited media path")
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and checksum(destination)!=row["sha256"]: raise ValueError("Existing output image differs")
        if not destination.exists(): shutil.copyfile(photo_path(base_manifest,row),destination)
    for staged_row in accepted:
        record=staged_row["record"]; sha=staged_row["sha256"]; ext="png" if record["image"]["mimeType"]=="image/png" else "jpg"
        relative=f"images/review-{sha}.{ext}"; destination=output/relative
        if destination.exists() and checksum(destination)!=sha: raise ValueError("Existing reviewed output image differs")
        if not destination.exists(): shutil.copyfile(staged_row["sourcePath"],destination)
        # Source-group identifiers are not claims about photographer identity.
        group=staged_row["componentId"]
        rows.append({"id":"review-"+sha,"path":relative,"sha256":sha,"originalSha256":sha,
            "normalizedPixelSha256":staged_row["pixel"],"perceptualDifferenceHash":staged_row["dhash"],
            "width":staged_row["width"],"height":staged_row["height"],"bytes":staged_row["bytes"],
            "classId":staged_row["classId"],"weakLabel":staged_row["classId"],"labelSource":"human_review",
            "groupId":group,"splitGroupId":group,"observerId":group,"observationId":"review-"+record["findId"],"photoId":"review-"+sha,
            "split":staged_row["split"],"cohort":"reviewed_local","inherited":False,"synthetic":False,
            "photoLicense":{"code":"local-training-permission","attribution":record["rights"]["attestedBy"],"licenseText":record["rights"]["license"]},
            "rights":record["rights"],"review":record["review"],"reviewProvenance":record.get("provenance",{}),
            "reviewReleaseId":export["releaseId"],"sourceUrl":"local-find:"+record["findId"],
            "taxon":{"id":None,"name":None,"rank":None},"speciesReported":record["labels"].get("speciesReported"),
            "taxonCandidate":record["labels"].get("taxonCandidate"),
            "speciesProfessionallyVerified":False,"annotations":[],"annotationStatus":"human_class_review_only","distanceM":None,
            "linkedPriorComponents":staged_row["linkedPriorComponents"]})
    report={"schema":"mycoscout.review-import.v1","releaseId":export["releaseId"],"baseManifestSha256":checksum(base_manifest),
            "exportSha256":checksum(export_path),"accepted":len(accepted),"acceptedPartitions":dict(Counter(r["split"] for r in accepted)),
            "rejected":rejected,"quarantinedOrExcluded":excluded,"knownPhotos":linked,
            "policy":"explicit fine class and local-training permission; exact/normalized/near duplicates reconcile inherited partitions; no frozen labels overwritten",
            "limitations":["Perceptual matching can miss altered photos or over-quarantine similar-looking images.","Reviewer identity/rights are self-attested, not professionally verified."]}
    result={**data,"createdAt":now(),"title":data["title"]+" + reviewed local release","images":rows,
            "parentDataset":{"manifestSha256":checksum(base_manifest),"imagesInherited":len(data["images"]),"splitAssignmentsPreserved":True,
                             "note":"All inherited partitions and labels remain unchanged"},"reviewImport":report,
            "summary":{"images":len(rows),"newReviewedPhotos":len(accepted),"splitCounts":{p:dict(Counter(r["classId"] for r in rows if r["split"]==p)) for p in ("train","validation","test")}}}
    write_json(output/"manifest.json",result)
    write_json(output/"manifest-sha256.json",{"sha256":checksum(output/"manifest.json"),"createdAt":now()})
    write_json(output/"review-import.json",report)
    validate_manifest(output/"manifest.json")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,required=True); parser.add_argument("--export",type=Path,required=True)
    parser.add_argument("--image-root",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=import_reviews(args.base,args.export,args.image_root,args.output)
    print(json.dumps({"accepted":report["accepted"],"acceptedPartitions":report["acceptedPartitions"],"knownPhotos":len(report["knownPhotos"]),
                      "rejected":len(report["rejected"]),"quarantinedOrExcluded":len(report["quarantinedOrExcluded"]),"output":str(args.output)}))


if __name__=="__main__": main()
