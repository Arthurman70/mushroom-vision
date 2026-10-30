"""Checks against leakage, malformed image input, and accidental confident unknowns."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from vision.multiclass import validate_manifest,decode_image,accept_mask,choose_policy,softmax,DATA_SCHEMA
from vision.acquire_groups import assign_class


class GroupsChecks(unittest.TestCase):
    def test_prior_observer_cannot_cross_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); rows=[]
            for i,color in enumerate(("red","green")):
                photo=root/f"{i}.png"; Image.new("RGB",(20,20),color).save(photo)
                sha=hashlib.sha256(photo.read_bytes()).hexdigest()
                rows.append({"id":str(i),"path":photo.name,"sha256":sha,"normalizedPixelSha256":sha,"classId":"a","synthetic":False,
                    "split":("train","test")[i],"photoLicense":{"code":"cc0"},"groupId":"sameobserver","splitGroupId":"sameobserver",
                    "observerId":7,"observationId":i,"photoId":i})
            manifest=root/"manifest.json"; manifest.write_text(json.dumps({"schema":DATA_SCHEMA,"status":"frozen","classes":[{"classId":"a"}],"images":rows}))
            (root/"manifest-sha256.json").write_text(json.dumps({"sha256":hashlib.sha256(manifest.read_bytes()).hexdigest()}))
            with self.assertRaisesRegex(ValueError,"leakage"): validate_manifest(manifest)

    def test_mime_claim_must_match_pixels(self):
        data=io.BytesIO(); Image.new("RGB",(32,32)).save(data,format="PNG")
        encoded=base64.b64encode(data.getvalue()).decode()
        self.assertEqual(decode_image(encoded,"image/png").size,(32,32))
        with self.assertRaisesRegex(ValueError,"MIME"): decode_image(encoded,"image/jpeg")
        with self.assertRaises(ValueError): decode_image("http://example.org/photo.jpg")

    def test_unknown_is_always_abstained(self):
        classes=[{"classId":"other_fungi"},{"classId":"a"},{"classId":"b"}]
        self.assertFalse(accept_mask(np.array([[.999,.0005,.0005]]),classes,.5,.1)[0])

    def test_threshold_not_selected_without_enough_evidence(self):
        classes=[{"classId":"other_fungi"},{"classId":"a"},{"classId":"b"}]
        scores=np.tile([.01,.98,.01],(19,1))
        policy=choose_policy(scores,np.ones(19,dtype=int),classes)
        self.assertFalse(policy["targetMet"])
        self.assertFalse(accept_mask(scores,classes,policy["threshold"],policy["margin"]).any())

    def test_weak_taxonomy_never_silently_chooses_overlapping_class(self):
        queries=[{"taxon_id":1,"classId":"a"},{"taxon_id":2,"classId":"b"}]
        self.assertEqual(assign_class({3},queries),"other_fungi")
        with self.assertRaises(ValueError): assign_class({1,2},queries)

    def test_large_logits_have_finite_normalized_scores(self):
        output=softmax(np.array([[100000,100001,-100000.0]]))
        self.assertTrue(np.isfinite(output).all())
        self.assertAlmostEqual(float(output.sum()),1)


if __name__=="__main__": unittest.main()
