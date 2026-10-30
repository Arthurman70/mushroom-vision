"""Reviewed imports must preserve existing holdouts, opt-in, and label provenance."""
import base64
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from vision.import_reviews import image_fingerprints, reconcile, stage_records, import_reviews
from vision.multiclass import DATA_SCHEMA
from vision.transfer import write_json, checksum, read_json


def staged(identity,dhash,group,partition="train",label="a",evaluation=False):
    return {"record":{"findId":identity,"partition":partition},"dhash":dhash,"sha256":identity,"pixel":identity,
            "tokens":{"source:"+group},"classId":label,"evaluationOnly":evaluation}


def original(identity,dhash,partition="test",label="a"):
    return {"id":identity,"sha256":identity,"normalizedPixelSha256":identity,"perceptualDifferenceHash":dhash,
            "split":partition,"classId":label,"splitGroupId":identity}


class ReviewImportChecks(unittest.TestCase):
    def test_reencoded_known_test_image_keeps_entire_session_held_out(self):
        known=original("original","0000000000000000")
        # Different bytes/pixel hash, same perceptual image; second image is a new view in its session.
        a=staged("reencoded","0000000000000001","session")
        b=staged("newview","ffffffffffffffff","session")
        accepted,excluded,linked=reconcile([a,b],[known])
        self.assertEqual(len(accepted),1); self.assertEqual(accepted[0]["split"],"test")
        self.assertEqual(linked[0]["inheritedPartition"],"test"); self.assertFalse(excluded)

    def test_bridge_between_frozen_partitions_is_quarantined(self):
        base=[original("train","0000000000000000","train"),original("test","ffffffffffffffff","test")]
        a=staged("a","0000000000000000","same"); b=staged("b","ffffffffffffffff","same")
        accepted,excluded,linked=reconcile([a,b],base)
        self.assertFalse(accepted); self.assertEqual(len(excluded),2)

    def test_declared_test_cannot_be_downgraded_by_known_training_photo(self):
        base=[original("knowntrain","0000000000000000","train")]
        known=staged("knowncopy","0000000000000000","same","test")
        new=staged("novel","ffffffffffffffff","same","test")
        accepted,excluded,linked=reconcile([known,new],base)
        self.assertFalse(accepted); self.assertEqual(len(excluded),2)
        self.assertIn("downgrade",excluded[0]["reason"])

    def test_evaluation_only_never_fits(self):
        accepted,excluded,_=reconcile([staged("a","0000000000000000","g",evaluation=True)],[])
        self.assertFalse(accepted); self.assertIn("evaluation-only",excluded[0]["reason"])

    def test_different_photo_from_known_observer_inherits_prior_split(self):
        base=original("known","0000000000000000","validation"); base["observerId"]=123
        new=staged("different","ffffffffffffffff","local"); new["declaredSourceIds"]={"observerId":"123"}
        accepted,excluded,linked=reconcile([new],[base])
        self.assertEqual(len(accepted),1); self.assertEqual(accepted[0]["split"],"validation")
        self.assertFalse(linked); self.assertFalse(excluded)

    def test_crop_from_test_parent_cannot_enter_training(self):
        parent_hash="a"*64
        base=original(parent_hash,"0000000000000000","test")
        crop=staged("crop","ffffffffffffffff","different-session","train")
        crop["parentImageSha256"]=parent_hash
        accepted,excluded,linked=reconcile([crop],[base])
        self.assertEqual(len(accepted),1); self.assertEqual(accepted[0]["split"],"test")
        self.assertFalse(excluded); self.assertFalse(linked)

    def test_unknown_parent_siblings_cannot_cross_partitions(self):
        first=staged("crop-a","0000000000000000","a","train")
        second=staged("crop-b","ffffffffffffffff","b","test")
        for row in (first,second): row["tokens"].add("sha:"+"b"*64)
        accepted,excluded,linked=reconcile([first,second],[])
        self.assertFalse(accepted); self.assertEqual(len(excluded),2)

    def test_coarse_label_and_species_text_never_supply_fine_class(self):
        export={"schema":"mycoscout.reviewed-training-export.v1","labelsFrom":"explicit_human_reviews_only", "records":[
            {"findId":"a","labels":{"groupId":"gilled","speciesReported":"Some species","labelSource":"human_review"}}]}
        rows,rejected=stage_records(export,Path.cwd(),{"a":{"groupId":"gilled"}})
        self.assertFalse(rows); self.assertIn("explicit",rejected[0]["reason"])

    def test_complete_import_freezes_copy_and_preserves_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); base=root/"base"; source=root/"finds"; (base/"images").mkdir(parents=True); (source/"photos").mkdir(parents=True)
            images=[Image.fromarray(np.random.default_rng(seed).integers(0,256,(64,64,3),dtype=np.uint8)) for seed in (10,22)]
            initial=base/"images/base.png"; images[0].save(initial); sha=checksum(initial); pix,dh=image_fingerprints(images[0])
            row={"id":"base","path":"images/base.png","sha256":sha,"originalSha256":sha,"normalizedPixelSha256":pix,
                "perceptualDifferenceHash":dh,"classId":"a","synthetic":False,"split":"test","photoLicense":{"code":"cc0"},
                "groupId":"old","splitGroupId":"old","observerId":1,"observationId":1,"photoId":1,"inherited":False}
            write_json(base/"manifest.json",{"schema":DATA_SCHEMA,"status":"frozen","title":"fixture","images":[row],"classes":[{"classId":"a","groupId":"gilled"}]})
            original_hash=checksum(base/"manifest.json"); write_json(base/"manifest-sha256.json",{"sha256":original_hash})
            new=source/"photos/new.png"; images[1].save(new); new_hash=checksum(new)
            record={"findId":"human","image":{"relativePath":"photos/new.png","sha256":new_hash,"mimeType":"image/png"},
                "labels":{"classId":"a","groupId":"gilled","labelSource":"human_review"},"sourceGroupId":"human-session","partition":"train",
                "rights":{"trainingOptIn":True,"license":"Local training allowed","attestedBy":"Photographer"},
                "review":{"action":"confirm_group","reviewerName":"Reviewer"},"provenance":{}}
            release=source/"release.json"; write_json(release,{"schema":"mycoscout.reviewed-training-export.v1","labelsFrom":"explicit_human_reviews_only","releaseId":"r1","records":[record]})
            report=import_reviews(base/"manifest.json",release,source,root/"new")
            self.assertEqual(report["accepted"],1); self.assertEqual(checksum(base/"manifest.json"),original_hash)
            new_rows=read_json(root/"new/manifest.json")["images"]
            self.assertEqual([r["split"] for r in new_rows],["test","train"])
            record["provenance"]={"sourceMetadata":{"parentImageSha256":sha,"pixelBox":[0,0,32,32]}}
            write_json(release,{"schema":"mycoscout.reviewed-training-export.v1","labelsFrom":"explicit_human_reviews_only","releaseId":"r2","records":[record]})
            crop_report=import_reviews(base/"manifest.json",release,source,root/"crop-version")
            self.assertEqual(crop_report["acceptedPartitions"],{"test":1})
            crop_row=read_json(root/"crop-version/manifest.json")["images"][-1]
            self.assertEqual(crop_row["reviewProvenance"]["sourceMetadata"]["parentImageSha256"],sha)
            with self.assertRaisesRegex(ValueError,"latest complete export"):
                import_reviews(root/"new/manifest.json",release,source,root/"another")


if __name__=="__main__": unittest.main()
