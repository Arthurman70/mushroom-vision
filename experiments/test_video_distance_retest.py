"""Label-leakage and evidence-boundary regression checks for the distance retest."""
import unittest
from PIL import Image
from experiments.video_distance_retest import known_box_crop, protected_sources, summarize_classification
from experiments.video_distance_crop_context import context_crop, fitted_square
from experiments.evaluate_ai_candidate_distance import verify_no_overlap


class RetestTests(unittest.TestCase):
    def test_context_and_padding_keep_original_target_pixels_without_reconstruction(self):
        image = Image.new("RGB", (50, 50), "green")
        image.putpixel((0, 0), (7, 11, 19))
        crop = context_crop(image, [0, 0, 8, 20])
        self.assertEqual(crop.size, (25, 25))
        self.assertEqual(crop.getpixel((9, 3)), (7, 11, 19))
        self.assertEqual(crop.getpixel((0, 0)), (127, 127, 127))
        fitted = fitted_square(image.crop((0, 0, 8, 20)))
        self.assertEqual(fitted.size, (20, 20))
        self.assertEqual(fitted.getpixel((6, 0)), (7, 11, 19))

    def test_ai_candidate_fit_and_distance_test_are_disjoint(self):
        training = [{"sha256": "fit", "sourceGroupId": "fit-source"}]
        verify_no_overlap(training, [{"imageSha256": "test", "sourceGroupId": "test-source"}])
        for row in [{"imageSha256": "fit", "sourceGroupId": "test-source"}, {"imageSha256": "test", "sourceGroupId": "fit-source"}]:
            with self.assertRaisesRegex(ValueError, "overlap"):
                verify_no_overlap(training, [row])

    def test_subpixel_crop_never_passes_native_evidence_guard(self):
        crop, evidence = known_box_crop(Image.new("RGB", (100, 100)), [40.1, 50.1, 40.79, 50.85])
        self.assertEqual(crop.size, (1, 1))
        self.assertAlmostEqual(evidence["nativeTargetWidth"], .69)
        self.assertFalse(evidence["productionEligible"])
        self.assertFalse(evidence["localizationEvaluated"])
        self.assertTrue(evidence["knownTargetLocation"])
        _, boundary = known_box_crop(Image.new("RGB", (100, 100)), [1.1, 1.1, 33.099, 40.0])
        self.assertFalse(boundary["productionEligible"])
        _, eligible = known_box_crop(Image.new("RGB", (100, 100)), [1, 1, 33, 40])
        self.assertTrue(eligible["productionEligible"])

    def test_exact_image_and_creator_overlap_fail_before_any_fit_or_inference(self):
        refs = [{"imageSha256": "test-photo", "sourceGroupId": "held-out-creator"}]
        for row in [
            {"split": "train", "sha256": "test-photo"},
            {"split": "validation", "originalSha256": "test-photo"},
            {"split": "train", "sourceGroupId": "held-out-creator"},
            {"split": "validation", "splitGroupId": "held-out-creator"},
        ]:
            with self.assertRaisesRegex(ValueError, "overlaps"):
                protected_sources({"images": [row]}, refs)
        result = protected_sources({"images": [{"split": "test", "sha256": "test-photo"},
                                                {"split": "train", "sha256": "different", "groupId": "different"}]}, refs)
        self.assertEqual(result["trainingAndValidationRowsChecked"], 1)
        self.assertFalse(result["nearDuplicateGuarantee"])

    def test_category_agreement_remains_ai_reference_not_human_truth(self):
        correct = {"abstained": True, "guesses": [{"classId": "true_morels"}]}
        wrong = {"abstained": False, "guesses": [{"classId": "boletes"}, {"classId": "true_morels"}]}
        result = summarize_classification([{"crop": {"productionEligible": False},
                                            "classification": {"baseline": correct, "expanded": wrong}}])
        self.assertEqual(result["referenceAuthority"], "assistant")
        self.assertEqual(result["referenceClass"], "true_morels")
        self.assertFalse(result["humanSpeciesGroundTruth"])
        self.assertEqual(result["productionEligible"], 0)
        self.assertEqual(result["baseline"]["top1Agreement"], 1)
        self.assertEqual(result["baseline"]["retainedAgreement"], 0)
        self.assertEqual(result["expanded"]["top1Agreement"], 0)
        self.assertEqual(result["expanded"]["top3Agreement"], 1)
        self.assertEqual(result["expanded"]["retainedAgreement"], 0)


if __name__ == "__main__":
    unittest.main()
