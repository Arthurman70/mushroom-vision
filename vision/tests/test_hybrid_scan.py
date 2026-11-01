import unittest

from vision.hybrid_scan import HybridScanScheduler


class HybridScanTests(unittest.TestCase):
    def scan(self, scheduler, timestamp, seconds=.1):
        plan = scheduler.plan(timestamp)
        return plan, scheduler.commit(plan, seconds)

    def test_defaults_mix_fast_and_ground_with_every_third_detail_full(self):
        scheduler = HybridScanScheduler()
        actual = [self.scan(scheduler, timestamp)[0].profile for timestamp in range(0, 24, 2)]
        self.assertEqual(actual, ["fast", "fast", "fast", "ground-tiles", "fast", "fast",
                                  "ground-tiles", "fast", "fast", "full-tiles", "fast", "fast"])

    def test_overdue_details_discard_missed_windows_and_fast_floor_prevents_starvation(self):
        scheduler = HybridScanScheduler()
        self.scan(scheduler, 0)
        plan, metadata = self.scan(scheduler, 30, 28.)
        self.assertEqual(plan.profile, "fast")
        self.assertTrue(metadata["detailDeferredForFastFrames"])
        self.assertTrue(metadata["exceedsRequestedInterval"])
        plan, metadata = self.scan(scheduler, 60, 28.)
        self.assertEqual(plan.profile, "ground-tiles")
        self.assertEqual(metadata["skippedDetailWindows"], 11)
        self.assertEqual(metadata["nextDetailDueTimestampSeconds"], 65)
        self.assertEqual(metadata["queuedDetailScans"], 0)
        self.assertEqual(self.scan(scheduler, 90)[0].profile, "fast")
        self.assertEqual(self.scan(scheduler, 120)[0].profile, "fast")
        self.assertEqual(self.scan(scheduler, 150)[0].profile, "ground-tiles")

    def test_planning_without_success_does_not_consume_due_detail(self):
        scheduler = HybridScanScheduler()
        self.scan(scheduler, 0)
        self.scan(scheduler, 2)
        plan = scheduler.plan(6)
        self.assertEqual(plan.profile, "ground-tiles")
        retry = scheduler.plan(6)
        self.assertEqual(retry, plan)
        scheduler.commit(retry, .1)
        with self.assertRaises(ValueError):
            scheduler.commit(plan, .1)

    def test_repeated_paused_frame_does_not_satisfy_fast_floor(self):
        scheduler = HybridScanScheduler()
        self.scan(scheduler, 0)
        for _ in range(10):
            _, metadata = self.scan(scheduler, 0)
            self.assertEqual(metadata["fastFramesSinceDetail"], 1)
            self.assertFalse(metadata["sourceTimestampAdvanced"])
        self.assertEqual(self.scan(scheduler, 6)[0].profile, "fast")
        self.assertEqual(self.scan(scheduler, 6)[0].profile, "fast")
        self.assertEqual(self.scan(scheduler, 7)[0].profile, "ground-tiles")

    def test_backwards_seek_resets_only_when_success_commits(self):
        scheduler = HybridScanScheduler()
        self.scan(scheduler, 10)
        self.scan(scheduler, 12)
        self.scan(scheduler, 16)
        planned = scheduler.plan(1)
        self.assertTrue(planned.reset)
        self.assertEqual(scheduler.plan(20).detail_index, 1)
        metadata = scheduler.commit(planned, .1)
        self.assertTrue(metadata["clockReset"])
        self.assertEqual(metadata["nextDetailDueTimestampSeconds"], 6)
        self.assertEqual(metadata["detailSweepIndex"], 0)

    def test_configuration_and_timestamp_validation(self):
        for value in (True, float("nan"), float("inf"), -1, 0, 3601):
            with self.subTest(value=value), self.assertRaises(ValueError):
                HybridScanScheduler(interval_seconds=value)
        for key in ("full_frame_every", "min_fast_frames"):
            for value in (True, 0, 1.5, 1001):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    HybridScanScheduler(**{key: value})
        scheduler = HybridScanScheduler()
        for value in (True, None, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scheduler.plan(value)
        plan = scheduler.plan(0)
        with self.assertRaises(ValueError):
            HybridScanScheduler().commit(plan, 0)
        with self.assertRaises(ValueError):
            scheduler.commit(plan, float("nan"))
        scheduler.commit(plan, .1)


if __name__ == "__main__":
    unittest.main()
