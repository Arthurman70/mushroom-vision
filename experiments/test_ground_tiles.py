import unittest

from experiments.evaluate_ground_tiles import match_predictions,roi_relationship,score_record,summarize


def prediction(box,confidence=.8,source='original'):
    return {'bboxPixelsXYXY':box,'confidence':confidence,'sourcePass':source}


class GroundTileEvaluationTests(unittest.TestCase):
    def test_one_target_cannot_receive_duplicate_true_positive_credit(self):
        truth=[{'bbox':[10,10,20,20]}]
        result=match_predictions(truth,[prediction([10,10,20,20],.9),prediction([10,10,20,20],.8),prediction([70,70,80,80])])
        self.assertEqual(result['truePositives'],1)
        self.assertEqual(result['falseNegatives'],0)
        self.assertEqual(result['unmatchedPredictionIndexes'],[1,2])
        self.assertEqual(match_predictions(truth,[])['falseNegatives'],1)

    def test_unmatched_positive_alerts_are_separate_from_false_negative_controls(self):
        row={'truth':[{'bbox':[10,10,20,20]}],'referenceStatus':'visible','referenceAuthority':'human','referenceComplete':False,'confounders':[],'roiBoundsPixelsXYXY':[0,35,100,100]}
        result=score_record(row,[prediction([70,70,80,80])])
        self.assertEqual(result['unmatchedPositiveAlerts'],1)
        self.assertIsNone(result['negativeFalseAlerts'])
        self.assertEqual(result['targetROI'][0]['relation'],'outside')
        self.assertFalse(result['targetROI'][0]['matched'])

    def test_ai_negative_references_never_become_human_false_alarm_metrics(self):
        row={'truth':[],'referenceStatus':'absent','referenceAuthority':'ai','referenceComplete':True,'confounders':[{'kind':'hat','bboxPixelsXYXY':[10,10,20,20]}],'roiBoundsPixelsXYXY':[0,35,100,100]}
        result=score_record(row,[prediction([10,10,20,20])])
        self.assertIsNone(result['negativeFalseAlerts'])
        self.assertEqual(result['aiReferenceNegativeAlerts'],1)
        self.assertEqual(result['confounderHits'][0]['overlappingAlertIndexes'],[0])
        partial=score_record({**row,'referenceComplete':False},[prediction([10,10,20,20])])
        self.assertIsNone(partial['aiReferenceNegativeAlerts'])
        summary=summarize([{**row,'methods':{'original':{'seconds':1,'modelPasses':1,'metrics':result}}}])[0]
        self.assertEqual(summary['negativeFalseAlerts'],0)
        self.assertEqual(summary['aiReferenceNegativeAlerts'],1)
        self.assertEqual(summary['humanNegativeFrames'],0)

    def test_roi_does_not_hide_targets_that_need_context(self):
        bounds=[0,35,100,100]
        self.assertEqual(roi_relationship([10,10,20,35],bounds),'outside')
        self.assertEqual(roi_relationship([10,20,20,40],bounds),'straddles')
        self.assertEqual(roi_relationship([10,35,20,90],bounds),'inside')
        row={'truth':[{'bbox':[10,10,20,20]}],'referenceStatus':'visible','referenceAuthority':'human','referenceComplete':True,'confounders':[],'roiBoundsPixelsXYXY':bounds}
        result=score_record(row,[prediction([10,10,20,20],source='original')])
        self.assertEqual(result['targetROI'][0]['matchedBy'],'original')
        self.assertEqual(result['truePositives'],1)


if __name__=='__main__':unittest.main()
