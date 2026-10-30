import copy
import unittest

import numpy as np

from vision.live_occupancy import OccupancyMapper


def sensor(stamp=100, noise=.005):
    camera={"calibrationId":"test-depth", "rectified":True,
            "intrinsics":{"fx":40,"fy":40,"cx":32,"cy":32},
            "cameraToMap4x4":np.eye(4).tolist(),"poseTimestampSeconds":stamp,
            "positionStdM":.001,"orientationStdRad":.00001}
    meta={"calibrationId":"test-depth","alignedTo":"color","method":"rgbd",
          "timestampSeconds":stamp,"scaleToMeters":1,"minRangeM":.1,"maxRangeM":5,"stdM":noise}
    return camera,meta


def at(volume, point):
    index=[int(np.floor((point[i]-volume["origin"][axis])/volume["resolutionM"])) for i,axis in enumerate(("x","y","z"))]
    return volume["voxels"].get(",".join(map(str,index)),{}).get("state","unknown")


class ObservedOccupancyTests(unittest.TestCase):
    def make(self, depth=None, stamp=100, mapper=None, noise=.005, revision="m1"):
        camera,meta=sensor(stamp,noise)
        mapper=mapper or OccupancyMapper("test",size=(16,16,16))
        volume=mapper.update(np.full((64,64),2.) if depth is None else depth,camera=camera,depth_meta=meta,timestamp_seconds=stamp,map_revision=revision)
        return mapper,volume

    def test_free_volume_requires_observed_depth_footprint_and_ends_at_surface(self):
        _,volume=self.make()
        self.assertEqual(at(volume,[.25,.25,1.25]),"free")
        self.assertEqual(at(volume,[.25,.25,2.25]),"occupied")
        self.assertEqual(at(volume,[.25,.25,3.25]),"unknown")
        self.assertEqual(at(volume,[.25,.25,-1.25]),"unknown")
        self.assertEqual(at(volume,[3.25,.25,1.25]),"unknown")

    def test_one_unknown_pixel_prevents_clearing_its_whole_footprint(self):
        depth=np.full((64,64),2.);depth[40,40]=np.nan
        _,volume=self.make(depth)
        self.assertNotEqual(at(volume,[.25,.25,1.25]),"free")

    def test_near_occluder_prevents_free_space_behind_it(self):
        depth=np.full((64,64),2.);depth[32:48,32:48]=.8
        _,volume=self.make(depth)
        self.assertNotEqual(at(volume,[.25,.25,1.25]),"free")

    def test_recent_occupied_evidence_wins_then_ages_out(self):
        mapper,first=self.make()
        _,second=self.make(np.full((64,64),3.5),stamp=100.5,mapper=mapper)
        self.assertEqual(at(second,[.25,.25,2.25]),"occupied")
        _,third=self.make(np.full((64,64),3.5),stamp=101.1,mapper=mapper)
        self.assertEqual(at(third,[.25,.25,2.25]),"free")

    def test_missing_depth_never_refreshes_old_cells(self):
        mapper,_=self.make()
        _,volume=self.make(np.full((64,64),np.nan),stamp=101.1,mapper=mapper)
        self.assertEqual(volume["voxels"],{})

    def test_revision_reset_discards_old_evidence(self):
        mapper,_=self.make()
        _,volume=self.make(np.full((64,64),np.nan),stamp=100.1,mapper=mapper,revision="m2")
        self.assertEqual(volume["voxels"],{});self.assertEqual(volume["mapRevision"],"m2")

    def test_increased_uncertainty_cannot_create_more_free_cells(self):
        low,_=self.make(noise=.005);high,_=self.make(noise=.1)
        self.assertLessEqual(high.last_report["freeCells"],low.last_report["freeCells"])
        camera,meta=sensor();camera["positionStdM"]=100
        mapper=OccupancyMapper("bad-pose",size=(16,16,16))
        volume=mapper.update(np.full((64,64),2.),camera=camera,depth_meta=meta,timestamp_seconds=100,map_revision="m1")
        self.assertEqual(volume["voxels"],{});self.assertTrue(mapper.last_report["uncertaintyBlocked"])

    def test_bounds_and_geometry_compatibility(self):
        _,volume=self.make()
        self.assertLessEqual(len(volume["voxels"]),16**3)
        self.assertEqual(volume["frame"],"map_enu");self.assertEqual(volume["unit"],"m")
        for key,cell in volume["voxels"].items():
            self.assertTrue(all(0<=v<16 for v in map(int,key.split(','))))
            self.assertEqual(cell["observedMs"],100000)
        with self.assertRaises(ValueError): OccupancyMapper("too-large",size=(128,128,128))

    def test_depth_pose_order_and_calibration_are_required(self):
        mapper,_=self.make();camera,meta=sensor()
        with self.assertRaises(ValueError): mapper.update(np.ones((64,64)),camera=camera,depth_meta=meta,timestamp_seconds=100,map_revision="m1")
        camera,meta=sensor(101);meta["timestampSeconds"]=100
        with self.assertRaisesRegex(ValueError,"synchronized"):
            mapper.update(np.ones((64,64)),camera=camera,depth_meta=meta,timestamp_seconds=101,map_revision="m1")
        camera,meta=sensor(101);meta["calibrationId"]="other"
        with self.assertRaisesRegex(ValueError,"matching calibration"):
            mapper.update(np.ones((64,64)),camera=camera,depth_meta=meta,timestamp_seconds=101,map_revision="m1")


if __name__=='__main__': unittest.main()
