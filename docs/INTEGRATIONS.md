# Optional sensor integrations

Status checked October 26, 2026. These adapters add real input contracts to the MycoScout prototype. They do not implement or validate autonomous woodland flight.

## Implemented and verified locally

- Standard-library conversion of a ROS OccupancyGrid JSON snapshot to the application's map contract.
- A one-shot ROS 2 subscriber that reads a fresh map message and never publishes flight commands.
- An optional GroundingDINO image CLI using the published Hugging Face model through Transformers.
- Explicit snapshot replacement, request authentication, conservative unknown-cell handling and unlocated image-review results.
- 27 standard-library adapter tests passed, including CLI help with Python site packages disabled.
- The real JSON conversion output was accepted by the current core map validator with traversability still false.

The pretrained inference path has now run locally on downloaded weights and 17 group photos using a Windows CPU; see VISION_RUNTIME.md and VISION_RESULTS.md. The ROS subscription, drone sensors, GPU/Jetson, physical phone and headset remain untested. No model weights are bundled. There is no trained or validated mushroom species classifier.

## RTAB-Map snapshot workflow

[RTAB-Map ROS 2](https://github.com/introlab/rtabmap_ros/tree/ros2) supports RGB-D, stereo and lidar mapping. Its ROS wrapper is BSD-3-Clause; optional core-library dependencies can carry different licenses. Install and calibrate RTAB-Map on a Linux ROS 2 companion computer separately. The adapter does not install, start or configure that stack.

Use Python 3.10+ from the project directory. A saved OccupancyGrid must have ROS field names: header.frame_id, header.stamp, info.width, info.height, info.resolution, info.origin.position, info.origin.orientation and data.

An offline conversion can be tried without ROS:

    python adapters/rtabmap_snapshot.py --json adapters/examples/occupancy-grid.json --output occupancy-review.json

The included file is synthetic contract data, not a forest survey.

To capture one new message from a running ROS 2 system, first source its installed ROS environment:

    python3 adapters/rtabmap_snapshot.py --ros-topic /rtabmap/map --frame map --output forest-snapshot.json

The topic is an example; verify the actual nav_msgs/OccupancyGrid topic in your system. The subscriber uses best-effort, volatile QoS and waits for a new publication. It rejects messages older than five seconds or from another frame. ROS and sensor clocks must agree. A latched publisher that is not currently republishing will time out. Bag playback with simulated time requires an appropriately adapted clock setup; this CLI does not silently bypass timestamp checks.

By default this only saves a file. To replace the application's survey map, set MYCOSCOUT_TOKEN in the process environment if the server uses authentication, then explicitly opt into replacement:

    python3 adapters/rtabmap_snapshot.py --ros-topic /rtabmap/map --output next-snapshot.json --upload --replace-survey --server http://127.0.0.1:8787

This performs exactly one POST to /api/map. Replacement clears the survey's existing sightings. It is intentionally not a continuous map feed. A non-loopback server requires HTTPS and MYCOSCOUT_TOKEN; use a trusted certificate and the server's configured token. Tokens are passed through the Authorization header, never included in the URL. Redirects are refused.

Conversion rules:

| Source occupancy | Exported cell |
|---|---|
| -1, or an ambiguous probability below the occupied threshold | -1 unknown/blocked |
| Exactly 0 | 0 geometrically unoccupied |
| 65–100 by default | 1 occupied obstacle |

Code 1 in an imported map means a generic obstacle. It is not evidence of a tree. The occupied threshold is configurable; the free threshold is deliberately fixed at zero.

Maps must be 2 to 256 cells on each axis, resolution 0.1 to 100 metres, with a finite local origin. Cell order stays row-major, x varying fastest. Identity quaternions q and -q are accepted. Rotated grids, invalid quaternions, nonzero origin elevation and unexpected frame IDs are rejected rather than silently misaligned. Crop or conservatively resample upstream; never change resolution without actually resampling occupancy.

Every snapshot exports groundTraversabilityVerified: false. A 2D free cell cannot prove that the ground supports walking, that a cliff is absent, or that overhead airspace is clear. The core application blocks survey walking routes until ground traversability is explicitly verified. Do not set that flag simply because SLAM produced a map. Geometry alone does not classify mud, water or cliffs.

## GroundingDINO candidate workflow

The adapter uses [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny), whose model card declares Apache-2.0. [The original GroundingDINO project](https://github.com/IDEA-Research/GroundingDINO) also uses Apache-2.0. Preserve upstream attribution and license notices if redistributing model assets.

Create a separate Python 3.10 or 3.11 environment. The following example targets a Linux CPU workstation; the [official PyTorch version matrix](https://pytorch.org/get-started/previous-versions/) also lists matching Windows and CUDA installations. Jetson builds require a compatible NVIDIA-provided runtime and must be validated separately.

    python3.11 -m venv .venv-vision
    source .venv-vision/bin/activate
    python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r adapters/requirements-vision.txt

The supplied Transformers pin follows the [versioned GroundingDINO API documentation](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/grounding-dino). These direct dependency versions do not lock every transitive dependency. Validate an environment and capture a complete dependency lock before shipping it.

First explicitly allow the pretrained model download:

    python adapters/detect_fungi.py --image forest-photo.jpg --output candidate-review.json --cache-dir models --allow-model-download

Subsequent use defaults to the local cache, without permission to download:

    python adapters/detect_fungi.py --image next-photo.jpg --output next-review.json --cache-dir models

The default device is CPU. --device cuda requires a working compatible PyTorch CUDA environment. The initial download includes substantial model assets; no size or inference-speed claim is made. The local photograph is not uploaded. Network access is only used to fetch the selected model when explicitly enabled.

The default revision is main. For a repeatable deployment, use a verified model commit with --revision or provide a local snapshot directory with --model. The review output records the resolved commit when the model exposes it. The loader disables remote Python code and requires safetensors weights.

Outputs contain image-pixel bounding boxes, model phrases and uncalibrated detection scores. Every result is marked UNLOCATED, poseKnown:false, depthKnown:false and reviewed:false. There are no x/y/z world coordinates or claimed species. A custom --prompt changes the detector's object query; it does not create species certainty.

The JSON review file is not a /api/observations payload. Before creating a mapped observation, an additional validated pipeline must combine camera intrinsics, synchronized camera pose, usable depth and the correct map-frame transform. RGB-only detection does not recover metric location. Detection scores are not calibrated probabilities that a species identification is correct. Neither this detector nor this app determines edibility.

## Flight stack integration still required

A plausible implementation uses an onboard sensor/IMU/odometry stack, a 3D local obstacle map and a validated trajectory controller. Link the phone and headset to a mission service for supervision; keep collision avoidance, state estimation and loss-of-link behaviour onboard.

| Upstream project | State and intended role |
|---|---|
| [PX4 ROS2 control interface](https://docs.px4.io/main/en/ros2/px4_ros2_control_interface) | Useful mode health and failsafe integration. Current docs still mark parts experimental. Firmware, message definitions and interfaces must be pinned together. No flight-control code is included by these adapters. |
| [EGO-Swarm official ROS2 branch](https://github.com/ZJU-FAST-Lab/ego-planner-swarm/tree/ros2_version) | Research trajectory planner; GPLv3. ROS2 branch documents middleware lag and recommends CycloneDDS. Candidate for simulation and later hardware integration, not merged or flight-tested here. |
| [Aerostack2](https://github.com/aerostack2/aerostack2) | BSD-3-Clause ROS2 aerial robotics framework; documented development on Humble/Ubuntu 22.04. Possible orchestration layer, not included. |
| [PX4-Avoidance](https://github.com/PX4/PX4-Avoidance) | Archived August 1, 2024. Do not present it as a maintained turnkey forest-flight component. |
| [OpenCV](https://github.com/opencv/opencv) | Apache-2.0; useful for calibration, image preprocessing and marker pose estimation. Not bundled by this adapter. |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT; optional deployment engine after a compatible model export and hardware benchmarking. Installing it does not supply a trained mushroom model. |

The current imported grid is only a two-dimensional visualization/planning input. It is not a 3D flight collision map. Safe airborne clearance and pedestrian terrain are separate representations and require separate validation. Small branches, vegetation movement, poor light, depth gaps and localization loss must be covered by real hardware testing before live autonomous operation.

## Phone and headset alignment

[Google lists Nothing Phone (3a) as supporting ARCore and its Depth API](https://developers.google.com/ar/devices). This does not make its depth estimate a survey instrument. [WebXR on Android requires a supported device, Google Play Services for AR and a secure context](https://developers.google.com/ar/develop/webxr/requirements). A plain HTTP LAN address is not a secure context; use trusted HTTPS or development port forwarding.

Meta documents [WebXR mixed reality and anchors](https://developers.meta.com/vr/documentation/web/webxr-mixed-reality/) and [WebXR PWAs](https://developers.meta.com/vr/documentation/web/pwa-webxr/). Browser support still needs to be checked at runtime and tested on the actual Quest 3.

The drone map, phone AR session and Quest session each start with independent reference frames. Sharing coordinates over a network does not align them. A practical next step is a measured fiducial or surveyed correspondences, with a map-to-device transform, map version, calibration timestamp and uncertainty. Tracking resets invalidate alignment until checked again. Google's [Geospatial API relies on Street View-derived visual positioning](https://developers.google.com/ar/develop/geospatial); accurate forest coverage must not be assumed.

## Optional species research

[DanishFungiDataset](https://github.com/BohemianVRA/DanishFungiDataset) publishes [BVRA classifier checkpoints](https://huggingface.co/collections/BVRA/danish-fungi-2020). Its README explicitly restricts training data, metadata and models to non-commercial research despite a BSD repository license. These weights are not bundled. Performance on curated close-up observations does not establish performance on aerial forest imagery.

[FungiTastic](https://github.com/BohemianVRA/FungiTastic) provides fungal observation datasets and classification/segmentation benchmarks with BSD-3-Clause repository code. Rights for every downloaded image and model asset were not independently verified here. Do not infer unrestricted image/model redistribution from a code-license badge.

## Repeat local verification

    python -m unittest discover -s adapters/tests -v
    python -S adapters/rtabmap_snapshot.py --help
    python -S adapters/detect_fungi.py --help

These checks cover deterministic input/output contracts. They do not validate pretrained detections, ROS timing, environmental sensing, 3D collision avoidance, mushroom identification or physical AR alignment.

