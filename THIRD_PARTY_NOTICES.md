# Third-party notices
- **Three.js 0.180.0**, MIT: bundled `public/vendor/three.module.js` and `three.core.js`. Full upstream license is in `public/vendor/LICENSE`. [Source](https://github.com/mrdoob/three.js/tree/r180)
- **RTAB-Map ROS2**, BSD-3-Clause wrapper: optional external installation; no upstream source or binaries bundled. Core optional dependencies may have additional terms. [Source](https://github.com/introlab/rtabmap_ros/tree/ros2)
- **GroundingDINO / IDEA-Research grounding-dino-tiny**, Apache-2.0 project/model metadata: optional external model via Transformers. No weights bundled. Verify the pinned model revision/card when deploying. [Project](https://github.com/IDEA-Research/GroundingDINO), [model](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
- **Transformers**, Apache-2.0; **PyTorch**, BSD-style; **Pillow**, HPND-style: optional Python dependencies, not bundled. Exact direct pins are in adapters/requirements-vision.txt.
- **Gradle wrapper**, Apache-2.0: included for Android builds. Android SDK/Gradle plugin dependencies are downloaded separately under their own terms.
- **EGO-Swarm**, GPLv3, is referenced for future research only; no code integrated. Incorporating it later requires complying with its license.
- **DanishFungiDataset checkpoints/data** were evaluated as a possible species research source but are not bundled. Its repository README limits dataset/metadata/models to non-commercial research despite the code license.

The MycoScout integration code does not relicense these projects. Preserve upstream notices and review model/data licenses separately from library source licenses.

## Black-trumpet research additions

- **iNaturalist photos and metadata**: 1,174 local research photos, each retaining source attribution and its CC0, CC BY or CC BY-SA code. These are independently licensed third-party works; the software license does not replace their terms. Photos and source manifests are excluded from app/release archives. The local review identifies resized copies and overlays as modifications. See `docs/DATASET_V2.md` and the local per-photo manifest for provenance.
- **Torchvision ResNet18 IMAGENET1K_V1**: optional official pretrained backbone downloaded separately, SHA-256 verified and loaded with `weights_only=True`. Torchvision uses a BSD-style license; pretrained model/data provenance requires separate review. The small MycoScout JSON linear head is included with its training limits; neither the official backbone nor training photos are bundled. [Torchvision](https://github.com/pytorch/vision), [model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- **NVIDIA SegFormer ADE20K b0**: optional local **non-commercial research/evaluation only** model, revision `489d5cd81a0b59fab9b7ea758d3548ebe99677da`. Its NVIDIA use restriction applies independently of Transformers. It is not included in deployable archives and must be replaced or appropriately licensed before use beyond research/evaluation. The downloader saves the complete upstream license. [Model](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512), [license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE).

## Group-classification expansion (0.4)
The V3 corpus adds 2,076 licensed iNaturalist photos to the existing1,174. Each local manifest row retains its source attribution and Creative Commons license. Photos are not included in public web/APK/source packages. The new multiclass linear head uses the same official torchvision ResNet18 backbone as the V2 study. Michigan catalog text is a paraphrased evidence checklist with dated primary-source links; it does not redistribute complete MAMI study guides.

The review screenshots contain a public CC BY photograph attributed to Annika Lindqvist, from https://www.inaturalist.org/observations/407707147, https://creativecommons.org/licenses/by/4.0/. The original image is shown inside the application UI; the screenshot is a derived presentation and retains the credit in the interface.

## Optional continuous-video preview

[MediaMTX](https://github.com/bluenviron/mediamtx) is MIT licensed. The optional
installer verifies the official v1.21.3 release checksum and retains its license
in the separate runtime folder. Its executable and video recordings are excluded
from the source bundle. FFmpeg is supplied separately; its build license applies.

## Public review and pixel-test evidence

The October 31 acquisition added 193 iNaturalist photographs under their
individual CC0, CC BY or CC BY-SA terms. The public Gleaner lab retains original
credits, source links and license links. Saved pixel-test images are modified
presentations of those photographs and licensed video frames, with the
transformation and source shown in the inspector. These images are excluded
from the GitHub source package; public result JSON contains metadata and boxes,
not image bytes. Neither the project's MIT license nor a contributor annotation
changes the source image license. See [source records](docs/DATA_SOURCES.md) and
[pixel-test scope](docs/FULL_PIXEL_V5.md).
