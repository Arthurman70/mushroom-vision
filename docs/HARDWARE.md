# Hardware decision — checked October 26, 2026
## Finding
There is no verified low-cost used drone that autonomously explores dense woods, detects and identifies small fungi, reconstructs accurate ground hazards and delivers calibrated phone/Quest overlays. Do not purchase a cheap Wi-Fi toy expecting software to add the missing sensing and onboard compute.

The user's target is Nothing Phone (3a), Quest 3, United States, minimum cost but autonomous woodland flight ultimately required.

## Buying options
| Option | Price evidence | Appropriate use |
|---|---:|---|
| Used Ryze Tello | Approximately $100–150 indexed asking prices, stock varies | Indoor programming experiments only; 720p/5MP, no forest obstacle avoidance |
| Used DJI Mini 4 Pro + RC-N2 | Approximately $600–750 working-kit shopping range; indexed MPB listings near $709–729, category snapshots vary | Supervised reconnaissance/photography only; not a validated woodland autopilot |
| Holybro X500 v2 + onboard compute/depth | $1,337 for three core new components; approximately $1,800–2,500+ complete research budget | Programmable autonomy development, not a completed forest solution |
| ModalAI Starling 2 Max | $4,199.99 drone-only current listed starting price; about 60 business days stated lead | More integrated development platform; not proven forest exploration |

Used prices are asking-price evidence, not reserved offers. Recheck condition, included controller, charger, batteries, activation/account status, repairs and current stock. An MPB Mini 4 Pro at $254 was explicitly spares/repairs, not a working bargain.

Sources:
- [MPB Mini 4 Pro inventory](https://www.mpb.com/en-us/product/dji-mini-4-pro)
- [MPB used drone category](https://www.mpb.com/en-us/category/used-video-cameras/drones-and-action/drones-and-aerial-imaging)
- [Used Tello listings](https://www.ebay.com/b/DJI-Camera-Drones-DJI-Tello/179697/bn_7115171385)
- [Tello official manual](https://dl-cdn.ryzerobotics.com/downloads/Tello/20180404/Tello_User_Manual_V1.2_EN.pdf)

## Lowest-cost development baseline found
| Component | Listed price |
|---|---:|
| [Holybro X500 v2 development kit / Pixhawk 6C](https://holybro.com/products/px4-development-kit-x500-v2) | $609 |
| [Luxonis OAK-D S2 stereo depth + 12MP RGB](https://shop.luxonis.com/products/oak-d-s2) | $329 |
| [NVIDIA Jetson Orin Nano Super developer kit](https://marketplace.nvidia.com/en-us/enterprise/robotics-edge/jetson-orin-nano-super-developer-kit/) | $399, listed out of stock |
| Subtotal | $1,337 |

The older $249 Jetson marketing price differs from the checked current storefront. The complete-budget estimate includes batteries, charger, independent RC takeover, regulated power, storage, communications, mounts and Remote ID. It is an engineering estimate, not a vendor quote. Extra cameras/lidar, protective structure and tuning can increase cost considerably.

The X500 is a 500mm-wheelbase aircraft; the vendor's unloaded hover endurance is about 18 minutes. Payload reduces endurance. Its size makes it a poor fit for narrow woodland gaps. One forward depth camera leaves unobserved directions and cannot guarantee detection of thin branches, transparent water, low-texture surfaces or fast motion. This is a development baseline, not a recommendation to immediately fly it under canopy. [Holybro specifications](https://docs.holybro.com/drone-development-kit/px4-development-kit-x500v2)

## Consumer fallback
DJI Mini 4 Pro now supports **Android MSDK v5**. The RC-N2 works with an Android phone; the screen-equipped RC2 is not a substitute for a third-party phone application. Its camera provides 48MP/4K capture and up to 1080p transmitted view, but small distant mushrooms may occupy too few pixels and lack diagnostic details. Vision sensing depends on lighting and scene texture. The software in this repository does **not** include a DJI SDK driver or app key. [DJI developer support](https://developer.dji.com/), [Mini 4 Pro specifications](https://www.dji.com/mini-4-pro/specs), [FAQ](https://www.dji.com/mini-4-pro/faq)

## Integrated autonomy alternative
[Starling 2 Max](https://www.modalai.com/products/starling-2-max) includes VOXL2, PX4, visual-inertial navigation and imaging/tracking cameras. The checked product page disclosed degraded GNSS acquisition/accuracy on its current revision; it said visual navigation was unaffected. [VOXL Mapper](https://docs.modalai.com/voxl-mapper/) is documented as an indoor reference implementation. Its presence does not establish autonomous forest capability.

## United States operation
Autonomy does not remove visual-line-of-sight requirements. Recreational FPV/headset flight needs a nearby observer who can maintain direct sight. A drone hidden behind trees creates a practical and legal limit. Larger development aircraft need registration and generally Remote ID; verify the rules applicable to the actual aircraft and use. [FAA recreational flyers](https://www.faa.gov/uas/recreational_flyers), [FPV guidance AC91-57D](https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_91-57D.pdf), [FAA Remote ID](https://www.faa.gov/uas/getting_started/remote_id)

Nothing Phone (3a) is listed with Depth API support by [Google ARCore](https://developers.google.com/ar/devices). That does not mean a WebView wrapper automatically exposes native ARCore depth; this companion uses the camera for preview and WebXR in supported browsers for the tabletop map.
