# Local video streaming preview

The mission station can receive a WebRTC preview independently of detector
updates. This addon uses MediaMTX v1.21.3 and the already-downloaded real mushroom
recording to exercise that transport. It is **recorded video**, not a connected
drone camera, and has no flight-control connection.

From the project directory in Windows PowerShell:

```powershell
& .\scripts\install-video-preview.ps1
& .\scripts\start-video-preview.ps1
```

The install helper downloads only the official Windows AMD64 release into
`runtime-media`, verifies its pinned SHA-256, and extracts it within the project.
No administrator installation is used. FFmpeg must already be on PATH. The
launch helper runs both background processes with hidden windows, writes their
identities and logs under `runtime-media`, and refuses a duplicate running
preview. It verifies the extracted executable digest before launch.

Use this WHEP URL in the station's video connection control:

```text
http://127.0.0.1:8889/mushrooms/whep
```

The publisher reads
`data/archive-video-v1/originals/Freestonew-theHugeMushroom948.flv`, normalizes its
video to a seekable local MP4 with H264 baseline and no B frames, then loops that
recording. The old FLV cannot reliably be looped directly. The source and attribution remain in the
existing collection records; the preview creates no new training permissions.

Stop only this project's matching process identities with:

```powershell
& .\scripts\start-video-preview.ps1 -Stop
```

The configuration `deploy/mediamtx-local.yml` binds only 127.0.0.1:

| Transport | Port | Purpose |
| --- | --- | --- |
| RTSP TCP | 8554 | FFmpeg publishes the recorded stream |
| HTTP/WHEP TCP | 8889 | Browser negotiates WebRTC |
| ICE UDP | 8189 | Local WebRTC media |

Other listeners, including RTMP, HLS, SRT, MoQ, API, metrics, profiler and playback,
are disabled. There are no public ICE/STUN/TURN servers. Cross-origin WHEP access
is limited to the local station origins on 8787 and the local QA station on 8788.
Only loopback clients can publish/read the `mushrooms` path. This configuration
intentionally does not expose a LAN stream to the phone or headset. Remote
devices need a separately configured authenticated HTTPS media endpoint and
network deployment; changing the address to all interfaces is not sufficient.

The compressed preview and detector frames are separate channels. Unless the
sensor producer supplies matching capture timestamps, do not assume a box
belongs to the currently displayed video frame. The station continues to retain
native image/landmark evidence separately.

Verified on this workstation: official archive digest, successful H264 publisher,
RTSP frame decoding, WHEP OPTIONS 204 with the intended CORS origin, and all
running listeners bound to loopback. Browser playback is checked separately in
the station. No drone camera, headset, or physical flight has been tested by
starting this preview.

Official release:
[MediaMTX v1.21.3](https://github.com/bluenviron/mediamtx/releases/tag/v1.21.3).

- Release archive SHA-256: `b2f5246b07cb72961df03b8d5e8cad43dc657bcf3f49a87bb031bdf22721af74`
- Extracted executable SHA-256: `13a0bfaa852d83ce976c94342a1af130941bd79fac2c1e95a2175879c92304c0`

Do not include `runtime-media` binaries or runtime logs in application bundles;
ship the helper, local configuration and this guide instead. MediaMTX's license
is included in its downloaded release directory.
