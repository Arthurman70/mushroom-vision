# Deployment
## Included deliverables
- `dist/mycoscout-phone-debug.apk`: signed Android test installer; not a store release.
- `dist/mycoscout-quest-web.zip`: static WebXR/PWA assets for an HTTPS host. This is a web package, not an APK.
- `dist/mycoscout-development-bundle.zip`: application source, adapters, native project and instructions.

No public site, account, cloud service, certificate or domain has been created. The phone installer runs a demonstration immediately. Real shared data requires a configured station.

## Local desktop
Install Node.js 24+ and run `npm start`. Open http://127.0.0.1:8787. The server binds loopback only and persists data in `data/state.json`. Keep that file for your surveys. Do not replace it with demo data unintentionally.

## Phone
Install the signed debug APK as described in [ANDROID.md](ANDROID.md). It contains all 3D library assets and the offline demo. Its camera button displays a preview; capture creates an in-app still preview. Native blob downloads are unavailable, so use the HTTPS browser version for file export. The JSON file picker handles detector and survey files.

For browser AR, use Chrome with Google Play Services for AR. Device certification alone does not guarantee every optional WebXR feature; the UI checks runtime support. [WebXR requirements](https://developers.google.com/ar/develop/webxr/requirements)

## Quest 3
Host the contents of `mycoscout-quest-web.zip` over trusted HTTPS or serve the full station. Open that URL in Quest Browser. The **Enter AR** button shows passthrough with a miniature map; **Quest VR** shows the map in VR. Both controls are disabled when unsupported.

The view has no persistent shared spatial anchors and does not place mushrooms over their true locations. Do not follow the miniature map as a physical AR trail. No Meta developer account or sideloaded APK is required for the browser route. Physical-headset behavior is untested. [Meta WebXR](https://developers.meta.com/vr/documentation/web/webxr-overview/)

## Shared HTTPS station
Use a trusted network and a proper TLS reverse proxy. Remote camera/WebXR features do not work from an ordinary `http://192.168.x.x` origin. The native app rejects remote cleartext and invalid certificates.

The optional Docker Compose template expects Docker/Compose, a DNS name pointing to the host, and reachable ports 80/443 for certificate issuance. Review it before exposing a service:
1. Copy `deploy/.env.example` to `deploy/.env`.
2. Set `MYCOSCOUT_DOMAIN` to your DNS hostname.
3. Generate a random token: `node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"` and paste it into MYCOSCOUT_TOKEN.
4. Run `docker compose --env-file deploy/.env -f deploy/compose.yaml up --build -d`.
5. Open https://YOUR_DOMAIN on both devices. In Connection, enter that URL and the token. The token is kept in browser memory only, so re-enter it after restarting.

The station HTTP port is internal to the Compose network. Caddy obtains a certificate and proxies requests. The data volume persists surveys. These templates were inspected but **Docker/certificate issuance was not run here**.

If hosting the Quest static package at a different origin, set the station's `MYCOSCOUT_ALLOWED_ORIGIN` to that exact HTTPS origin, without a trailing slash. Android's exact local asset origin is already allowed. Never embed a token in a URL or static asset.

For an existing proxy, bind the station on a trusted internal interface using MYCOSCOUT_HOST and a token of at least 24 characters. The default loopback server rejects unrelated Host headers; configure the proxy to forward Host 127.0.0.1:8787, or use the container setup. Avoid exposing the unencrypted backend directly.

## Real sensor workflow
Run RTAB-Map on a supported onboard Linux computer with calibrated stereo/RGB-D and odometry. Use the provided snapshot importer. Its output marks ground traversability unverified; it is mapping input, not a walking-safety certificate.

Run the optional detector on a captured image; it outputs unlocated candidates. Group guesses and human classification are available in Review captures; automatic depth projection is not included. Use the API for already-qualified observations. See [INTEGRATIONS.md](INTEGRATIONS.md).

A true outdoor setup also requires camera calibration, synchronization, reliable visual-inertial odometry, a local 3D map, motion planning, flight-controller failsafe integration, and shared drone/phone/Quest transforms. These are remaining engineering work, not hidden configuration switches.

## Version 0.4 review station
The shared review inbox stores `data/finds/finds.sqlite` and its associated `photos` directory. Back up both together using the database instructions. The Docker image now uses Node24 for SQLite and includes the Michigan catalog; it does not contain Python, trained backbone weights or image datasets. Model inference requires separately configured compatible local runtime and checkpoints. See [human review](HUMAN_REVIEW.md) for Quest controller review, capture-folder ingestion and portable training exports.
