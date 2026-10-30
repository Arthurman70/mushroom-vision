# MycoScout Android companion

The Android app bundles the shared MycoScout interface and demonstration scene. It can run the demonstration without a server or internet. It is a phone companion; real observations, perception and drone telemetry require the separate backend and supported hardware.

Package: `org.mycoscout.companion`. Minimum Android: 8.0 (API 26). Intended phone: Nothing Phone (3a). The package has not been exercised on that physical phone in this workspace.

## Install the APK

The deliverable is `dist/mycoscout-phone-debug.apk` (the raw build output is `android/app/build/outputs/apk/debug/app-debug.apk`), a signed **debug/testing APK**, not a Play Store release. Enable USB debugging on the phone, connect it to your computer, approve the device connection, then run from the project directory:

```powershell
adb install -r dist/mycoscout-phone-debug.apk
adb shell am start -n org.mycoscout.companion/.MainActivity
```

Alternatively, copy the APK to the phone, open it in Files and permit installation from that source when Android asks. Camera and location permissions are requested only when a feature needs them. You can revoke them in Android Settings.

## Connect a real backend

Set the backend address and access token in the shared interface's connection controls. Use HTTPS with a certificate trusted by the phone. The app blocks cleartext HTTP and does not bypass invalid TLS certificates. A computer's localhost refers to that computer, not the phone.

The backend must permit the exact CORS origin `https://appassets.androidplatform.net` and the `Authorization` and `Content-Type` request headers. Camera, location and file selection requests are accepted only from the packaged local origin. There is no native JavaScript bridge. App data lives in private WebView storage; clearing storage or uninstalling removes it. Keep API tokens scoped to this deployment.

Camera access provides a live camera background where the shared UI supports it. It does not establish ARCore tracking, persistent anchors or accurate alignment with the forest. Those require device/sensor integration and field validation. Blob-based file downloads are not supported by the native shell. Its Capture preview action shows an inline photo, and Export in browser opens the configured HTTPS station in the system browser. Use the browser version to save survey JSON or close-up photographs. Detection and survey JSON imports use the Android document picker; the shell honors application/json and .json accept types.

## Quest 3

Open the same MycoScout web interface in Quest Browser over trusted HTTPS. Use the immersive-view control when the browser exposes WebXR. **Android WebView is not the Quest WebXR runtime**, and this companion APK does not provide immersive VR. This build does not produce a separate Quest-specific APK. The phone and headset should connect to the same backend and project.

A headset map or route is only as reliable as its calibrated sensor and terrain observations. The demo does not establish flight readiness.

## Rebuild

Install JDK 17 and Android SDK platform 36 / matching build tools. Put the SDK path in `android/local.properties` as `sdk.dir=...`; this machine-specific file is ignored. From the `android` directory:

```powershell
.\gradlew.bat --no-daemon assembleDebug
```

The build stages `../public` into the APK every run. Rebuild whenever shared assets change. All JavaScript and 3D vendor files needed for the offline demonstration must exist under public before packaging. No signing credentials from other projects are used.

Check installation, offline demo loading, camera denial/grant, approximate/precise location, backend TLS/CORS, JSON/image selection and landscape layout on the actual phone before field use. Desktop checks cannot establish those device behaviors.

## Verified build — October 28, 2026

- Output: `dist/mycoscout-phone-debug.apk`.
- Package: `org.mycoscout.companion`; version name `0.7.0`; version code `7`.
- Minimum SDK 26; target/compile SDK 36; JDK 17; Gradle 8.14.3; Android Gradle Plugin 8.11.1.
- `assembleDebug` succeeded. `lintDebug` passed with no errors; its three warnings concern intentionally enabled JavaScript, a newer Gradle version, and legacy backup-configuration guidance (backup is disabled).
- `apksigner verify --verbose --print-certs` succeeded, using APK Signature Scheme v2 with one Android Debug RSA-2048 signer. This is a test signature, not a production distribution signature.
- APK SHA-256: `97b28f94a0efb482e97d19e2c2dcdb14dd6949c8a0516285c3672a556dc421dc`.
- Signer certificate SHA-256: `f6ba1e4a4c4fef2fa64124b3118e699b8ae72779dcda08b079e234bbcd9113e0`.
- Packaged `assets/www/app.js` matches `public/app.js`: SHA-256 `b8f795254308fa208c54c026131459892ca74b1de4e977b2fecbea9fef575177`.
- All 34 bundled public assets match the shared source byte-for-byte. No USB-debugging device or emulator was available, so device installation/camera/permission behavior remains unverified.

On this Windows host the first Maven-provided AAPT2 daemon failed to start while command startup was delayed. The successful build used the already installed Android SDK Build Tools 35.0.0 AAPT2, without installing system components:

```powershell
.\gradlew.bat --offline --no-daemon "-Pandroid.aapt2FromMavenOverride=<YOUR_ANDROID_SDK>/build-tools/35.0.0/aapt2.exe" assembleDebug
```

Replace the placeholder with your SDK path. Offline mode requires dependencies already downloaded. The ordinary build command above remains the portable default.
Version 0.7 adds the video grading feed, per-frame human reviews, source timelines and mushroom tracks, plus video-frame comparisons in the Vision Workbench. Version 0.6 introduced uploaded images, method comparisons, annotations and saved experiments. Configure the HTTPS station to share captured photos and review labels. The classifier runs on the station, not in the Android package.

The v0.5 build required `JAVA_OPTS=-Djava.net.preferIPv4Stack=true` on this host after Gradle's local lock listener failed to bind. The offline build then succeeded without source changes.
