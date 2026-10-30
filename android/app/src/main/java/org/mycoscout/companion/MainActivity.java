package org.mycoscout.companion;

import android.Manifest;
import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Insets;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.webkit.GeolocationPermissions;
import android.webkit.PermissionRequest;
import android.webkit.ServiceWorkerClient;
import android.webkit.ServiceWorkerController;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.view.WindowInsets;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/** Packaged HTTPS shell; no native JavaScript bridge or aircraft-control API. */
public final class MainActivity extends Activity {
    private static final String HOST = "appassets.androidplatform.net";
    private static final String HOME = "https://" + HOST + "/index.html";
    private static final int CAMERA_REQUEST = 20, LOCATION_REQUEST = 21, FILE_REQUEST = 22;
    private WebView web;
    private PermissionRequest pendingCamera;
    private GeolocationPermissions.Callback pendingLocation;
    private String pendingLocationOrigin;
    private ValueCallback<Uri[]> pendingFile;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        web = new WebView(this);
        web.setBackgroundColor(Color.rgb(8, 17, 15));
        web.setFitsSystemWindows(true);
        if (Build.VERSION.SDK_INT >= 30) {
            web.setOnApplyWindowInsetsListener((view, insets) -> {
                Insets bars = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                view.setPadding(bars.left, bars.top, bars.right, bars.bottom);
                return insets;
            });
        }
        setContentView(web);
        WebSettings settings = web.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(false);
        // User-selected document URIs need content access; no storage permission is requested.
        settings.setAllowContentAccess(true);
        settings.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        settings.setMediaPlaybackRequiresUserGesture(true);
        settings.setGeolocationEnabled(true);
        settings.setSupportMultipleWindows(false);
        settings.setJavaScriptCanOpenWindowsAutomatically(false);
        web.setWebViewClient(new WebViewClient() {
            @Override public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                return localAsset(request);
            }
            @Override public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                if (isLocal(request.getUrl())) return false;
                if (request.isForMainFrame() && request.hasGesture() && "https".equals(request.getUrl().getScheme())) {
                    try { startActivity(new Intent(Intent.ACTION_VIEW, request.getUrl())); }
                    catch (ActivityNotFoundException ignored) { }
                }
                return true;
            }
            // Default SSL handling cancels invalid certificates. No error bypass.
        });
        web.setWebChromeClient(new WebChromeClient() {
            @Override public void onPermissionRequest(PermissionRequest request) {
                runOnUiThread(() -> handleCameraRequest(request));
            }
            @Override public void onPermissionRequestCanceled(PermissionRequest request) {
                if (pendingCamera == request) pendingCamera = null;
            }
            @Override public void onGeolocationPermissionsShowPrompt(String origin, GeolocationPermissions.Callback callback) {
                if (!isLocal(Uri.parse(origin)) || !currentPageIsLocal() || pendingLocation != null) {
                    callback.invoke(origin, false, false);
                    return;
                }
                if (hasLocationPermission()) callback.invoke(origin, true, false);
                else {
                    pendingLocation = callback;
                    pendingLocationOrigin = origin;
                    requestPermissions(new String[]{Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION}, LOCATION_REQUEST);
                }
            }
            @Override public void onGeolocationPermissionsHidePrompt() {
                pendingLocation = null;
                pendingLocationOrigin = null;
            }
            @Override public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback, FileChooserParams params) {
                if (!currentPageIsLocal()) return false;
                if (pendingFile != null) pendingFile.onReceiveValue(null);
                pendingFile = callback;
                Intent pick = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                pick.addCategory(Intent.CATEGORY_OPENABLE);
                String[] accept = params.getAcceptTypes();
                boolean json = Arrays.asList(accept).contains("application/json") || Arrays.asList(accept).contains(".json");
                pick.setType(json ? "application/json" : "image/*");
                pick.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, params.getMode() == FileChooserParams.MODE_OPEN_MULTIPLE);
                try { startActivityForResult(pick, FILE_REQUEST); }
                catch (ActivityNotFoundException missingPicker) {
                    pendingFile.onReceiveValue(null);
                    pendingFile = null;
                }
                return true;
            }
        });
        ServiceWorkerController workers = ServiceWorkerController.getInstance();
        workers.getServiceWorkerWebSettings().setAllowFileAccess(false);
        workers.getServiceWorkerWebSettings().setAllowContentAccess(false);
        workers.setServiceWorkerClient(new ServiceWorkerClient() {
            @Override public WebResourceResponse shouldInterceptRequest(WebResourceRequest request) {
                return localAsset(request);
            }
        });
        web.loadUrl(HOME);
    }

    private static boolean isLocal(Uri uri) {
        return uri != null && "https".equals(uri.getScheme()) && HOST.equals(uri.getHost())
            && uri.getPort() == -1 && uri.getUserInfo() == null;
    }
    private boolean currentPageIsLocal() {
        return web != null && web.getUrl() != null && isLocal(Uri.parse(web.getUrl()));
    }
    private boolean hasLocationPermission() {
        return checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
            || checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED;
    }
    private void handleCameraRequest(PermissionRequest request) {
        if (!isLocal(request.getOrigin()) || !currentPageIsLocal() || pendingCamera != null
                || !Arrays.asList(request.getResources()).contains(PermissionRequest.RESOURCE_VIDEO_CAPTURE)) {
            request.deny();
            return;
        }
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            request.grant(new String[]{PermissionRequest.RESOURCE_VIDEO_CAPTURE});
        else {
            pendingCamera = request;
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_REQUEST);
        }
    }
    @Override public void onRequestPermissionsResult(int code, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(code, permissions, results);
        if (code == CAMERA_REQUEST && pendingCamera != null) {
            PermissionRequest request = pendingCamera;
            pendingCamera = null;
            if (currentPageIsLocal() && checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
                request.grant(new String[]{PermissionRequest.RESOURCE_VIDEO_CAPTURE});
            else request.deny();
        }
        if (code == LOCATION_REQUEST && pendingLocation != null) {
            pendingLocation.invoke(pendingLocationOrigin, currentPageIsLocal() && hasLocationPermission(), false);
            pendingLocation = null;
            pendingLocationOrigin = null;
        }
    }
    @Override protected void onActivityResult(int code, int result, Intent data) {
        super.onActivityResult(code, result, data);
        if (code == FILE_REQUEST && pendingFile != null) {
            pendingFile.onReceiveValue(currentPageIsLocal() ? WebChromeClient.FileChooserParams.parseResult(result, data) : null);
            pendingFile = null;
        }
    }
    private WebResourceResponse localAsset(WebResourceRequest request) {
        if (!isLocal(request.getUrl())) return null;
        if (!"GET".equals(request.getMethod())) return errorResponse(405, "Method Not Allowed");
        String path = request.getUrl().getPath();
        if (path == null || path.equals("/")) path = "/index.html";
        if (path.indexOf('\\') >= 0 || path.indexOf('\0') >= 0 || Arrays.asList(path.split("/")).contains(".."))
            return errorResponse(403, "Forbidden");
        try {
            InputStream body = getAssets().open("www" + path);
            Map<String, String> headers = new HashMap<>();
            headers.put("X-Content-Type-Options", "nosniff");
            headers.put("Cache-Control", "no-cache");
            return new WebResourceResponse(mime(path), "UTF-8", 200, "OK", headers, body);
        } catch (IOException missing) { return errorResponse(404, "Not Found"); }
    }
    private static WebResourceResponse errorResponse(int status, String reason) {
        return new WebResourceResponse("text/plain", "UTF-8", status, reason, Collections.emptyMap(),
            new ByteArrayInputStream(reason.getBytes(StandardCharsets.UTF_8)));
    }
    private static String mime(String path) {
        String p = path.toLowerCase(Locale.ROOT);
        if (p.endsWith(".html")) return "text/html";
        if (p.endsWith(".js") || p.endsWith(".mjs")) return "application/javascript";
        if (p.endsWith(".css")) return "text/css";
        if (p.endsWith(".json") || p.endsWith(".webmanifest")) return "application/json";
        if (p.endsWith(".svg")) return "image/svg+xml";
        if (p.endsWith(".png")) return "image/png";
        if (p.endsWith(".jpg") || p.endsWith(".jpeg")) return "image/jpeg";
        if (p.endsWith(".webp")) return "image/webp";
        if (p.endsWith(".ico")) return "image/x-icon";
        if (p.endsWith(".wasm")) return "application/wasm";
        if (p.endsWith(".woff2")) return "font/woff2";
        return "application/octet-stream";
    }
    @Override protected void onPause() { if (web != null) web.onPause(); super.onPause(); }
    @Override protected void onResume() { super.onResume(); if (web != null) web.onResume(); }
    @Override protected void onDestroy() {
        if (pendingCamera != null) pendingCamera.deny();
        if (pendingLocation != null) pendingLocation.invoke(pendingLocationOrigin, false, false);
        if (pendingFile != null) pendingFile.onReceiveValue(null);
        if (web != null) web.destroy();
        super.onDestroy();
    }
}
