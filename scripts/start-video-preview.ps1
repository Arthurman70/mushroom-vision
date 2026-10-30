param([switch]$Stop)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $projectRoot 'runtime-media'
$statePath = Join-Path $runtimeDir 'preview-processes.json'
$serverExe = Join-Path $runtimeDir 'mediamtx-v1.21.3\mediamtx.exe'
$config = Join-Path $projectRoot 'deploy\mediamtx-local.yml'
$recording = Join-Path $projectRoot 'data\archive-video-v1\originals\Freestonew-theHugeMushroom948.flv'
if ($Stop) {
    if (Test-Path -LiteralPath $statePath) {
        $saved = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        foreach ($entry in @($saved.publisher, $saved.server)) {
            $process = Get-Process -Id $entry.id -ErrorAction SilentlyContinue
            if ($process -and $process.Path -eq $entry.path -and $process.StartTime.ToUniversalTime().Ticks -eq ([datetime]$entry.startedAt).ToUniversalTime().Ticks) {
                Stop-Process -Id $process.Id
            }
        }
        Remove-Item -LiteralPath $statePath
    }
    Write-Output 'Stopped this project preview, if it was running.'
    exit
}
if (-not (Test-Path -LiteralPath $serverExe)) { throw 'Run scripts/install-video-preview.ps1 first.' }
if ((Get-FileHash -LiteralPath $serverExe -Algorithm SHA256).Hash.ToLowerInvariant() -ne '13a0bfaa852d83ce976c94342a1af130941bd79fac2c1e95a2175879c92304c0') { throw 'MediaMTX executable differs from the verified release. Reinstall it before starting.' }
if (-not (Test-Path -LiteralPath $recording)) { throw 'The recorded mushroom source clip is missing. No synthetic camera feed was substituted.' }
$ffmpeg = (Get-Command ffmpeg -ErrorAction Stop).Source
if (Test-Path -LiteralPath $statePath) {
    $saved = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $running = Get-Process -Id $saved.server.id -ErrorAction SilentlyContinue
    if ($running -and $running.Path -eq $saved.server.path -and $running.StartTime.ToUniversalTime().Ticks -eq ([datetime]$saved.server.startedAt).ToUniversalTime().Ticks) {
        throw 'Preview is already running. Use -Stop to stop it first.'
    }
}
$loopRecording = Join-Path $runtimeDir 'preview-recording.mp4'
# This old FLV is not reliably seekable when looped. Normalize its video stream
# once to a seekable local MP4; no invented frames or detection labels are added.
& $ffmpeg -hide_banner -loglevel error -y -i $recording -map '0:v:0' -an -sn -dn -c:v libx264 -preset ultrafast -profile:v baseline -pix_fmt yuv420p -bf 0 -g 30 -movflags '+faststart' $loopRecording
if ($LASTEXITCODE -ne 0) { throw 'Could not prepare the recorded source for looping.' }
$server = Start-Process -FilePath $serverExe -ArgumentList @('"' + $config + '"') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeDir 'mediamtx.stdout.log') -RedirectStandardError (Join-Path $runtimeDir 'mediamtx.stderr.log')
Start-Sleep -Milliseconds 600
if ($server.HasExited) { throw 'MediaMTX did not start. Read runtime-media/mediamtx.stderr.log.' }
try {
    # Re-encode the real recording to browser-compatible H264 with no B frames.
    $quotedRecording = '"' + $loopRecording + '"'
    $arguments = @('-hide_banner','-loglevel','warning','-re','-stream_loop','-1','-i',$quotedRecording,'-an','-c:v','copy','-f','rtsp','-rtsp_transport','tcp','rtsp://127.0.0.1:8554/mushrooms')
    $publisher = Start-Process -FilePath $ffmpeg -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeDir 'publisher.stdout.log') -RedirectStandardError (Join-Path $runtimeDir 'publisher.stderr.log')
    Start-Sleep -Milliseconds 800
    if ($publisher.HasExited) { throw 'Video publisher did not start. Read runtime-media/publisher.stderr.log.' }
    $state = @{ recordedVideoOnly=$true; whep='http://127.0.0.1:8889/mushrooms/whep'; server=@{id=$server.Id;path=$serverExe;startedAt=$server.StartTime.ToUniversalTime().ToString('o')}; publisher=@{id=$publisher.Id;path=$ffmpeg;startedAt=$publisher.StartTime.ToUniversalTime().ToString('o')} }
    $state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statePath -Encoding UTF8
    Write-Output 'Recorded mushroom video is streaming locally: http://127.0.0.1:8889/mushrooms/whep'
    Write-Output 'This is the saved source clip, not a connected drone camera. Use -Stop to stop both processes.'
} catch {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id }
    throw
}
