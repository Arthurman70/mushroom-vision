$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $projectRoot 'runtime-media'
$version = 'v1.21.3'
$assetName = 'mediamtx_v1.21.3_windows_amd64.zip'
# GitHub release API asset digest, verified 2026-10-28.
$expectedSha256 = 'b2f5246b07cb72961df03b8d5e8cad43dc657bcf3f49a87bb031bdf22721af74'
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$archive = Join-Path $runtimeDir $assetName
if (-not (Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -Uri "https://github.com/bluenviron/mediamtx/releases/download/$version/$assetName" -OutFile $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedSha256) {
    throw 'MediaMTX archive checksum mismatch. The archive was not extracted or run.'
}
$destination = Join-Path $runtimeDir "mediamtx-$version"
Expand-Archive -LiteralPath $archive -DestinationPath $destination -Force
Write-Output "Verified MediaMTX $version installed within the project: $destination"
