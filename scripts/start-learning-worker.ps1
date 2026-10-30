$ErrorActionPreference = 'Stop'
$workerRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$workerData = Join-Path $workerRoot 'data\learning'
New-Item -ItemType Directory -Force -Path $workerData | Out-Null
$workerLock = $null
try {
    $workerLock = [System.IO.File]::Open((Join-Path $workerData 'supervisor.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
} catch [System.IO.IOException] { exit 0 }
try {
    $workerNode = (Get-Command node.exe -ErrorAction Stop).Source
    Set-Location -LiteralPath $workerRoot
    while (-not (Test-Path -LiteralPath (Join-Path $workerData 'worker-stop'))) {
        $workerUp = $false
        $health = $null
        try {
            $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8787/api/health' -TimeoutSec 5
            $workerUp = $true
        } catch {
            if ($_.Exception.Response) { throw 'The worker port is occupied or needs credentials; inspect the existing station before restarting.' }
        }
        if ($workerUp -and $health.application -ne 'MycoScout') { throw 'Another application is using the worker port.' }
        if ($workerUp) { Start-Sleep -Seconds 30; continue }
        $workerProcess = Start-Process -FilePath $workerNode -ArgumentList 'server.mjs' -WorkingDirectory $workerRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $workerData 'worker-stdout.log') -RedirectStandardError (Join-Path $workerData 'worker-stderr.log')
        $workerProcess.Id | Set-Content -LiteralPath (Join-Path $workerData 'worker.pid')
        $workerProcess.WaitForExit()
        Start-Sleep -Seconds 30
    }
} finally { if ($workerLock) { $workerLock.Dispose() } }
