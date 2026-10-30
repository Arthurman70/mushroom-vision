$ErrorActionPreference = 'Stop'
$workerScript = (Resolve-Path (Join-Path $PSScriptRoot 'start-learning-worker.ps1')).Path
$workerTaskName = 'MycoScout-Gleaner-LearningWorker'
$workerIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$workerArguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $workerScript + '"'
$existingWorkerTask = Get-ScheduledTask -TaskName $workerTaskName -ErrorAction SilentlyContinue
if ($existingWorkerTask -and (@($existingWorkerTask.Actions).Count -ne 1 -or $existingWorkerTask.Actions[0].Arguments -ne $workerArguments)) {
    throw 'A different task already uses this name; it was left unchanged.'
}
$workerAction = New-ScheduledTaskAction -Execute (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') -Argument $workerArguments
$workerTrigger = New-ScheduledTaskTrigger -AtLogOn -User $workerIdentity
$workerSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
$workerPrincipal = New-ScheduledTaskPrincipal -UserId $workerIdentity -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $workerTaskName -Action $workerAction -Trigger $workerTrigger -Settings $workerSettings -Principal $workerPrincipal -Description 'Runs the local MycoScout review-to-training worker and publishes aggregate Gleaner progress while this user is signed in.' -Force | Out-Null
Start-ScheduledTask -TaskName $workerTaskName
Get-ScheduledTask -TaskName $workerTaskName | Select-Object TaskName, State
