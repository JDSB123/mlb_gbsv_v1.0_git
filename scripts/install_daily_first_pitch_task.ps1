param(
  [string]$TaskName = "MLB-Daily-FirstPitch-Trigger",
  [string]$RunAtLocalTime = "06:00",
  [string]$OutputRoot = "C:\Users\JDSB\OneDrive - Green Bier Capital\Early Stage Sport Ventures - Documents\MLB - Green Bier Sports"
)

$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $PSScriptRoot "run_daily_before_first_pitch.ps1"
if (!(Test-Path $scriptPath)) {
  throw "Missing script: $scriptPath"
}

$actionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -OutputRoot `"$OutputRoot`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $actionArgs
$trigger = New-ScheduledTaskTrigger -Daily -At $RunAtLocalTime

try {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
}
catch {
}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Description "Run MLB trigger one hour before first pitch and archive outputs." | Out-Null

Write-Output "Installed scheduled task: $TaskName"
Write-Output "Run time: $RunAtLocalTime local time"
Write-Output "Output root: $OutputRoot"
