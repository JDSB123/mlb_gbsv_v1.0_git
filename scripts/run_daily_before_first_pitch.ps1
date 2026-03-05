param(
  [string]$OutputRoot = "C:\Users\JDSB\OneDrive - Green Bier Capital\Early Stage Sport Ventures - Documents\MLB - Green Bier Sports",
  [string]$TriggerUrl = "https://mlb0951-aca.delightfulsea-71b6793e.centralus.azurecontainerapps.io/trigger",
  [string]$HealthUrl = "https://mlb0951-aca.delightfulsea-71b6793e.centralus.azurecontainerapps.io/health",
  [string]$OddsApiKey = "",
  [string]$TriggerApiKey = "",
  [switch]$SkipWait,
  [switch]$ForceTriggerIfNoGames
)

$ErrorActionPreference = "Stop"

function Set-EnvFromDotEnv {
  param([string]$DotEnvPath)
  if (!(Test-Path $DotEnvPath)) { return }

  Get-Content $DotEnvPath | ForEach-Object {
    if ($_ -match '^\s*#') { return }
    if ($_ -match '^\s*$') { return }
    if ($_ -match '^\s*([^=]+)=(.*)$') {
      $name = $matches[1].Trim()
      $value = $matches[2].Trim().Trim('"').Trim("'")
      if (![string]::IsNullOrWhiteSpace($name) -and [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, "Process"))) {
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
      }
    }
  }
}

function Resolve-SecretValue {
  param(
    [string]$Primary,
    [string]$EnvName,
    [string]$FallbackFile
  )

  if (![string]::IsNullOrWhiteSpace($Primary)) { return $Primary }

  $envValue = [Environment]::GetEnvironmentVariable($EnvName, "Process")
  if (![string]::IsNullOrWhiteSpace($envValue)) { return $envValue }

  if (Test-Path $FallbackFile) {
    return (Get-Content $FallbackFile -Raw).Trim()
  }

  return ""
}

function Get-FirstPitchUtc {
  param([string]$ApiKey)

  if ([string]::IsNullOrWhiteSpace($ApiKey)) {
    throw "ODDS_API_KEY is required to compute first pitch time."
  }

  $oddsUrl = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey=$ApiKey&regions=us&markets=h2h"
  $resp = Invoke-RestMethod -Method Get -Uri $oddsUrl -TimeoutSec 60

  if ($null -eq $resp -or $resp.Count -eq 0) {
    throw "No MLB games returned by Odds API."
  }

  $nowUtc = (Get-Date).ToUniversalTime()
  $cutoffUtc = $nowUtc.AddHours(36)

  $times = @()
  foreach ($game in $resp) {
    if ($null -eq $game.commence_time) { continue }
    $t = [datetime]::Parse($game.commence_time).ToUniversalTime()
    if ($t -ge $nowUtc.AddHours(-6) -and $t -le $cutoffUtc) {
      $times += $t
    }
  }

  if ($times.Count -eq 0) { return $null }

  return ($times | Sort-Object | Select-Object -First 1)
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-EnvFromDotEnv -DotEnvPath (Join-Path $repoRoot ".env")

$resolvedOddsApiKey = Resolve-SecretValue -Primary $OddsApiKey -EnvName "ODDS_API_KEY" -FallbackFile ""
$resolvedTriggerApiKey = Resolve-SecretValue -Primary $TriggerApiKey -EnvName "TRIGGER_API_KEY" -FallbackFile (Join-Path $repoRoot ".trigger_api_key.txt")

if ([string]::IsNullOrWhiteSpace($resolvedTriggerApiKey)) {
  throw "TRIGGER_API_KEY is missing (env or .trigger_api_key.txt)."
}

$runDate = Get-Date -Format "yyyy-MM-dd"
$runStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outputDir = Join-Path $OutputRoot $runDate
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$firstPitchUtc = Get-FirstPitchUtc -ApiKey $resolvedOddsApiKey
$nowUtc = (Get-Date).ToUniversalTime()
$targetUtc = if ($firstPitchUtc) { $firstPitchUtc.AddHours(-1) } else { $null }

$scheduleMeta = [ordered]@{
  generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
  first_pitch_utc = if ($firstPitchUtc) { $firstPitchUtc.ToString("o") } else { $null }
  trigger_target_utc = if ($targetUtc) { $targetUtc.ToString("o") } else { $null }
  skip_wait = [bool]$SkipWait
  force_trigger_if_no_games = [bool]$ForceTriggerIfNoGames
  trigger_url = $TriggerUrl
  health_url = $HealthUrl
}
$scheduleMeta | ConvertTo-Json -Depth 5 | Out-File -FilePath (Join-Path $outputDir "$runStamp-schedule.json") -Encoding utf8

$healthBeforePath = Join-Path $outputDir "$runStamp-health-before.json"
$triggerPath = Join-Path $outputDir "$runStamp-trigger-response.json"
$healthAfterPath = Join-Path $outputDir "$runStamp-health-after.json"
$logsPath = Join-Path $outputDir "$runStamp-containerapp-logs.txt"

if ($firstPitchUtc -and !$SkipWait -and $nowUtc -lt $targetUtc) {
  $sleepSeconds = [math]::Floor(($targetUtc - $nowUtc).TotalSeconds)
  if ($sleepSeconds -gt 0) {
    Start-Sleep -Seconds $sleepSeconds
  }
}

if (-not $firstPitchUtc -and -not $ForceTriggerIfNoGames) {
  '{"status":"no_games_found","message":"No MLB first pitch found in next 36 hours; skipping trigger."}' | Out-File -FilePath $triggerPath -Encoding utf8
  Write-Output "Saved run artifacts to: $outputDir"
  exit 0
}

try {
  $healthBefore = Invoke-WebRequest -Uri $HealthUrl -Method Get -UseBasicParsing -TimeoutSec 60
  $healthBefore.Content | Out-File -FilePath $healthBeforePath -Encoding utf8
}
catch {
  $_ | Out-String | Out-File -FilePath $healthBeforePath -Encoding utf8
}

$headers = @{ "X-Trigger-Key" = $resolvedTriggerApiKey }
try {
  $triggerResp = Invoke-WebRequest -Uri $TriggerUrl -Method Post -Headers $headers -UseBasicParsing -TimeoutSec 120
  $triggerResp.Content | Out-File -FilePath $triggerPath -Encoding utf8
}
catch {
  $_ | Out-String | Out-File -FilePath $triggerPath -Encoding utf8
}

Start-Sleep -Seconds 15

try {
  $healthAfter = Invoke-WebRequest -Uri $HealthUrl -Method Get -UseBasicParsing -TimeoutSec 60
  $healthAfter.Content | Out-File -FilePath $healthAfterPath -Encoding utf8
}
catch {
  $_ | Out-String | Out-File -FilePath $healthAfterPath -Encoding utf8
}

$azPath = Get-Command az -ErrorAction SilentlyContinue
if ($azPath) {
  try {
    az containerapp logs show -n mlb0951-aca -g mlb-prod-centralus --tail 200 | Out-File -FilePath $logsPath -Encoding utf8
  }
  catch {
    $_ | Out-String | Out-File -FilePath $logsPath -Encoding utf8
  }
}

Write-Output "Saved run artifacts to: $outputDir"
