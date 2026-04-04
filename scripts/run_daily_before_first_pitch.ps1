param(
  [string]$OutputRoot = $(if ($env:MLBV1_OUTPUT_ROOT) { $env:MLBV1_OUTPUT_ROOT } else { Join-Path (Split-Path -Parent $PSScriptRoot) "artifacts" }),
  [string]$TriggerUrl = "",
  [string]$HealthUrl = "",
  [string]$ResourceGroup = "",
  [string]$ContainerAppName = "",
  [string]$OddsApiKey = "",
  [string]$TriggerApiKey = "",
  [switch]$SkipWait,
  [switch]$ForceTriggerIfNoGames
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$repoRoot = Split-Path -Parent $PSScriptRoot
$envFilePath = Join-Path $repoRoot ".env"

function Get-DotEnvValues {
  param([string]$DotEnvPath)

  $values = @{}
  if (!(Test-Path $DotEnvPath)) { return $values }

  Get-Content $DotEnvPath | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    if ($_ -match '^\s*([^=]+)=(.*)$') {
      $values[$matches[1].Trim()] = $matches[2].Trim().Trim('"').Trim("'")
    }
  }

  return $values
}

function Resolve-ConfigValue {
  param(
    [string]$Primary,
    [string[]]$Names,
    [string]$FallbackFile = ""
  )

  if (![string]::IsNullOrWhiteSpace($Primary)) { return $Primary }

  foreach ($name in $Names) {
    if ($script:DotEnvValues.ContainsKey($name) -and ![string]::IsNullOrWhiteSpace($script:DotEnvValues[$name])) {
      return [string]$script:DotEnvValues[$name]
    }

    $envValue = [Environment]::GetEnvironmentVariable($name, "Process")
    if (![string]::IsNullOrWhiteSpace($envValue)) { return $envValue }
  }

  if (![string]::IsNullOrWhiteSpace($FallbackFile) -and (Test-Path $FallbackFile)) {
    return (Get-Content $FallbackFile -Raw).Trim()
  }

  return ""
}

function Resolve-ContainerAppUrls {
  param(
    [string]$ResolvedTriggerUrl,
    [string]$ResolvedHealthUrl,
    [string]$ResolvedResourceGroup,
    [string]$ResolvedContainerAppName
  )

  if (![string]::IsNullOrWhiteSpace($ResolvedTriggerUrl) -and ![string]::IsNullOrWhiteSpace($ResolvedHealthUrl)) {
    return @{
      TriggerUrl = $ResolvedTriggerUrl
      HealthUrl = $ResolvedHealthUrl
    }
  }

  if ([string]::IsNullOrWhiteSpace($ResolvedResourceGroup) -or [string]::IsNullOrWhiteSpace($ResolvedContainerAppName)) {
    throw "Provide TriggerUrl/HealthUrl directly or set ACA_RESOURCE_GROUP + ACA_APP_NAME."
  }

  $azPath = Get-Command az -ErrorAction SilentlyContinue
  if (-not $azPath) {
    throw "Azure CLI is required to derive Container App URLs."
  }

  $fqdn = az containerapp show `
    --only-show-errors `
    --name $ResolvedContainerAppName `
    --resource-group $ResolvedResourceGroup `
    --query properties.configuration.ingress.fqdn `
    --output tsv 2>$null

  if ([string]::IsNullOrWhiteSpace($fqdn)) {
    throw "Could not resolve Container App ingress FQDN."
  }

  return @{
    TriggerUrl = if (![string]::IsNullOrWhiteSpace($ResolvedTriggerUrl)) { $ResolvedTriggerUrl } else { "https://$fqdn/trigger" }
    HealthUrl = if (![string]::IsNullOrWhiteSpace($ResolvedHealthUrl)) { $ResolvedHealthUrl } else { "https://$fqdn/health" }
  }
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

$script:DotEnvValues = Get-DotEnvValues -DotEnvPath $envFilePath

$resolvedResourceGroup = Resolve-ConfigValue -Primary $ResourceGroup -Names @("ACA_RESOURCE_GROUP", "AZURE_RESOURCE_GROUP")
$resolvedContainerAppName = Resolve-ConfigValue -Primary $ContainerAppName -Names @("ACA_APP_NAME", "AZURE_CONTAINER_APP")
$resolvedOddsApiKey = Resolve-ConfigValue -Primary $OddsApiKey -Names @("ODDS_API_KEY")
$resolvedTriggerApiKey = Resolve-ConfigValue -Primary $TriggerApiKey -Names @("TRIGGER_API_KEY") -FallbackFile (Join-Path $repoRoot ".trigger_api_key.txt")
$resolvedTriggerUrl = Resolve-ConfigValue -Primary $TriggerUrl -Names @("MLB_TRIGGER_URL")
$resolvedHealthUrl = Resolve-ConfigValue -Primary $HealthUrl -Names @("MLB_HEALTH_URL")
$resolvedUrls = Resolve-ContainerAppUrls `
  -ResolvedTriggerUrl $resolvedTriggerUrl `
  -ResolvedHealthUrl $resolvedHealthUrl `
  -ResolvedResourceGroup $resolvedResourceGroup `
  -ResolvedContainerAppName $resolvedContainerAppName

if ([string]::IsNullOrWhiteSpace($resolvedTriggerApiKey)) {
  throw "TRIGGER_API_KEY is missing (argument, .env, process env, or .trigger_api_key.txt)."
}

$runDate = Get-Date -Format "yyyy-MM-dd"
$runStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outputDir = Join-Path $OutputRoot $runDate
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$firstPitchUtc = Get-FirstPitchUtc -ApiKey $resolvedOddsApiKey
$nowUtc = (Get-Date).ToUniversalTime()
$targetUtc = if ($firstPitchUtc) { $firstPitchUtc.AddHours(-1) } else { $null }

$scheduleMeta = [ordered]@{
  generated_at_utc          = (Get-Date).ToUniversalTime().ToString("o")
  first_pitch_utc           = if ($firstPitchUtc) { $firstPitchUtc.ToString("o") } else { $null }
  trigger_target_utc        = if ($targetUtc) { $targetUtc.ToString("o") } else { $null }
  skip_wait                 = [bool]$SkipWait
  force_trigger_if_no_games = [bool]$ForceTriggerIfNoGames
  resource_group            = $resolvedResourceGroup
  container_app_name        = $resolvedContainerAppName
  trigger_url               = $resolvedUrls.TriggerUrl
  health_url                = $resolvedUrls.HealthUrl
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
  $healthBefore = Invoke-WebRequest -Uri $resolvedUrls.HealthUrl -Method Get -UseBasicParsing -TimeoutSec 60
  $healthBefore.Content | Out-File -FilePath $healthBeforePath -Encoding utf8
}
catch {
  $_ | Out-String | Out-File -FilePath $healthBeforePath -Encoding utf8
}

$headers = @{ "X-Trigger-Key" = $resolvedTriggerApiKey }
try {
  $triggerResp = Invoke-WebRequest -Uri $resolvedUrls.TriggerUrl -Method Post -Headers $headers -UseBasicParsing -TimeoutSec 120
  $triggerResp.Content | Out-File -FilePath $triggerPath -Encoding utf8
}
catch {
  $_ | Out-String | Out-File -FilePath $triggerPath -Encoding utf8
}

Start-Sleep -Seconds 15

try {
  $healthAfter = Invoke-WebRequest -Uri $resolvedUrls.HealthUrl -Method Get -UseBasicParsing -TimeoutSec 60
  $healthAfter.Content | Out-File -FilePath $healthAfterPath -Encoding utf8
}
catch {
  $_ | Out-String | Out-File -FilePath $healthAfterPath -Encoding utf8
}

$azPath = Get-Command az -ErrorAction SilentlyContinue
if ($azPath -and ![string]::IsNullOrWhiteSpace($resolvedContainerAppName) -and ![string]::IsNullOrWhiteSpace($resolvedResourceGroup)) {
  try {
    az containerapp logs show --only-show-errors -n $resolvedContainerAppName -g $resolvedResourceGroup --tail 200 | Out-File -FilePath $logsPath -Encoding utf8
  }
  catch {
    $_ | Out-String | Out-File -FilePath $logsPath -Encoding utf8
  }
}

Write-Output "Saved run artifacts to: $outputDir"
