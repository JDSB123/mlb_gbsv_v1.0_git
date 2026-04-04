# Azure Key Vault Secrets Setup Script
# Upload all required API keys and credentials to Azure Key Vault
#
# Prerequisites:
# 1. Azure CLI installed and logged in: az login
# 2. Have all API keys ready (see .env or environment variables)

param(
  [Parameter(Mandatory = $false)]
  [string]$KeyVaultName = "mlb-gbsv-v1-az-kv",

  [Parameter(Mandatory = $false)]
  [string]$ResourceGroup = "mlb-gbsv-v1-az-rg",

  [Parameter(Mandatory = $false)]
  [string]$EnvFilePath = ""
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$repoRoot = Split-Path -Parent $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($EnvFilePath)) {
  $EnvFilePath = Join-Path $repoRoot ".env"
}

function Get-DotEnvValues {
  param([string]$DotEnvPath)

  $values = @{}
  if (!(Test-Path $DotEnvPath)) {
    return $values
  }

  Get-Content $DotEnvPath | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    if ($_ -match '^\s*([^=]+)=(.*)$') {
      $values[$matches[1].Trim()] = $matches[2].Trim().Trim('"').Trim("'")
    }
  }

  return $values
}

function Get-ConfigValue {
  param([string]$Name)

  if ($script:DotEnvValues.ContainsKey($Name) -and ![string]::IsNullOrWhiteSpace($script:DotEnvValues[$Name])) {
    return [string]$script:DotEnvValues[$Name]
  }

  return [Environment]::GetEnvironmentVariable($Name, "Process")
}

function Protect-CliValue {
  param([string]$Value)

  return '"' + $Value.Replace('"', '\"') + '"'
}

$script:DotEnvValues = Get-DotEnvValues -DotEnvPath $EnvFilePath

Write-Host "MLB Prediction Model - Azure Key Vault Setup" -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

# Check if Key Vault exists
Write-Host "Checking Key Vault: $KeyVaultName..." -ForegroundColor Yellow
$kvExists = az keyvault show --only-show-errors --name $KeyVaultName --resource-group $ResourceGroup
if ($LASTEXITCODE -ne 0) {
  Write-Host "[ERROR] Key Vault '$KeyVaultName' not found in resource group '$ResourceGroup'" -ForegroundColor Red
  Write-Host "   Run the Bicep deployment first: az deployment group create -g $ResourceGroup -f infra/main.bicep" -ForegroundColor Yellow
  exit 1
}
Write-Host "[OK] Key Vault found`n" -ForegroundColor Green

# Load from .env file if it exists
if (Test-Path $EnvFilePath) {
  Write-Host "Using repo environment file: $EnvFilePath`n" -ForegroundColor Gray
}

# Secret mapping: Key Vault secret name -> Environment variable name
$secrets = @{
  "odds-api-key"            = "ODDS_API_KEY"
  "bets-api-key"            = "BETS_API_KEY"
  "visual-crossing-api-key" = "VISUAL_CROSSING_API_KEY"
  "trigger-api-key"         = "TRIGGER_API_KEY"
  "discord-webhook-url"     = "DISCORD_WEBHOOK_URL"
  "teams-webhook-url"       = "TEAMS_WEBHOOK_URL"
  "smtp-password"           = "SMTP_PASSWORD"
  "action-network-password" = "ACTION_NETWORK_PASSWORD"
  "action-network-email"    = "ACTION_NETWORK_EMAIL"
}

$uploaded = 0
$skipped = 0
$failed = 0

foreach ($kvSecret in $secrets.Keys) {
  $envVar = $secrets[$kvSecret]
  $value = Get-ConfigValue -Name $envVar

  Write-Host "Processing: $kvSecret (from $envVar)..." -ForegroundColor Yellow

  if ([string]::IsNullOrWhiteSpace($value)) {
    Write-Host "  [WARN] Skipped - No value found for $envVar" -ForegroundColor Yellow
    $skipped++
    continue
  }

  try {
    # Upload secret to Key Vault
    az keyvault secret set `
      --only-show-errors `
      --vault-name $KeyVaultName `
      --name $kvSecret `
      --value (Protect-CliValue -Value $value) `
      --output none

    if ($LASTEXITCODE -eq 0) {
      Write-Host "  [OK] Uploaded successfully" -ForegroundColor Green
      $uploaded++
    }
    else {
      Write-Host "  [ERROR] Failed to upload" -ForegroundColor Red
      $failed++
    }
  }
  catch {
    Write-Host "  [ERROR] $_" -ForegroundColor Red
    $failed++
  }
}

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  Uploaded: $uploaded" -ForegroundColor Green
Write-Host "  Skipped:  $skipped" -ForegroundColor Yellow
Write-Host "  Failed:   $failed" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })
Write-Host "================================================`n" -ForegroundColor Cyan

if ($uploaded -eq 0) {
  Write-Host "[WARN] No secrets were uploaded!" -ForegroundColor Yellow
  Write-Host "   Make sure you have API keys in environment variables or .env file" -ForegroundColor Yellow
  exit 1
}

if ($failed -gt 0) {
  Write-Host "[ERROR] Some secrets failed to upload. Check the errors above." -ForegroundColor Red
  exit 1
}

Write-Host "[OK] Secrets setup complete!" -ForegroundColor Green
Write-Host "   Next step: Deploy the infrastructure with 'az deployment group create'" -ForegroundColor Cyan
