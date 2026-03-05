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
  [string]$ResourceGroup = "mlb-gbsv-v1-az-rg"
)

$ErrorActionPreference = "Stop"

Write-Host "🔐 MLB Prediction Model - Azure Key Vault Setup" -ForegroundColor Cyan
Write-Host "================================================`n" -ForegroundColor Cyan

# Check if Key Vault exists
Write-Host "Checking Key Vault: $KeyVaultName..." -ForegroundColor Yellow
$kvExists = az keyvault show --name $KeyVaultName --resource-group $ResourceGroup 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Host "❌ Key Vault '$KeyVaultName' not found in resource group '$ResourceGroup'" -ForegroundColor Red
  Write-Host "   Run the Bicep deployment first: az deployment group create -g $ResourceGroup -f infra/main.bicep" -ForegroundColor Yellow
  exit 1
}
Write-Host "✅ Key Vault found`n" -ForegroundColor Green

# Load from .env file if it exists
$envFile = Join-Path $PSScriptRoot "..\\.env"
if (Test-Path $envFile) {
  Write-Host "Loading environment variables from .env file..." -ForegroundColor Yellow
  Get-Content $envFile | ForEach-Object {
    if ($_ -match '^([^#][^=]+)=(.*)$') {
      $key = $matches[1].Trim()
      $value = $matches[2].Trim().Trim('"').Trim("'")
      [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
  }
  Write-Host "✅ Environment variables loaded`n" -ForegroundColor Green
}

# Secret mapping: Key Vault secret name -> Environment variable name
$secrets = @{
  "odds-api-key"            = "ODDS_API_KEY"
  "bets-api-key"            = "BETS_API_KEY"
  "visual-crossing-api-key" = "VISUAL_CROSSING_API_KEY"
  "trigger-api-key"         = "TRIGGER_API_KEY"
  "discord-webhook-url"     = "DISCORD_WEBHOOK_URL"
  "smtp-password"           = "SMTP_PASSWORD"
  "action-network-password" = "ACTION_NETWORK_PASSWORD"
  "action-network-email"    = "ACTION_NETWORK_EMAIL"
}

$uploaded = 0
$skipped = 0
$failed = 0

foreach ($kvSecret in $secrets.Keys) {
  $envVar = $secrets[$kvSecret]
  $value = [Environment]::GetEnvironmentVariable($envVar, "Process")

  Write-Host "Processing: $kvSecret (from $envVar)..." -ForegroundColor Yellow

  if ([string]::IsNullOrWhiteSpace($value)) {
    Write-Host "  ⚠️  Skipped - No value found for $envVar" -ForegroundColor Yellow
    $skipped++
    continue
  }

  try {
    # Upload secret to Key Vault
    az keyvault secret set `
      --vault-name $KeyVaultName `
      --name $kvSecret `
      --value $value `
      --output none

    if ($LASTEXITCODE -eq 0) {
      Write-Host "  ✅ Uploaded successfully" -ForegroundColor Green
      $uploaded++
    }
    else {
      Write-Host "  ❌ Failed to upload" -ForegroundColor Red
      $failed++
    }
  }
  catch {
    Write-Host "  ❌ Error: $_" -ForegroundColor Red
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
  Write-Host "⚠️  WARNING: No secrets were uploaded!" -ForegroundColor Yellow
  Write-Host "   Make sure you have API keys in environment variables or .env file" -ForegroundColor Yellow
  exit 1
}

if ($failed -gt 0) {
  Write-Host "❌ Some secrets failed to upload. Check the errors above." -ForegroundColor Red
  exit 1
}

Write-Host "✅ Secrets setup complete!" -ForegroundColor Green
Write-Host "   Next step: Deploy the infrastructure with 'az deployment group create'" -ForegroundColor Cyan