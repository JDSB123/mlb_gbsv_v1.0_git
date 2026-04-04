<# 
.SYNOPSIS
    Configure a Teams Workflow webhook URL for the GBSV MLB container app.

.DESCRIPTION
    After creating a "Post to a channel when a webhook request is received" 
    Workflow in Teams, run this script with the URL to wire it into the 
    Azure Container App.

.PARAMETER WebhookUrl
    The webhook URL from the Teams Workflow (starts with https://).

.EXAMPLE
    .\setup_teams_webhook.ps1 -WebhookUrl "https://prod-XX.westus.logic.azure.com:443/workflows/..."
#>
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^https://')]
    [string]$WebhookUrl,

    [string]$ContainerAppName = 'mlb-gbsv-v1-az-aca',
    [string]$ResourceGroup    = 'mlb-gbsv-v1-az-rg',
    [string]$KeyVaultName     = 'mlb-gbsv-v1-az-kv'
)

$ErrorActionPreference = 'Stop'
$app  = $ContainerAppName
$rg   = $ResourceGroup
$kv   = $KeyVaultName

Write-Host "=== Setting Teams webhook URL ===" -ForegroundColor Cyan

# 1. Store in Key Vault
Write-Host "Storing webhook URL in Key Vault ($kv)..."
az keyvault secret set --vault-name $kv --name 'teams-webhook-url' --value $WebhookUrl --output none
if ($LASTEXITCODE -ne 0) { throw "Failed to store secret in Key Vault" }

# 2. Add container secret
Write-Host "Adding container secret..."
az containerapp secret set -n $app -g $rg --secrets "teams-webhook-url=keyvaultref:https://$kv.vault.azure.net/secrets/teams-webhook-url,identityref:system" --output none 2>$null
if ($LASTEXITCODE -ne 0) {
    # Fallback: set as plain secret
    Write-Host "KV ref failed, setting as direct secret..."
    az containerapp secret set -n $app -g $rg --secrets "teams-webhook-url=$WebhookUrl" --output none
}

# 3. Set environment variable
Write-Host "Setting TEAMS_WEBHOOK_URL env var..."
az containerapp update -n $app -g $rg --set-env-vars "TEAMS_WEBHOOK_URL=secretref:teams-webhook-url" --output none

Write-Host ""
Write-Host "Done! Teams webhook configured." -ForegroundColor Green
Write-Host "Test with: Invoke-RestMethod -Uri 'https://<YOUR-ACA-FQDN>/trigger' -Method POST -Headers @{'X-Trigger-Key'='<your-key>'}"
