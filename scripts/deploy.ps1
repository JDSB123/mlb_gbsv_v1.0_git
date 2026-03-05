# Complete Azure Deployment Script
# Deploys MLB Prediction Model to Azure Container Apps
#
# Prerequisites:
# 1. Azure CLI installed: az --version
# 2. Logged into Azure: az login
# 3. Docker Desktop running

param(
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroup = "mlb-gbsv-v1-az-rg",

    [Parameter(Mandatory=$false)]
    [string]$Location = "centralus",

    [Parameter(Mandatory=$false)]
    [string]$NamePrefix = "mlb-gbsv-v1-az",

    [Parameter(Mandatory=$false)]
    [string]$ServicePrincipalObjectId = ""
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 MLB Prediction Model - Azure Deployment" -ForegroundColor Cyan
Write-Host "==========================================`n" -ForegroundColor Cyan

# Step 1: Check prerequisites
Write-Host "Step 1: Checking prerequisites..." -ForegroundColor Yellow

# Check Azure CLI
$azVersion = az --version 2>&1 | Select-String "azure-cli"
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Azure CLI not found. Install from: https://aka.ms/installazurecli" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ Azure CLI: $azVersion" -ForegroundColor Green

# Check Docker
docker --version 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker not found. Install Docker Desktop" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ Docker is running" -ForegroundColor Green

# Check Azure login
$account = az account show 2>&1 | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Not logged into Azure. Run: az login" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ Logged into Azure: $($account.name)" -ForegroundColor Green
Write-Host "     Subscription: $($account.name) ($($account.id))`n" -ForegroundColor Gray

# Step 2: Create resource group
Write-Host "Step 2: Creating resource group..." -ForegroundColor Yellow
$rgExists = az group show --name $ResourceGroup 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Creating resource group: $ResourceGroup in $Location..." -ForegroundColor Yellow
    az group create --name $ResourceGroup --location $Location --output none
    Write-Host "  ✅ Resource group created" -ForegroundColor Green
} else {
    Write-Host "  ℹ️  Resource group already exists" -ForegroundColor Gray
}

# Step 3: Deploy infrastructure (Bicep)
Write-Host "`nStep 3: Deploying infrastructure (Bicep)..." -ForegroundColor Yellow
$bicepFile = Join-Path $PSScriptRoot "..\infra\main.bicep"
if (!(Test-Path $bicepFile)) {
    Write-Host "❌ Bicep file not found: $bicepFile" -ForegroundColor Red
    exit 1
}

# Generate SQL admin password
$sqlPassword = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 16 | ForEach-Object {[char]$_})
$sqlPassword = "Sql$sqlPassword!"

$deployParams = @(
    "--resource-group", $ResourceGroup,
    "--template-file", $bicepFile,
    "--parameters", "location=$Location",
    "--parameters", "namePrefix=$NamePrefix",
    "--parameters", "sqlAdminPassword=$sqlPassword",
    "--parameters", "allowUnauthTrigger=false"
)

if (![string]::IsNullOrWhiteSpace($ServicePrincipalObjectId)) {
    $deployParams += "--parameters"
    $deployParams += "servicePrincipalObjectId=$ServicePrincipalObjectId"
}

$triggerApiKey = [Environment]::GetEnvironmentVariable("TRIGGER_API_KEY", "Process")
if (![string]::IsNullOrWhiteSpace($triggerApiKey)) {
    $deployParams += "--parameters"
    $deployParams += "triggerApiKey=$triggerApiKey"
} else {
    Write-Host "  ⚠️  TRIGGER_API_KEY not set; /trigger endpoint will remain disabled until configured." -ForegroundColor Yellow
}

Write-Host "  Deploying Azure resources (this may take 5-10 minutes)..." -ForegroundColor Yellow
$deployment = az deployment group create @deployParams --output json 2>&1 | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Bicep deployment failed. Check the error above." -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ Infrastructure deployed successfully`n" -ForegroundColor Green

# Extract resource names from deployment outputs
$acrName = ("$NamePrefix-acr").Replace("-", "")
$kvName = "$NamePrefix-kv"
$acaName = "$NamePrefix-aca"

# Step 4: Upload secrets to Key Vault
Write-Host "Step 4: Uploading secrets to Key Vault..." -ForegroundColor Yellow
$secretsScript = Join-Path $PSScriptRoot "setup_secrets.ps1"
if (Test-Path $secretsScript) {
    & $secretsScript -KeyVaultName $kvName -ResourceGroup $ResourceGroup
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️  Secrets upload had issues. Check output above." -ForegroundColor Yellow
        Write-Host "   You can manually upload secrets later with: .\scripts\setup_secrets.ps1" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⚠️  Secrets setup script not found. Upload secrets manually:" -ForegroundColor Yellow
    Write-Host "     az keyvault secret set --vault-name $kvName --name odds-api-key --value YOUR_KEY" -ForegroundColor Gray
}

# Step 5: Build and push Docker image
Write-Host "`nStep 5: Building and pushing Docker image..." -ForegroundColor Yellow

# Login to ACR
Write-Host "  Logging into Azure Container Registry..." -ForegroundColor Yellow
az acr login --name $acrName --output none
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to login to ACR" -ForegroundColor Red
    exit 1
}

# Get ACR login server
$acrServer = az acr show --name $acrName --query loginServer --output tsv
Write-Host "  ACR Login Server: $acrServer" -ForegroundColor Gray

# Build and push image
$imageName = "${acrServer}/mlb-predictor:latest"
$imageTag = "$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$imageTagged = "${acrServer}/mlb-predictor:$imageTag"

Write-Host "  Building Docker image..." -ForegroundColor Yellow
docker build -t $imageName -t $imageTagged . --platform linux/amd64
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed" -ForegroundColor Red
    exit 1
}

Write-Host "  Pushing Docker image to ACR..." -ForegroundColor Yellow
docker push $imageName
docker push $imageTagged
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker push failed" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ Docker image pushed: $imageName" -ForegroundColor Green

# Step 6: Update Container App
Write-Host "`nStep 6: Updating Container App..." -ForegroundColor Yellow
az containerapp update `
    --name $acaName `
    --resource-group $ResourceGroup `
    --image $imageName `
    --output none

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Container App update failed" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ Container App updated with new image`n" -ForegroundColor Green

# Step 7: Get endpoint and verify
Write-Host "Step 7: Verifying deployment..." -ForegroundColor Yellow
$fqdn = az containerapp show `
    --name $acaName `
    --resource-group $ResourceGroup `
    --query properties.configuration.ingress.fqdn `
    --output tsv

$healthUrl = "https://$fqdn/health"
Write-Host "  Health endpoint: $healthUrl" -ForegroundColor Gray

Write-Host "  Waiting 30 seconds for container to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

try {
    $response = Invoke-WebRequest -Uri $healthUrl -Method Get -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "  ✅ Health check passed!" -ForegroundColor Green
        Write-Host "     Response: $($response.Content)" -ForegroundColor Gray
    } else {
        Write-Host "  ⚠️  Health check returned status: $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ⚠️  Health check failed: $_" -ForegroundColor Yellow
    Write-Host "     The app may still be starting. Check logs with:" -ForegroundColor Yellow
    Write-Host "     az containerapp logs show -n $acaName -g $ResourceGroup --follow" -ForegroundColor Gray
}

Write-Host "`n===========================================" -ForegroundColor Cyan
Write-Host "🎉 Deployment Complete!" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Resources Created:" -ForegroundColor Cyan
Write-Host "  • Resource Group: $ResourceGroup" -ForegroundColor Gray
Write-Host "  • Container Registry: $acrName" -ForegroundColor Gray
Write-Host "  • Key Vault: $kvName" -ForegroundColor Gray
Write-Host "  • Container App: $acaName" -ForegroundColor Gray
Write-Host "`nEndpoints:" -ForegroundColor Cyan
Write-Host "  • Service URL: https://$fqdn" -ForegroundColor Gray
Write-Host "  • Health: $healthUrl" -ForegroundColor Gray
Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "  1. Test health endpoint: Invoke-WebRequest $healthUrl" -ForegroundColor Gray
Write-Host "  2. Trigger pipeline: Invoke-WebRequest https://$fqdn/trigger -Method POST" -ForegroundColor Gray
Write-Host "  3. View logs: az containerapp logs show -n $acaName -g $ResourceGroup --follow" -ForegroundColor Gray
Write-Host "  4. Monitor metrics in Azure Portal: Application Insights" -ForegroundColor Gray
Write-Host "==========================================`n" -ForegroundColor Cyan