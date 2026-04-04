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
    [string]$NamePrefix = "mlb-gbsv-$(Get-Random -Minimum 1000 -Maximum 9999)-az",

    [Parameter(Mandatory=$false)]
    [string]$ServicePrincipalObjectId = "",

    [Parameter(Mandatory=$false)]
    [string]$EnvFilePath = "",

    [Parameter(Mandatory=$false)]
    [switch]$UseLegacyResourceNames
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
    param(
        [string]$Name,
        [string]$Default = ""
    )

    if ($script:DotEnvValues.ContainsKey($Name) -and ![string]::IsNullOrWhiteSpace($script:DotEnvValues[$Name])) {
        return [string]$script:DotEnvValues[$Name]
    }

    $processValue = [Environment]::GetEnvironmentVariable($Name, "Process")
    if (![string]::IsNullOrWhiteSpace($processValue)) {
        return $processValue
    }

    return $Default
}

function Format-BicepStringParameter {
    param(
        [string]$Name,
        [string]$Value
    )

    $escapedValue = $Value.Replace('"', '\"')
    return "$Name=`"$escapedValue`""
}

function Format-BicepLiteralParameter {
    param(
        [string]$Name,
        [string]$Value
    )

    return "$Name=$Value"
}

$script:DotEnvValues = Get-DotEnvValues -DotEnvPath $EnvFilePath

function Get-LegacyResourceNameOverrides {
    param([string]$ResolvedNamePrefix)

    $compactPrefix = $ResolvedNamePrefix.Replace("-", "").ToLowerInvariant()
    return @{
        AcrName = "$($compactPrefix)acr"
        StorageAccountName = "$($compactPrefix)sto"
        KeyVaultName = "$ResolvedNamePrefix-kv"
        SqlServerName = "$($compactPrefix)sql"
    }
}

function Get-ExistingContainerAppEnvValue {
    param(
        [string]$ResolvedResourceGroup,
        [string]$ResolvedContainerAppName,
        [string]$EnvName
    )

    $result = az containerapp show `
        --only-show-errors `
        --name $ResolvedContainerAppName `
        --resource-group $ResolvedResourceGroup `
        --query "properties.template.containers[0].env[?name=='$EnvName'].value | [0]" `
        --output tsv 2>$null

    if ($LASTEXITCODE -ne 0) {
        return ""
    }

    return $result
}

function Get-ExistingKeyVaultSecretValue {
    param(
        [string]$ResolvedKeyVaultName,
        [string]$SecretName
    )

    $result = az keyvault secret show `
        --only-show-errors `
        --vault-name $ResolvedKeyVaultName `
        --name $SecretName `
        --query value `
        --output tsv 2>$null

    if ($LASTEXITCODE -ne 0) {
        return ""
    }

    return $result
}

function Get-ExistingContainerAppImage {
    param(
        [string]$ResolvedResourceGroup,
        [string]$ResolvedContainerAppName
    )

    $result = az containerapp show `
        --only-show-errors `
        --name $ResolvedContainerAppName `
        --resource-group $ResolvedResourceGroup `
        --query "properties.template.containers[0].image" `
        --output tsv 2>$null

    if ($LASTEXITCODE -ne 0) {
        return ""
    }

    return $result
}

function Get-ExistingContainerAppPrincipalId {
    param(
        [string]$ResolvedResourceGroup,
        [string]$ResolvedContainerAppName
    )

    $result = az containerapp show `
        --only-show-errors `
        --name $ResolvedContainerAppName `
        --resource-group $ResolvedResourceGroup `
        --query "identity.principalId" `
        --output tsv 2>$null

    if ($LASTEXITCODE -ne 0) {
        return ""
    }

    return $result
}

function Get-ExistingRoleAssignmentName {
    param(
        [string]$ResolvedResourceGroup,
        [string]$ResolvedKeyVaultName,
        [string]$PrincipalId
    )

    if ([string]::IsNullOrWhiteSpace($ResolvedKeyVaultName) -or [string]::IsNullOrWhiteSpace($PrincipalId)) {
        return ""
    }

    $kvId = az keyvault show `
        --only-show-errors `
        --name $ResolvedKeyVaultName `
        --resource-group $ResolvedResourceGroup `
        --query id `
        --output tsv 2>$null

    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($kvId)) {
        return ""
    }

    $result = az role assignment list `
        --only-show-errors `
        --scope $kvId `
        --assignee-object-id $PrincipalId `
        --role "Key Vault Secrets User" `
        --query "[0].name" `
        --output tsv 2>$null

    if ($LASTEXITCODE -ne 0) {
        return ""
    }

    return $result
}

Write-Host "MLB Prediction Model - Azure Deployment" -ForegroundColor Cyan
Write-Host "==========================================`n" -ForegroundColor Cyan
if (Test-Path $EnvFilePath) {
    Write-Host "Using repo environment file: $EnvFilePath`n" -ForegroundColor Gray
}

# Step 1: Check prerequisites
Write-Host "Step 1: Checking prerequisites..." -ForegroundColor Yellow

# Check Azure CLI
$azPath = Get-Command az -ErrorAction SilentlyContinue
if (-not $azPath) {
    Write-Host "[ERROR] Azure CLI not found. Install from: https://aka.ms/installazurecli" -ForegroundColor Red
    exit 1
}
$azVersionInfo = az version --only-show-errors | ConvertFrom-Json
$azVersion = $azVersionInfo.'azure-cli'
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($azVersion)) {
    Write-Host "[ERROR] Azure CLI is installed but version could not be determined." -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] Azure CLI: $azVersion" -ForegroundColor Green

# Check Docker (Skipped - utilizing Azure ACR Build instead)
# docker --version 2>&1 | Out-Null
# if ($LASTEXITCODE -ne 0) {
#     Write-Host "[ERROR] Docker not found. Install Docker Desktop" -ForegroundColor Red
#     exit 1
# }
# Write-Host "  [OK] Docker is running" -ForegroundColor Green

# Check Azure login
$account = az account show --only-show-errors | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Not logged into Azure. Run: az login" -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] Logged into Azure: $($account.name)" -ForegroundColor Green
Write-Host "     Subscription: $($account.name) ($($account.id))`n" -ForegroundColor Gray

# Step 2: Create resource group
Write-Host "Step 2: Creating resource group..." -ForegroundColor Yellow
$rgExists = az group show --only-show-errors --name $ResourceGroup
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Creating resource group: $ResourceGroup in $Location..." -ForegroundColor Yellow
    az group create --only-show-errors --name $ResourceGroup --location $Location --output none
    Write-Host "  [OK] Resource group created" -ForegroundColor Green
} else {
    Write-Host "  [INFO] Resource group already exists" -ForegroundColor Gray
}

# Step 3: Deploy infrastructure (Bicep)
Write-Host "`nStep 3: Deploying infrastructure (Bicep)..." -ForegroundColor Yellow
$bicepFile = Join-Path $PSScriptRoot "..\infra\main.bicep"
if (!(Test-Path $bicepFile)) {
    Write-Host "[ERROR] Bicep file not found: $bicepFile" -ForegroundColor Red
    exit 1
}

# Generate SQL admin password
$sqlPassword = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 16 | ForEach-Object {[char]$_})
$sqlPassword = "Sql$sqlPassword!"

$deployParams = @(
    "--resource-group", $ResourceGroup,
    "--template-file", $bicepFile,
    "--parameters", (Format-BicepStringParameter -Name "location" -Value $Location),
    "--parameters", (Format-BicepStringParameter -Name "namePrefix" -Value $NamePrefix),
    "--parameters", (Format-BicepStringParameter -Name "sqlAdminPassword" -Value $sqlPassword),
    "--parameters", (Format-BicepLiteralParameter -Name "enableRoleAssignments" -Value "true"),
    "--parameters", (Format-BicepLiteralParameter -Name "allowUnauthTrigger" -Value (Get-ConfigValue -Name 'ALLOW_UNAUTH_TRIGGER' -Default 'false')),
    "--parameters", (Format-BicepStringParameter -Name "trackingDbPath" -Value (Get-ConfigValue -Name 'TRACKING_DB_PATH' -Default 'artifacts/tracking.db')),
    "--parameters", (Format-BicepStringParameter -Name "slateTimezone" -Value (Get-ConfigValue -Name 'SLATE_TIMEZONE' -Default 'America/Chicago')),
    "--parameters", (Format-BicepLiteralParameter -Name "allowSyntheticFallback" -Value (Get-ConfigValue -Name 'ALLOW_SYNTHETIC_FALLBACK' -Default 'false')),
    "--parameters", (Format-BicepLiteralParameter -Name "liveContextDays" -Value (Get-ConfigValue -Name 'LIVE_CONTEXT_DAYS' -Default '120')),
    "--parameters", (Format-BicepStringParameter -Name "triggerMinIntervalSeconds" -Value (Get-ConfigValue -Name 'TRIGGER_MIN_INTERVAL_SECONDS' -Default '30'))
)

if ($UseLegacyResourceNames) {
    $legacyNames = Get-LegacyResourceNameOverrides -ResolvedNamePrefix $NamePrefix
    $deployParams += "--parameters"
    $deployParams += (Format-BicepStringParameter -Name "acrNameOverride" -Value $legacyNames.AcrName)
    $deployParams += "--parameters"
    $deployParams += (Format-BicepStringParameter -Name "storageAccountNameOverride" -Value $legacyNames.StorageAccountName)
    $deployParams += "--parameters"
    $deployParams += (Format-BicepStringParameter -Name "keyVaultNameOverride" -Value $legacyNames.KeyVaultName)
    $deployParams += "--parameters"
    $deployParams += (Format-BicepStringParameter -Name "sqlServerNameOverride" -Value $legacyNames.SqlServerName)
}

$existingAcaName = "$NamePrefix-aca"
$existingKvName = if ($UseLegacyResourceNames) { $legacyNames.KeyVaultName } else { "" }
$currentContainerImage = Get-ExistingContainerAppImage -ResolvedResourceGroup $ResourceGroup -ResolvedContainerAppName $existingAcaName
if (![string]::IsNullOrWhiteSpace($currentContainerImage)) {
    $deployParams += "--parameters"
    $deployParams += (Format-BicepStringParameter -Name "containerImage" -Value $currentContainerImage)
    Write-Host "  [INFO] Preserving current Container App image during infra update: $currentContainerImage" -ForegroundColor Gray
}

$existingAcaPrincipalId = Get-ExistingContainerAppPrincipalId -ResolvedResourceGroup $ResourceGroup -ResolvedContainerAppName $existingAcaName
if ($UseLegacyResourceNames -and ![string]::IsNullOrWhiteSpace($existingAcaPrincipalId)) {
    $existingAcaKvRoleAssignmentName = Get-ExistingRoleAssignmentName `
        -ResolvedResourceGroup $ResourceGroup `
        -ResolvedKeyVaultName $existingKvName `
        -PrincipalId $existingAcaPrincipalId

    if (![string]::IsNullOrWhiteSpace($existingAcaKvRoleAssignmentName)) {
        $deployParams += "--parameters"
        $deployParams += (Format-BicepStringParameter -Name "acaKvRoleAssignmentNameOverride" -Value $existingAcaKvRoleAssignmentName)
        Write-Host "  [INFO] Reusing existing Key Vault role assignment: $existingAcaKvRoleAssignmentName" -ForegroundColor Gray
    }
}

if (![string]::IsNullOrWhiteSpace($ServicePrincipalObjectId)) {
    $deployParams += "--parameters"
    $deployParams += (Format-BicepStringParameter -Name "servicePrincipalObjectId" -Value $ServicePrincipalObjectId)
}

$triggerApiKey = Get-ConfigValue -Name "TRIGGER_API_KEY"
if (![string]::IsNullOrWhiteSpace($triggerApiKey)) {
    $deployParams += "--parameters"
    $deployParams += (Format-BicepStringParameter -Name "triggerApiKey" -Value $triggerApiKey)
} else {
    Write-Host "  [WARN] TRIGGER_API_KEY not set; /trigger endpoint will remain disabled until configured." -ForegroundColor Yellow
}

# Teams channel config (Graph API posting from ACA)
$teamsGroupId = Get-ConfigValue -Name "TEAMS_GROUP_ID"
$teamsChannelId = Get-ConfigValue -Name "TEAMS_CHANNEL_ID"
if ($UseLegacyResourceNames -and ([string]::IsNullOrWhiteSpace($teamsGroupId) -or [string]::IsNullOrWhiteSpace($teamsChannelId))) {
    if ([string]::IsNullOrWhiteSpace($teamsGroupId)) {
        $teamsGroupId = Get-ExistingContainerAppEnvValue -ResolvedResourceGroup $ResourceGroup -ResolvedContainerAppName $existingAcaName -EnvName "TEAMS_GROUP_ID"
    }
    if ([string]::IsNullOrWhiteSpace($teamsChannelId)) {
        $teamsChannelId = Get-ExistingContainerAppEnvValue -ResolvedResourceGroup $ResourceGroup -ResolvedContainerAppName $existingAcaName -EnvName "TEAMS_CHANNEL_ID"
    }
}
if (![string]::IsNullOrWhiteSpace($teamsGroupId) -and ![string]::IsNullOrWhiteSpace($teamsChannelId)) {
    $deployParams += "--parameters"
    $deployParams += (Format-BicepStringParameter -Name "teamsGroupId" -Value $teamsGroupId)
    $deployParams += "--parameters"
    $deployParams += (Format-BicepStringParameter -Name "teamsChannelId" -Value $teamsChannelId)
    Write-Host "  [OK] Teams channel config will be deployed" -ForegroundColor Green
} else {
    Write-Host "  [WARN] TEAMS_GROUP_ID / TEAMS_CHANNEL_ID not set; Teams posting disabled." -ForegroundColor Yellow
}

# Teams webhook URL (fallback/primary for ACA - Graph API is delegated-only)
$teamsWebhookUrl = Get-ConfigValue -Name "TEAMS_WEBHOOK_URL"
if ($UseLegacyResourceNames -and [string]::IsNullOrWhiteSpace($teamsWebhookUrl)) {
    $teamsWebhookUrl = Get-ExistingKeyVaultSecretValue -ResolvedKeyVaultName $existingKvName -SecretName "teams-webhook-url"
}
if (![string]::IsNullOrWhiteSpace($teamsWebhookUrl)) {
    $deployParams += "--parameters"
    $deployParams += (Format-BicepStringParameter -Name "teamsWebhookUrl" -Value $teamsWebhookUrl)
    Write-Host "  [OK] Teams webhook URL will be deployed" -ForegroundColor Green
} else {
    Write-Host "  [WARN] TEAMS_WEBHOOK_URL not set; set this for Teams posting from ACA." -ForegroundColor Yellow
}

Write-Host "  Deploying Azure resources (this may take 5-10 minutes)..." -ForegroundColor Yellow
$deployment = az deployment group create @deployParams --only-show-errors --output json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Bicep deployment failed. Check the error above." -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] Infrastructure deployed successfully`n" -ForegroundColor Green

# Extract resource names from deployment outputs
$acrName = $deployment.properties.outputs.acrName.value
$acrServer = $deployment.properties.outputs.acrLoginServer.value
$kvName = $deployment.properties.outputs.keyVaultName.value
$acaName = $deployment.properties.outputs.containerAppName.value

# Step 4: Upload secrets to Key Vault
Write-Host "Step 4: Uploading secrets to Key Vault..." -ForegroundColor Yellow
$secretsScript = Join-Path $PSScriptRoot "setup_secrets.ps1"
if (Test-Path $secretsScript) {
    & $secretsScript -KeyVaultName $kvName -ResourceGroup $ResourceGroup -EnvFilePath $EnvFilePath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Secrets upload had issues. Check output above." -ForegroundColor Yellow
        Write-Host "   You can manually upload secrets later with: .\scripts\setup_secrets.ps1" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [WARN] Secrets setup script not found. Upload secrets manually:" -ForegroundColor Yellow
    Write-Host "     az keyvault secret set --vault-name $kvName --name odds-api-key --value YOUR_KEY" -ForegroundColor Gray
}

# Step 5: Build and push Docker image
Write-Host "`nStep 5: Building and pushing Docker image..." -ForegroundColor Yellow

# Get ACR login server
Write-Host "  ACR Login Server: $acrServer" -ForegroundColor Gray

# Build and push image using Azure Container Registry Tasks
$imageTag = "$(Get-Date -Format 'yyyyMMdd-HHmmss')"

Write-Host "  Building and pushing Docker image via Azure ACR Build (No local daemon required)..." -ForegroundColor Yellow
$buildRun = az acr build `
    --only-show-errors `
    --no-logs `
    --registry $acrName `
    --image "mlb-predictor:latest" `
    --image "mlb-predictor:$imageTag" `
    . `
    --output json | ConvertFrom-Json

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] ACR Build failed" -ForegroundColor Red
    exit 1
}

$buildRunId = $buildRun.runId
if ([string]::IsNullOrWhiteSpace($buildRunId)) {
    Write-Host "[ERROR] ACR Build queued but no run ID was returned." -ForegroundColor Red
    exit 1
}

Write-Host "  [INFO] ACR build queued: $buildRunId" -ForegroundColor Gray

$terminalBuildStates = @("Succeeded", "Failed", "Canceled", "CancelRequested", "Error", "Timeout")
$lastBuildStatus = ""
for ($attempt = 0; $attempt -lt 180; $attempt++) {
    $buildStatus = az acr task show-run `
        --only-show-errors `
        --registry $acrName `
        --run-id $buildRunId `
        --output json | ConvertFrom-Json

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Could not fetch ACR build status for run $buildRunId." -ForegroundColor Red
        exit 1
    }

    $currentBuildStatus = [string]$buildStatus.status
    if ($currentBuildStatus -ne $lastBuildStatus) {
        Write-Host "  [INFO] ACR build status: $currentBuildStatus" -ForegroundColor Gray
        $lastBuildStatus = $currentBuildStatus
    }

    if ($terminalBuildStates -contains $currentBuildStatus) {
        if ($currentBuildStatus -ne "Succeeded") {
            Write-Host "[ERROR] ACR Build finished with status '$currentBuildStatus'." -ForegroundColor Red
            exit 1
        }

        Write-Host "  [OK] Docker image built and pushed via ACR" -ForegroundColor Green
        break
    }

    Start-Sleep -Seconds 10
}

if ($lastBuildStatus -ne "Succeeded") {
    Write-Host "[ERROR] ACR Build did not reach 'Succeeded' before timing out." -ForegroundColor Red
    exit 1
}

# Step 6: Update Container App
Write-Host "`nStep 6: Updating Container App..." -ForegroundColor Yellow
$imageToDeploy = "${acrServer}/mlb-predictor:$imageTag"
Write-Host "  [INFO] Deploying image: $imageToDeploy" -ForegroundColor Gray
az containerapp update `
    --only-show-errors `
    --name $acaName `
    --resource-group $ResourceGroup `
    --image $imageToDeploy `
    --output none

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Container App update failed" -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] Container App updated with new image`n" -ForegroundColor Green

# Step 7: Get endpoint and verify
Write-Host "Step 7: Verifying deployment..." -ForegroundColor Yellow
$fqdn = az containerapp show `
    --only-show-errors `
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
        Write-Host "  [OK] Health check passed!" -ForegroundColor Green
        Write-Host "     Response: $($response.Content)" -ForegroundColor Gray
    } else {
        Write-Host "  [WARN] Health check returned status: $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [WARN] Health check failed: $_" -ForegroundColor Yellow
    Write-Host "     The app may still be starting. Check logs with:" -ForegroundColor Yellow
    Write-Host "     az containerapp logs show -n $acaName -g $ResourceGroup --follow" -ForegroundColor Gray
}

Write-Host "`n===========================================" -ForegroundColor Cyan
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Resources Created:" -ForegroundColor Cyan
Write-Host "  - Resource Group: $ResourceGroup" -ForegroundColor Gray
Write-Host "  - Container Registry: $acrName" -ForegroundColor Gray
Write-Host "  - Key Vault: $kvName" -ForegroundColor Gray
Write-Host "  - Container App: $acaName" -ForegroundColor Gray
Write-Host "`nEndpoints:" -ForegroundColor Cyan
Write-Host "  - Service URL: https://$fqdn" -ForegroundColor Gray
Write-Host "  - Health: $healthUrl" -ForegroundColor Gray
Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "  1. Test health endpoint: Invoke-WebRequest $healthUrl" -ForegroundColor Gray
Write-Host "  2. Trigger pipeline: Invoke-WebRequest https://$fqdn/trigger -Method POST" -ForegroundColor Gray
Write-Host "  3. View logs: az containerapp logs show -n $acaName -g $ResourceGroup --follow" -ForegroundColor Gray
Write-Host "  4. Monitor metrics in Azure Portal: Application Insights" -ForegroundColor Gray
Write-Host "==========================================`n" -ForegroundColor Cyan
