# 🚀 MLB Prediction Model - Deployment Guide

## Overview

Deploy MLB spread prediction model to Azure Container Apps with full observability, managed identity, and automated scaling.

## Markets & Approach

### **Betting Markets Covered**

- ✅ MLB Spreads (Run Line ±1.5) - **PRIMARY**
- ✅ Moneyline (Win/Loss)
- ✅ Totals (Over/Under)
- ✅ First 5 Innings variants (F5 Spread, F5 ML, F5 Total)

### **ML Strategy**

- **Ensemble Models**: Random Forest + Logistic Regression + XGBoost + LightGBM
- **Features**: 30+ engineered features including:
  - Rolling team stats (5-game and 20-game windows)
  - Pitcher metrics (ERA, wins, differentials)
  - Moneyline-implied probability
  - Rest days analysis
  - Weather conditions (temp, wind, precipitation)
  - Stadium effects (indoor normalization)
  - Statcast metrics (launch speed, launch angle)
  - Date features (month, weekend)
- **Bet Sizing**: Quarter-Kelly criterion for optimal bankroll management

### **Data Sources**

1. **Odds API** (`odds-api-key`) - Real-time betting lines and odds
2. **Bets API** (`bets-api-key`) - Supplementary betting data
3. **Action Network API** (`action-network-*`) - Professional betting insights
4. **Visual Crossing API** (`visual-crossing-api-key`) - Stadium weather data
5. **MLB Stats API** - Official MLB statistics (no key required)

### **Metrics & Monitoring**

- Prediction volume and accuracy
- Model performance (ROI, Sharpe ratio)
- API call tracking and latency
- Edge calculations (predicted vs. market probability)
- Bankroll progression

### **Endpoints Available**

- `GET /health` - Health check for Container Apps probes
- `GET /` - Service information
- `POST /trigger` - Manual pipeline execution

---

## Quick Start (Automated)

### **Option 1: One-Command Deployment** (Recommended)

```powershell
cd c:\Users\JDSB\dev\mlb_gbsv_local_v1.0
.\scripts\deploy.ps1
```

This script will:

1. ✅ Verify prerequisites (Azure CLI, Docker)
2. ✅ Create resource group
3. ✅ Deploy Bicep infrastructure (Key Vault, ACR, Container Apps, SQL DB, App Insights)
4. ✅ Upload secrets from .env file
5. ✅ Build and push Docker image
6. ✅ Deploy Container App with latest image
7. ✅ Verify health endpoint

### **Option 2: Manual Step-by-Step**

#### **Step 1: Configure API Keys**

Copy the template and add your API keys:

```powershell
Copy-Item .env.template .env
notepad .env
```

Fill in at minimum:

- `ODDS_API_KEY` - Get from [The Odds API](https://the-odds-api.com/)
- `BETS_API_KEY` - Get from [BetsAPI](https://betsapi.com/)
- `VISUAL_CROSSING_API_KEY` - Get from [Visual Crossing](https://www.visualcrossing.com/)
- `DISCORD_WEBHOOK_URL` (optional) - For bet alerts

#### **Step 2: Deploy Infrastructure**

```powershell
# Create resource group
az group create -n mlb-gbsv-v1-az-rg -l eastus

# Deploy Bicep template
az deployment group create `
  -g mlb-gbsv-v1-az-rg `
  -f infra/main.bicep `
  --parameters namePrefix=mlb-gbsv-v1-az `
  --parameters sqlAdminPassword="YourSecurePassword123!"
```

Resources created:

- 🔐 Azure Key Vault (secrets management)
- 📦 Azure Container Registry (Docker images)
- 🚀 Container Apps + Environment (app hosting)
- 📊 Application Insights + Log Analytics (observability)
- 🗄️ Azure SQL Database (persistent storage)
- 💾 Storage Account (model artifacts)

#### **Step 3: Upload Secrets**

```powershell
.\scripts\setup_secrets.ps1 -KeyVaultName mlb-gbsv-v1-az-kv
```

#### **Step 4: Build & Deploy**

```powershell
# Login to ACR
az acr login --name mlbgbsvv1azacr

# Build and push image
docker build -t mlbgbsvv1azacr.azurecr.io/mlb-predictor:latest . --platform linux/amd64
docker push mlbgbsvv1azacr.azurecr.io/mlb-predictor:latest

# Update Container App
az containerapp update `
  -n mlb-gbsv-v1-az-aca `
  -g mlb-gbsv-v1-az-rg `
  --image mlbgbsvv1azacr.azurecr.io/mlb-predictor:latest
```

#### **Step 5: Verify Deployment**

```powershell
# Get service URL
$fqdn = az containerapp show -n mlb-gbsv-v1-az-aca -g mlb-gbsv-v1-az-rg --query properties.configuration.ingress.fqdn -o tsv

# Test health endpoint
Invoke-WebRequest "https://$fqdn/health"

# Trigger pipeline manually
Invoke-WebRequest "https://$fqdn/trigger" -Method POST

# View logs
az containerapp logs show -n mlb-gbsv-v1-az-aca -g mlb-gbsv-v1-az-rg --follow
```

---

## GitHub Actions (CI/CD)

The repository includes three workflows:

### **1. Deploy Workflow** (`.github/workflows/deploy.yml`)

Triggers on: Push to `main` branch

Steps:

1. Lint and type check (`ruff`, `mypy`)
2. Run test suite (`pytest`)
3. Deploy infrastructure to Azure
4. Build Docker image
5. Push to ACR
6. Update Container App

**Required Secrets** (add in GitHub repo settings):

```text
AZURE_CREDENTIALS          # Service principal JSON
AZURE_SUBSCRIPTION_ID      # Subscription GUID
AZURE_RESOURCE_GROUP       # mlb-gbsv-v1-az-rg
ODDS_API_KEY              # The Odds API key
BETS_API_KEY              # BetsAPI key
VISUAL_CROSSING_API_KEY   # Weather API key
DISCORD_WEBHOOK_URL       # (optional) Discord alerts
```

### **2. Daily Pipeline** (`.github/workflows/daily.yml`)

Triggers: Cron schedule (daily at specific time)

Steps:

1. Fetch latest odds from APIs
2. Run prediction pipeline
3. Generate bet recommendations
4. Send alerts via Discord/email
5. Track ROI metrics

### **3. Test Workflow** (`.github/workflows/test.yml`)

Triggers: Pull requests

Steps:

1. Run unit tests
2. Run integration tests
3. Check coverage threshold (>80%)
4. Report results

---

## Architecture

```mermaid
graph TB
    subgraph "Azure Container Apps"
        ACA[Container App<br/>MLB Predictor]
        ACA --> |Health Probes| HEALTH[/health endpoint]
        ACA --> |Triggers| PIPE[Daily Pipeline]
    end

    subgraph "Data Sources"
        ODDS[Odds API]
        BETS[BetsAPI]
        WEATHER[Visual Crossing]
        MLBAPI[MLB Stats API]
    end

    subgraph "Azure Services"
        KV[Key Vault<br/>Secrets]
        ACR[Container Registry<br/>Docker Images]
        SQL[SQL Database<br/>Tracking]
        STORAGE[Blob Storage<br/>Model Artifacts]
        INSIGHTS[Application Insights<br/>Telemetry]
    end

    ACA --> |Get Secrets| KV
    ACA --> |Pull Image| ACR
    ACA --> |Store Results| SQL
    ACA --> |Save Models| STORAGE
    ACA --> |Send Metrics| INSIGHTS

    PIPE --> |Fetch Odds| ODDS
    PIPE --> |Fetch Data| BETS
    PIPE --> |Get Weather| WEATHER
    PIPE --> |Get Stats| MLBAPI

    PIPE --> |Predictions| DISCORD[Discord Alerts]
    PIPE --> |Alerts| EMAIL[Email Notifications]
```

---

## Cost Estimation

**Monthly Azure Costs** (East US region):

- Container Apps (1GB RAM, 0.5 CPU, always-on): ~$35/month
- Azure SQL Database (Basic tier): ~$5/month
- Storage Account (LRS, <10GB): ~$1/month
- Key Vault (secrets): ~$0.15/month
- Container Registry (Basic): ~$5/month
- Application Insights (5GB ingestion): ~$2/month

### Total: ~$48/month

**External API Costs**:

- Odds API: Free tier (500 requests/month) or $10-50/month
- BetsAPI: $50-200/month (varies by usage)
- Visual Crossing Weather: Free tier (1000 records/day)

---

## Monitoring & Operations

### **View Logs**

```powershell
# Real-time logs
az containerapp logs show -n mlb-gbsv-v1-az-aca -g mlb-gbsv-v1-az-rg --follow

# Query with KQL
az monitor log-analytics query \
  -w <workspace-id> \
  --analytics-query "ContainerAppConsoleLogs_CL | where ContainerAppName_s == 'mlb-gbsv-v1-az-aca' | project TimeGenerated, Log_s"
```

### **Check Metrics**

```powershell
# Application Insights metrics
az monitor app-insights metrics show \
  --app <app-insights-name> \
  --metric requests/count \
  --aggregation count
```

### **Scale Manually**

```powershell
# Scale to specific replica count
az containerapp update \
  -n mlb-gbsv-v1-az-aca \
  -g mlb-gbsv-v1-az-rg \
  --min-replicas 2 \
  --max-replicas 20
```

### **Restart Container**

```powershell
# Force restart (useful for config changes)
az containerapp revision restart \
  -n mlb-gbsv-v1-az-aca \
  -g mlb-gbsv-v1-az-rg
```

---

## Troubleshooting

### **Container won't start**

```powershell
# Check logs
az containerapp logs show -n mlb-gbsv-v1-az-aca -g mlb-gbsv-v1-az-rg --tail 100

# Common issues:
# - Missing API keys in Key Vault
# - SQL connection string incorrect
# - Docker image pull failure (check ACR permissions)
```

### **Health check failing**

```powershell
# Test locally first
docker run -p 8000:8000 mlbgbsvv1azacr.azurecr.io/mlb-predictor:latest
curl http://localhost:8000/health

# Check environment variables in Container App
az containerapp show -n mlb-gbsv-v1-az-aca -g mlb-gbsv-v1-az-rg --query properties.template.containers[0].env
```

### **Pipeline not running**

```powershell
# Manually trigger via API
$fqdn = az containerapp show -n mlb-gbsv-v1-az-aca -g mlb-gbsv-v1-az-rg --query properties.configuration.ingress.fqdn -o tsv
Invoke-WebRequest "https://$fqdn/trigger" -Method POST

# Check if cron schedule is correct in daily.yml
```

### **High costs**

```powershell
# Check Application Insights ingestion
az monitor app-insights component show --app <name> --resource-group mlb-gbsv-v1-az-rg

# Reduce replica count if not needed
az containerapp update -n mlb-gbsv-v1-az-aca -g mlb-gbsv-v1-az-rg --max-replicas 5

# Use serverless SQL tier (not implemented yet)
```

---

## Security Checklist

- ✅ Managed identity enabled (no explicit credentials)
- ✅ Key Vault for secrets (no env vars with sensitive data)
- ✅ Non-root container user (`mlbuser`)
- ✅ Pinned dependencies (exact versions)
- ✅ HTTPS only (Container Apps default)
- ✅ RBAC roles (Key Vault Secrets User)
- ⚠️ VNET integration (not implemented - Phase 4)
- ⚠️ Private endpoints (not implemented - Phase 4)

---

## Next Steps

1. **Test in Staging**: Deploy to a separate resource group first
2. **Configure Alerts**: Set up Azure Monitor alerts for failures/anomalies
3. **Optimize Costs**: Monitor Application Insights ingestion, right-size Container App
4. **Add VNET**: Implement VNET integration + private endpoints (Phase 4)
5. **Backtest Models**: Use `scripts/backtest.py` to validate historical performance
6. **Track ROI**: Use `src/mlbv1/tracking/roi.py` for bankroll progression

---

## Support & Resources

- 📖 **Azure Container Apps Docs**: [Microsoft Learn](https://learn.microsoft.com/azure/container-apps/)
- 🔐 **Key Vault Best Practices**: [Microsoft Learn](https://learn.microsoft.com/azure/key-vault/general/best-practices)
- 📊 **Application Insights**: [Microsoft Learn](https://learn.microsoft.com/azure/azure-monitor/app/app-insights-overview)
- 🐳 **Docker Best Practices**: [Docker Docs](https://docs.docker.com/develop/dev-best-practices/)

---

**Ready to deploy? Run:**

```powershell
.\scripts\deploy.ps1
```
