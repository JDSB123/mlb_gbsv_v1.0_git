param location string = 'eastus'
param namePrefix string = 'mlb-gbsv-v1-az'
param servicePrincipalObjectId string = ''
param enableRoleAssignments bool = false
param sqlAdminLogin string = 'mlbadmin'
@secure()
param sqlAdminPassword string = newGuid()
@secure()
param triggerApiKey string = ''
param allowUnauthTrigger bool = false
param containerImage string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
param teamsGroupId string = ''
param teamsChannelId string = ''
@secure()
param teamsWebhookUrl string = ''
param dailyCronSchedule string = '0 15 * * *' // 15:00 UTC = 11:00 AM ET (before first pitch)

var acrName = replace('${namePrefix}-acr', '-', '')
var storageName = replace('${namePrefix}-sto', '-', '')
var kvName = '${namePrefix}-kv'
var acaEnvName = '${namePrefix}-acaenv'
var acaName = '${namePrefix}-aca'
var logAnalyticsName = '${namePrefix}-logs'
var appInsightsName = '${namePrefix}-ai'
var sqlServerName = replace('${namePrefix}-sql', '-', '')
var sqlDbName = 'mlb-tracking'
var kvSecretsUserRoleId = '4633458b-17de-408a-b874-0445c86b69e6'
var acaJobName = '${namePrefix}-daily-trigger'
var hasTriggerApiKey = !empty(triggerApiKey)
var hasTeamsConfig = !empty(teamsGroupId) && !empty(teamsChannelId)
var hasTeamsWebhook = !empty(teamsWebhookUrl)

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

resource storage 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: kvName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      name: 'standard'
      family: 'A'
    }
    enableSoftDelete: true
    enabledForDeployment: true
    enabledForTemplateDeployment: true
    enableRbacAuthorization: true
  }
}

resource kvRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (enableRoleAssignments && servicePrincipalObjectId != '') {
  name: guid(keyVault.id, servicePrincipalObjectId, kvSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsUserRoleId)
    principalId: servicePrincipalObjectId
    principalType: 'ServicePrincipal'
  }
}

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

resource sqlServer 'Microsoft.Sql/servers@2023-05-01-preview' = {
  name: sqlServerName
  location: location
  properties: {
    administratorLogin: sqlAdminLogin
    administratorLoginPassword: sqlAdminPassword
    publicNetworkAccess: 'Enabled'
  }
}

resource sqlServerFirewall 'Microsoft.Sql/servers/firewallRules@2023-05-01-preview' = {
  parent: sqlServer
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

resource sqlDatabase 'Microsoft.Sql/servers/databases@2023-05-01-preview' = {
  parent: sqlServer
  name: sqlDbName
  location: location
  sku: {
    name: 'Basic'
    tier: 'Basic'
    capacity: 5
  }
  properties: {
    collation: 'SQL_Latin1_General_CP1_CI_AS'
    maxSizeBytes: 2147483648 // 2GB
    catalogCollation: 'SQL_Latin1_General_CP1_CI_AS'
    zoneRedundant: false
  }
}

resource acaEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: acaEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource acaApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: acaName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: acaEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.name
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: concat(
        [
          {
            name: 'acr-password'
            value: acr.listCredentials().passwords[0].value
          }
          {
            name: 'appinsights-connection-string'
            value: appInsights.properties.ConnectionString
          }
          {
            name: 'sql-connection-string'
            value: 'Server=tcp:${sqlServer.properties.fullyQualifiedDomainName},1433;Database=${sqlDbName};User ID=${sqlAdminLogin};Password=${sqlAdminPassword};Encrypt=true;Connection Timeout=30;'
          }
        ],
        hasTriggerApiKey
          ? [
              {
                name: 'trigger-api-key'
                value: triggerApiKey
              }
            ]
          : [],
        hasTeamsWebhook
          ? [
              {
                name: 'teams-webhook-url'
                value: teamsWebhookUrl
              }
            ]
          : []
      )
    }
    template: {
      containers: [
        {
          name: 'mlbv1'
          image: containerImage
          resources: {
            cpu: json('2.0')
            memory: '4Gi'
          }
          env: concat(
            [
              {
                name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
                secretRef: 'appinsights-connection-string'
              }
              {
                name: 'AZURE_KEY_VAULT_NAME'
                value: kvName
              }
              {
                name: 'SQL_CONNECTION_STRING'
                secretRef: 'sql-connection-string'
              }
              {
                name: 'ALLOW_UNAUTH_TRIGGER'
                value: string(allowUnauthTrigger)
              }
            ],
            hasTriggerApiKey
              ? [
                  {
                    name: 'TRIGGER_API_KEY'
                    secretRef: 'trigger-api-key'
                  }
                ]
              : [],
            hasTeamsConfig
              ? [
                  {
                    name: 'TEAMS_GROUP_ID'
                    value: teamsGroupId
                  }
                  {
                    name: 'TEAMS_CHANNEL_ID'
                    value: teamsChannelId
                  }
                ]
              : [],
            hasTeamsWebhook
              ? [
                  {
                    name: 'TEAMS_WEBHOOK_URL'
                    secretRef: 'teams-webhook-url'
                  }
                ]
              : []
          )
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              initialDelaySeconds: 30
              periodSeconds: 30
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              initialDelaySeconds: 10
              periodSeconds: 10
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 10
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '100'
              }
            }
          }
        ]
      }
    }
  }
}

// Daily trigger job — hits /trigger on a CRON schedule
resource dailyTriggerJob 'Microsoft.App/jobs@2023-05-01' = {
  name: acaJobName
  location: location
  properties: {
    environmentId: acaEnv.id
    configuration: {
      triggerType: 'Schedule'
      scheduleTriggerConfig: {
        cronExpression: dailyCronSchedule
        parallelism: 1
        replicaCompletionCount: 1
      }
      replicaTimeout: 600 // 10 min max
      replicaRetryLimit: 1
      secrets: hasTriggerApiKey
        ? [
            {
              name: 'trigger-api-key'
              value: triggerApiKey
            }
          ]
        : []
    }
    template: {
      containers: [
        {
          name: 'trigger'
          image: 'curlimages/curl:latest'
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
          command: [
            'sh'
            '-c'
            hasTriggerApiKey
              ? 'curl -sf -X POST -H "X-Trigger-Key: $TRIGGER_API_KEY" --max-time 300 https://${acaApp.properties.configuration.ingress.fqdn}/trigger'
              : 'curl -sf -X POST --max-time 300 https://${acaApp.properties.configuration.ingress.fqdn}/trigger'
          ]
          env: hasTriggerApiKey
            ? [
                {
                  name: 'TRIGGER_API_KEY'
                  secretRef: 'trigger-api-key'
                }
              ]
            : []
        }
      ]
    }
  }
}

// Grant Container App managed identity access to Key Vault
resource acaKvRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (enableRoleAssignments) {
  name: guid(keyVault.id, acaApp.name, kvSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsUserRoleId)
    principalId: acaApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

output acrLoginServer string = acr.properties.loginServer
output storageAccountName string = storage.name
output containerAppName string = acaApp.name
output keyVaultName string = keyVault.name
output appInsightsConnectionString string = appInsights.properties.ConnectionString
output logAnalyticsWorkspaceId string = logAnalytics.id
output containerAppIdentityPrincipalId string = acaApp.identity.principalId
output sqlServerName string = sqlServer.name
output sqlServerFqdn string = sqlServer.properties.fullyQualifiedDomainName
output sqlDatabaseName string = sqlDatabase.name
