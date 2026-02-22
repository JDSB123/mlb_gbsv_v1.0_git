param location string = 'eastus'
param namePrefix string = 'mlb-gbsv-v1-az'

var acrName = '${namePrefix}-acr'
var storageName = replace('${namePrefix}-sto', '-', '')
var kvName = '${namePrefix}-kv'
var acaEnvName = '${namePrefix}-acaenv'
var acaName = '${namePrefix}-aca'

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
    accessPolicies: []
  }
}

resource acaEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: acaEnvName
  location: location
  properties: {}
}

resource acaApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: acaName
  location: location
  properties: {
    managedEnvironmentId: acaEnv.id
    configuration: {
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.name
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: listCredentials(acr.id, acr.apiVersion).passwords[0].value
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'mlbv1'
          image: '${acr.properties.loginServer}/mlbv1:latest'
          resources: {
            cpu: 0.5
            memory: '1Gi'
          }
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
      }
    }
  }
}

output acrLoginServer string = acr.properties.loginServer
output storageAccountName string = storage.name
output containerAppName string = acaApp.name
