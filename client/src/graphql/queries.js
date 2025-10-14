/**
 * GraphQL Queries
 * All read-only operations
 */

import { gql } from '@apollo/client';

// Fragment for entity details (reusable)
export const ENTITY_DETAILS = gql`
  fragment EntityDetails on Entity {
    entityGroup
    word
    start
    end
    score
    confidencePercentage
    isHighConfidence
    detector
  }
`;

// Health check query
export const HEALTH_CHECK = gql`
  query HealthCheck {
    health {
      status
      timestamp
      isOperational
      components {
        detectionEngine
        anonymizationEngine
        modelManager
        mcpServers
      }
      loadedModels
      mcpServers {
        healthy
        total
        healthPercentage
        servers {
          name
          running
          healthy
          port
          url
        }
      }
    }
  }
`;

// Get system configuration
export const GET_CONFIG = gql`
  query GetConfig {
    config {
      version
      architecture
      totalPiiTypes
      supportedEntities
      anonymizationStrategies
      availableModels
    }
  }
`;

// Get supported PII types
export const GET_SUPPORTED_TYPES = gql`
  query GetSupportedTypes {
    supportedPiiTypes
    supportedStrategies
  }
`;

// Detect entities without anonymization (preview)
export const DETECT_ENTITIES = gql`
  ${ENTITY_DETAILS}
  query DetectEntities($text: String!, $options: PIIOptionsInput) {
    detectEntities(text: $text, options: $options) {
      totalEntities
      entityTypes
      processingTime
      entitiesByConfidence
      entities {
        ...EntityDetails
      }
      domainsDetected
    }
  }
`;

// MCP server status
export const MCP_STATUS = gql`
  query MCPStatus {
    mcpStatus {
      healthy
      total
      healthPercentage
      servers {
        name
        running
        healthy
        port
        url
        pid
        uptime
      }
    }
  }
`;
