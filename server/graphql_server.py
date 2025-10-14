"""
GraphQL Server Integration for FastAPI
Adds GraphQL endpoint to existing REST API
"""

import strawberry
from strawberry.fastapi import GraphQLRouter
from typing import Optional
import logging

from graphql_schema import Query, Mutation, Subscription, schema
from graphql_resolvers import (
    resolve_health,
    resolve_mcp_status,
    resolve_config,
    resolve_detect_entities,
    resolve_anonymize,
    resolve_anonymize_batch,
    resolve_anonymize_with_preview,
)

logger = logging.getLogger("GraphQLServer")


# ============================================================================
# ATTACH RESOLVERS TO SCHEMA
# ============================================================================
# Note: Resolvers are attached via the @strawberry.field decorator in schema
# The actual implementation is in graphql_resolvers.py


# ============================================================================
# CREATE GRAPHQL ROUTER
# ============================================================================

def create_graphql_router() -> GraphQLRouter:
    """
    Create GraphQL router for FastAPI
    
    Features:
    - GraphQL Playground UI at /graphql
    - Introspection enabled for development
    - Async execution
    - Error handling
    """
    
    graphql_app = GraphQLRouter(
        schema,
        graphiql=True,  # Enable GraphQL Playground UI
        path="/graphql"
    )
    
    logger.info("GraphQL router created successfully")
    logger.info("GraphQL Playground available at /graphql")
    
    return graphql_app


# ============================================================================
# EXAMPLE QUERIES FOR DOCUMENTATION
# ============================================================================

EXAMPLE_QUERIES = """
# ============================================================================
# EXAMPLE GRAPHQL QUERIES & MUTATIONS
# ============================================================================

# 1. HEALTH CHECK
# ============================================================================
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

# 2. GET CONFIGURATION
# ============================================================================
query GetConfig {
  config {
    version
    architecture
    totalPiiTypes
    supportedEntities
    anonymizationStrategies
    availableModels
    modelStats
  }
}

# 3. DETECT ENTITIES (Preview)
# ============================================================================
query DetectEntities {
  detectEntities(
    text: "John Smith works at Google in New York. Email: john@example.com"
  ) {
    totalEntities
    entityTypes
    processingTime
    entitiesByConfidence
    entities {
      entityGroup
      word
      start
      end
      score
      confidencePercentage
      isHighConfidence
      detector
    }
    domainsDetected
  }
}

# 4. ANONYMIZE TEXT (Simple)
# ============================================================================
mutation AnonymizeText {
  anonymize(
    text: "John Smith works at Google. Contact: john@example.com, 555-1234"
    fullRedaction: true
  ) {
    anonymizedText
    entitiesProcessed
    processingTime
    strategyUsed
    redactionPercentage
    entities {
      entityGroup
      word
      confidencePercentage
    }
    entitiesByType
    metadata {
      totalEntities
      domainsUsed
      detectorsUsed
    }
  }
}

# 5. ANONYMIZE WITH SELECTIVE PII TYPES
# ============================================================================
mutation AnonymizeSelective {
  anonymize(
    text: "John Smith works at Google. Email: john@example.com, Phone: 555-1234"
    options: {
      person: true
      emailAddress: true
      phoneNumber: false  # Don't anonymize phone numbers
      organization: false  # Don't anonymize organizations
    }
    fullRedaction: true
  ) {
    anonymizedText
    entitiesProcessed
    entities {
      entityGroup
      word
    }
  }
}

# 6. PARTIAL REDACTION (Masking)
# ============================================================================
mutation PartialRedaction {
  anonymize(
    text: "Contact John Smith at john.smith@example.com"
    fullRedaction: false  # Use masking instead
  ) {
    anonymizedText
    strategyUsed
  }
}

# 7. BATCH ANONYMIZATION
# ============================================================================
mutation AnonymizeBatch {
  anonymizeBatch(
    texts: [
      "John works at Google",
      "Mary's email is mary@example.com",
      "Call Bob at 555-1234"
    ]
    fullRedaction: true
  ) {
    batchSize
    totalProcessingTime
    averageTimePerText
    totalEntitiesFound
    successRate
    results {
      anonymizedText
      entitiesProcessed
      entities {
        entityGroup
        word
      }
    }
  }
}

# 8. PREVIEW BEFORE ANONYMIZATION
# ============================================================================
mutation PreviewAnonymization {
  anonymizeWithPreview(
    text: "John Smith works at Google. Email: john@example.com"
  ) {
    totalEntities
    entityTypes
    entitiesByConfidence
    entities {
      entityGroup
      word
      start
      end
      confidencePercentage
      isHighConfidence
    }
  }
}

# 9. GET SUPPORTED PII TYPES
# ============================================================================
query GetSupportedTypes {
  supportedPiiTypes
  supportedStrategies
}

# 10. MCP SERVER STATUS
# ============================================================================
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

# 11. COMPLEX QUERY WITH FRAGMENTS
# ============================================================================
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

query ComplexAnonymization {
  detectEntities(text: "John Smith at john@example.com") {
    entities {
      ...EntityDetails
    }
  }
}

mutation AnonymizeWithDetails {
  anonymize(text: "John Smith at john@example.com") {
    anonymizedText
    entities {
      ...EntityDetails
    }
    entitiesByType
    metadata {
      totalEntities
      domainsUsed
      detectorsUsed
      uniqueEntityTypes
    }
  }
}

# 12. ERROR HANDLING EXAMPLE
# ============================================================================
mutation HandleErrors {
  anonymize(text: "") {  # This will fail validation
    anonymizedText
  }
}

# 13. SELECTIVE FIELDS (GraphQL Advantage)
# ============================================================================
# Only request what you need - reduces payload size
mutation MinimalResponse {
  anonymize(text: "John Smith") {
    anonymizedText  # Only get the anonymized text, nothing else
  }
}

mutation DetailedResponse {
  anonymize(text: "John Smith") {
    anonymizedText
    entitiesProcessed
    processingTime
    strategyUsed
    redactionPercentage
    entitiesByType
    metadata {
      totalEntities
      domainsUsed
      detectorsUsed
      uniqueEntityTypes
    }
  }
}

# 14. MULTIPLE OPERATIONS IN ONE REQUEST
# ============================================================================
query MultipleOperations {
  health {
    status
    isOperational
  }
  config {
    version
    totalPiiTypes
  }
  supportedPiiTypes
}
"""


def get_example_queries() -> str:
    """Get example queries for documentation"""
    return EXAMPLE_QUERIES
