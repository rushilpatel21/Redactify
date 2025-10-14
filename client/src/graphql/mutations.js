/**
 * GraphQL Mutations
 * All write/action operations
 */

import { gql } from '@apollo/client';
import { ENTITY_DETAILS } from './queries';

// Anonymize single text
export const ANONYMIZE_TEXT = gql`
  ${ENTITY_DETAILS}
  mutation AnonymizeText(
    $text: String!
    $options: PIIOptionsInput
    $fullRedaction: Boolean!
  ) {
    anonymize(
      text: $text
      options: $options
      fullRedaction: $fullRedaction
    ) {
      anonymizedText
      entitiesProcessed
      processingTime
      strategyUsed
      redactionPercentage
      domainsDetected
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
`;

// Anonymize batch of texts
export const ANONYMIZE_BATCH = gql`
  ${ENTITY_DETAILS}
  mutation AnonymizeBatch(
    $texts: [String!]!
    $options: PIIOptionsInput
    $fullRedaction: Boolean!
  ) {
    anonymizeBatch(
      texts: $texts
      options: $options
      fullRedaction: $fullRedaction
    ) {
      batchSize
      totalProcessingTime
      averageTimePerText
      totalEntitiesFound
      successRate
      strategyUsed
      results {
        anonymizedText
        entitiesProcessed
        processingTime
        domainsDetected
        entities {
          ...EntityDetails
        }
        metadata {
          totalEntities
          domainsUsed
          detectorsUsed
        }
      }
    }
  }
`;

// Preview anonymization (detection only)
export const PREVIEW_ANONYMIZATION = gql`
  ${ENTITY_DETAILS}
  mutation PreviewAnonymization($text: String!, $options: PIIOptionsInput) {
    anonymizeWithPreview(text: $text, options: $options) {
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
