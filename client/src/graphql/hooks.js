/**
 * Custom React Hooks for GraphQL Operations
 * Provides easy-to-use hooks for components
 */

import { useMutation, useQuery } from '@apollo/client/react';
import {
  ANONYMIZE_TEXT,
  ANONYMIZE_BATCH,
  PREVIEW_ANONYMIZATION,
} from './mutations';
import {
  HEALTH_CHECK,
  GET_CONFIG,
  GET_SUPPORTED_TYPES,
  DETECT_ENTITIES,
  MCP_STATUS,
} from './queries';

/**
 * Hook for anonymizing text
 * @returns {Object} { anonymize, loading, error, data }
 */
export const useAnonymize = () => {
  const [anonymizeMutation, { loading, error, data }] = useMutation(ANONYMIZE_TEXT, {
    errorPolicy: 'all',
  });

  const anonymize = async (text, options = null, fullRedaction = true) => {
    try {
      const result = await anonymizeMutation({
        variables: {
          text,
          options: options ? convertOptionsToGraphQL(options) : null,
          fullRedaction,
        },
      });
      return result.data?.anonymize;
    } catch (err) {
      console.error('Anonymization error:', err);
      throw err;
    }
  };

  return { anonymize, loading, error, data };
};

/**
 * Hook for batch anonymization
 * @returns {Object} { anonymizeBatch, loading, error, data }
 */
export const useAnonymizeBatch = () => {
  const [anonymizeBatchMutation, { loading, error, data }] = useMutation(ANONYMIZE_BATCH, {
    errorPolicy: 'all',
  });

  const anonymizeBatch = async (texts, options = null, fullRedaction = true) => {
    try {
      const result = await anonymizeBatchMutation({
        variables: {
          texts,
          options: options ? convertOptionsToGraphQL(options) : null,
          fullRedaction,
        },
      });
      return result.data?.anonymizeBatch;
    } catch (err) {
      console.error('Batch anonymization error:', err);
      throw err;
    }
  };

  return { anonymizeBatch, loading, error, data };
};

/**
 * Hook for previewing anonymization (detection only)
 * @returns {Object} { preview, loading, error, data }
 */
export const usePreviewAnonymization = () => {
  const [previewMutation, { loading, error, data }] = useMutation(PREVIEW_ANONYMIZATION, {
    errorPolicy: 'all',
  });

  const preview = async (text, options = null) => {
    try {
      const result = await previewMutation({
        variables: {
          text,
          options: options ? convertOptionsToGraphQL(options) : null,
        },
      });
      return result.data?.anonymizeWithPreview;
    } catch (err) {
      console.error('Preview error:', err);
      throw err;
    }
  };

  return { preview, loading, error, data };
};

/**
 * Hook for health check
 * @returns {Object} { health, loading, error, refetch }
 */
export const useHealthCheck = () => {
  const { data, loading, error, refetch } = useQuery(HEALTH_CHECK, {
    pollInterval: 30000, // Poll every 30 seconds
    errorPolicy: 'all',
  });

  return {
    health: data?.health,
    loading,
    error,
    refetch,
  };
};

/**
 * Hook for system configuration
 * @returns {Object} { config, loading, error }
 */
export const useConfig = () => {
  const { data, loading, error } = useQuery(GET_CONFIG, {
    errorPolicy: 'all',
  });

  return {
    config: data?.config,
    loading,
    error,
  };
};

/**
 * Hook for supported PII types
 * @returns {Object} { types, strategies, loading, error }
 */
export const useSupportedTypes = () => {
  const { data, loading, error } = useQuery(GET_SUPPORTED_TYPES, {
    errorPolicy: 'all',
  });

  return {
    types: data?.supportedPiiTypes,
    strategies: data?.supportedStrategies,
    loading,
    error,
  };
};

/**
 * Hook for entity detection
 * @returns {Object} { detectEntities, loading, error, data }
 */
export const useDetectEntities = () => {
  const { data, loading, error, refetch } = useQuery(DETECT_ENTITIES, {
    skip: true, // Don't run automatically
    errorPolicy: 'all',
  });

  const detectEntities = async (text, options = null) => {
    try {
      const result = await refetch({
        text,
        options: options ? convertOptionsToGraphQL(options) : null,
      });
      return result.data?.detectEntities;
    } catch (err) {
      console.error('Detection error:', err);
      throw err;
    }
  };

  return { detectEntities, loading, error, data };
};

/**
 * Hook for MCP server status
 * @returns {Object} { mcpStatus, loading, error, refetch }
 */
export const useMCPStatus = () => {
  const { data, loading, error, refetch } = useQuery(MCP_STATUS, {
    pollInterval: 60000, // Poll every 60 seconds
    errorPolicy: 'all',
  });

  return {
    mcpStatus: data?.mcpStatus,
    loading,
    error,
    refetch,
  };
};

/**
 * Helper function to convert options object to GraphQL format
 * Converts from: { PERSON: true, EMAIL_ADDRESS: false }
 * To: { person: true, emailAddress: false }
 */
function convertOptionsToGraphQL(options) {
  const mapping = {
    'PERSON': 'person',
    'ORGANIZATION': 'organization',
    'LOCATION': 'location',
    'EMAIL_ADDRESS': 'emailAddress',
    'PHONE_NUMBER': 'phoneNumber',
    'CREDIT_CARD': 'creditCard',
    'SSN': 'ssn',
    'IP_ADDRESS': 'ipAddress',
    'URL': 'url',
    'DATE_TIME': 'dateTime',
    'PASSWORD': 'password',
    'API_KEY': 'apiKey',
    'ROLL_NUMBER': 'rollNumber',
    'DEPLOY_TOKEN': 'deployToken',
    'AUTHENTICATION': 'authentication',
    'FINANCIAL': 'financial',
    'CREDENTIAL': 'credential',
    'DEVICE': 'device',
    'MEDICAL': 'medical',
    'ID_NUMBER': 'idNumber',
  };

  const graphqlOptions = {};
  for (const [key, value] of Object.entries(options)) {
    const graphqlKey = mapping[key];
    if (graphqlKey) {
      graphqlOptions[graphqlKey] = value;
    }
  }

  return graphqlOptions;
}
