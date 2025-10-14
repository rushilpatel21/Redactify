/**
 * Apollo GraphQL Client Configuration
 * 
 * This replaces the REST fetch calls with a modern GraphQL client
 */

import { ApolloClient, InMemoryCache, createHttpLink } from '@apollo/client';

const BASE_URL = import.meta.env.VITE_BACKEND_BASE_URL || 'http://localhost:8000';

// Create HTTP link to GraphQL endpoint
const httpLink = createHttpLink({
  uri: `${BASE_URL}/graphql`,
  credentials: 'same-origin',
});

// Configure cache with type policies for better performance
const cache = new InMemoryCache({
  typePolicies: {
    Query: {
      fields: {
        // Cache health checks for 30 seconds
        health: {
          merge(existing, incoming) {
            return incoming;
          },
        },
        // Cache config indefinitely
        config: {
          merge(existing, incoming) {
            return incoming;
          },
        },
      },
    },
    Entity: {
      keyFields: ['start', 'end', 'entityGroup'],
    },
  },
});

// Create Apollo Client instance
const client = new ApolloClient({
  link: httpLink,
  cache,
  defaultOptions: {
    watchQuery: {
      fetchPolicy: 'cache-and-network',
      errorPolicy: 'all',
    },
    query: {
      fetchPolicy: 'network-only',
      errorPolicy: 'all',
    },
    mutate: {
      errorPolicy: 'all',
    },
  },
});

export default client;
