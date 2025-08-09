# Product Overview

Redactify is an advanced PII (Personally Identifiable Information) anonymization platform that helps organizations comply with data privacy regulations by detecting and removing sensitive information from text documents.

## Core Functionality

- **Multi-Method Detection**: Combines machine learning models (Hugging Face NER), rule-based patterns (regex), and Microsoft's Presidio Analyzer for high-accuracy PII detection
- **20 PII Types Supported**: Personal names, organizations, locations, email addresses, phone numbers, credit cards, SSNs, IP addresses, URLs, dates, passwords, API keys, roll numbers, and more
- **Flexible Redaction Options**: 
  - Full redaction with hash-based pseudonyms (e.g., `[PERSON-611732]`)
  - Partial redaction preserving context while masking sensitive parts
- **Selective Anonymization**: Enable/disable specific PII types for targeted redaction

## Architecture

- **Frontend**: React-based web application with intuitive UI
- **Backend**: FastAPI server with distributed MCP (Model Context Protocol) microservices
- **Detection Pipeline**: Specialized models for different domains (medical, technical, general, financial, legal)

## Key Design Principles

- Privacy-first approach with configurable confidence thresholds
- Performance optimization through concurrent processing
- User-friendly interface with real-time feedback
- Modular architecture supporting specialized detection models