# Redactify Server v2.0

Advanced PII detection and anonymization server using MCP (Model Context Protocol) architecture.

## Features

- **Multi-Method Detection**: Combines ML models, regex patterns, and Microsoft Presidio
- **20+ PII Types**: Person names, organizations, emails, phones, SSNs, credit cards, etc.
- **Flexible Anonymization**: Full redaction with pseudonyms or partial masking
- **Batch Processing**: Efficient processing of multiple texts
- **Domain Classification**: Automatic text classification for specialized models
- **High Performance**: Concurrent processing and model preloading

## Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Gemini API key
GEMINI_API_KEY=your_api_key_here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Server

```bash
python server.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### POST /anonymize
Main anonymization endpoint for single texts.

```json
{
  "text": "John Smith works at Acme Corp. Email: john@acme.com",
  "full_redaction": true,
  "options": {
    "PERSON": true,
    "ORGANIZATION": true,
    "EMAIL_ADDRESS": true
  }
}
```

### POST /anonymize_batch
Batch processing for multiple texts.

```json
{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "full_redaction": true,
  "options": {...}
}
```

### POST /detect
Entity detection only (for debugging).

### GET /config
Server configuration and available models.

### GET /health
Health check endpoint.

## Configuration

Key environment variables:

- `PORT`: Server port (default: 8000)
- `GEMINI_API_KEY`: Google Gemini API key for text classification
- `MAX_WORKERS`: Number of worker threads (default: 8)
- `MAX_MODEL_MEMORY_MB`: Maximum memory for models (default: 4096)
- `CONFIDENCE_THRESHOLD`: Entity confidence threshold (default: 0.5)

## Architecture

The server uses a modular MCP architecture:

- **Detection Engine**: Coordinates multiple detection methods
- **Model Manager**: Handles ML model loading and caching
- **Anonymization Engine**: Processes text anonymization
- **Specialized Models**: Domain-specific NER models (medical, legal, financial, etc.)

## Performance

- Models are preloaded at startup for faster response times
- Concurrent processing for multiple detection methods
- Batch processing support for high-throughput scenarios
- Memory-efficient model management with LRU eviction

## Development

### Project Structure

```
server/
├── server.py                 # Main server entry point
├── detection_engine.py       # Core PII detection logic
├── anonymization_engine.py   # Text anonymization logic
├── model_manager.py          # ML model management
├── models/                   # Model wrapper classes
├── a2a_ner_*/               # Specialized NER models
├── *.json                   # Configuration files
└── requirements.txt         # Python dependencies
```

### Adding New Models

1. Create model wrapper in `models/`
2. Add configuration to `model_manager.py`
3. Update detection logic in `detection_engine.py`

## Deployment

The server is ready for production deployment with:

- Docker support (Dockerfile included)
- Heroku deployment (Procfile included)
- Health check endpoints
- Comprehensive logging
- Error handling and recovery

## License

MIT License - see LICENSE file for details.