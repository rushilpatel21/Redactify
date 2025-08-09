# Project Structure

## Root Directory Layout
```
redactify/
├── client/              # React frontend application
├── server/              # FastAPI backend API
├── fine-tuned-model/    # ML model training and fine-tuning
├── assets/              # Project screenshots and images
├── .kiro/               # Kiro IDE configuration and steering
├── .vscode/             # VS Code workspace settings
└── README.md            # Main project documentation
```

## Frontend Structure (`client/`)
```
client/
├── src/                 # React source code
├── public/              # Static assets
├── node_modules/        # NPM dependencies
├── package.json         # Dependencies and scripts
├── vite.config.js       # Vite build configuration
├── eslint.config.js     # ESLint configuration
├── .env                 # Environment variables (VITE_BACKEND_BASE_URL)
└── README.md            # Frontend-specific documentation
```

## Backend Structure (`server/`)
```
server/
├── server.py            # Main FastAPI application entry point
├── server_app/          # Application modules (closed folder)
├── mcp_classifier/      # MCP text classification service
├── a2a_ner_*/          # Specialized NER models (financial, medical, legal, etc.)
├── requirements.txt     # Python dependencies
├── *.json              # Configuration files:
│   ├── blocklist.json           # Blocked terms
│   ├── config_static.json       # Static configuration
│   ├── default_pii_options.json # Default PII detection settings
│   ├── entity_type_mapping.json # Entity type mappings
│   ├── pseudonymize_types.json  # Pseudonymization rules
│   └── regex_patterns.json      # Regular expression patterns
├── .env                 # Environment variables (PORT, CONFIDENCE_THRESHOLD)
├── Procfile            # Heroku deployment configuration
├── runtime.txt         # Python runtime version
└── README.md           # Backend-specific documentation
```

## ML Model Structure (`fine-tuned-model/`)
```
fine-tuned-model/
├── fine-tune.ipynb     # Jupyter notebook for model training
├── fine-tuned-ner-model/ # Trained model artifacts
├── ner-model/          # Base NER model
├── logs/               # Training logs
└── not-used/           # Deprecated/unused model files
```

## Key Architectural Patterns

### MCP Microservices
- Each `a2a_ner_*` folder contains a specialized NER model service
- Services communicate via Model Context Protocol (JSON-RPC)
- Domain-specific models: financial, medical, legal, technical, general, PII-specialized

### Configuration Management
- JSON files for static configuration and patterns
- Environment variables for runtime configuration
- Separate `.env` files for client and server

### Development Workflow
- Frontend and backend run independently during development
- Client connects to backend via configurable API URL
- Hot reload enabled for both React (Vite) and FastAPI (uvicorn)

## File Naming Conventions
- **Python**: snake_case for files and modules
- **JavaScript**: camelCase for variables, PascalCase for components
- **Configuration**: lowercase with extensions (.json, .env, .md)
- **Models**: descriptive names with domain prefixes (a2a_ner_medical)