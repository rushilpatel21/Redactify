# Technology Stack

## Frontend (Client)
- **Framework**: React 19.0.0 with Vite build system
- **Language**: JavaScript (ES modules)
- **UI Libraries**: 
  - Framer Motion 12.4.7 (animations)
  - React Icons 5.5.0 (iconography)
  - SweetAlert2 11.17.2 (notifications)
- **Development**: ESLint for code quality, SWC for fast compilation
- **Build Tool**: Vite 6.1.0

## Backend (Server)
- **Framework**: FastAPI 0.95.0+ with Python 3.9+
- **Server**: Uvicorn ASGI server
- **PII Detection Stack**:
  - Microsoft Presidio Analyzer/Anonymizer 2.2.33+
  - Hugging Face Transformers 4.28.1+ (BERT NER models)
  - PyTorch 2.0.0+ for ML inference
  - OpenAI 1.0.0+ for LLM support
- **Architecture**: MCP (Model Context Protocol) 0.3.0+ for distributed microservices
- **Utilities**: python-dotenv, requests, pydantic

## Common Commands

### Development Setup
```bash
# Backend setup
cd server
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python server.py  # Runs on http://localhost:8000

# Frontend setup  
cd client
npm install
npm run dev  # Runs on http://localhost:5173
```

### Build & Deploy
```bash
# Frontend build
cd client
npm run build
npm run preview

# Backend production
cd server
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Development Tools
```bash
# Frontend linting
cd client
npm run lint

# Environment setup
# Backend: Configure .env with PORT, CONFIDENCE_THRESHOLD
# Frontend: Configure .env with VITE_BACKEND_BASE_URL
```

## Configuration Files
- **Frontend**: `vite.config.js`, `eslint.config.js`, `.env`
- **Backend**: `.env`, `requirements.txt`, various JSON config files
- **Deployment**: `Procfile`, `runtime.txt` for Heroku deployment