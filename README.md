# Multimodal Product Validator

A FastAPI service that validates product listings by comparing images against titles and descriptions using Google Vertex AI embeddings and Gemini LLM.

## Features

- **Embeddings-First Approach**: Fast cosine similarity checks using multimodal embeddings
- **Gray-Zone LLM Fallback**: Gemini-powered validation for ambiguous cases
- **Dual Endpoints**: 
  - `/evaluate`: Hybrid embeddings + LLM approach
  - `/evaluate/llm-only`: Direct LLM validation (bypass embeddings)
- **Structured Validation**: Returns conflicts across brand, product_type, color, and material
- **Pydantic Schema Validation**: Type-safe responses with automatic validation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Endpoints                         │
│  POST /evaluate          POST /evaluate/llm-only             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              Evaluation Pipeline (evaluator.py)              │
│                                                               │
│  1. Compute embeddings (image↔title, image↔desc, title↔desc) │
│  2. Decision gates:                                           │
│     • All sims ≥ sim_high → PASS                             │
│     • Both image sims ≤ sim_low → FAIL                       │
│     • Otherwise → GRAY ZONE (call LLM)                       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  LLM Comparator (llm.py)                     │
│  • Gemini vision model compares image + title + description  │
│  • Returns structured verdict with conflicts                 │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
multimodal-product-validator/
├── app/
│   ├── models.py              # Pydantic schemas
│   └── services/
│       ├── llm.py             # Gemini LLM helpers
│       └── evaluator.py       # Evaluation pipeline
├── config.py                  # Configuration & credentials
├── main.py                    # FastAPI application
├── test/
│   ├── test.py                # CLI demo script
│   └── screwdriver.jpeg       # Sample test image
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

### Prerequisites

- Python 3.8+
- Google Cloud project with Vertex AI enabled
- Service account credentials with Vertex AI permissions

### Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd multimodal-product-validator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:

Create a `.env` file in the project root:

```bash
# Google Cloud
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_CLIENT_EMAIL=your-service-account@project.iam.gserviceaccount.com
GOOGLE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"

# Model Configuration
GEMINI_MODEL_ID=gemini-2.5-flash
EMBEDDING_DIM=512

# Thresholds
SIM_LOW=0.08
SIM_HIGH=0.4

# Optional
DEBUG=0
PORT=8000
```

### Authentication

The service supports multiple authentication methods (in priority order):

1. **Inline service account** (recommended for production):
   - Set `GOOGLE_CLIENT_EMAIL` and `GOOGLE_PRIVATE_KEY` in `.env`

2. **Service account JSON file**:
   - Set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`

3. **Application Default Credentials** (for local development):
   ```bash
   gcloud auth application-default login
   ```

## Usage

### Running the API

Start the FastAPI server:

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at `http://localhost:8000`

Interactive docs: `http://localhost:8000/docs`

### API Endpoints

#### 1. Hybrid Evaluation (Embeddings + LLM)

**Endpoint**: `POST /evaluate`

Computes embeddings first, uses LLM only in gray zone.

```bash
curl -X POST http://localhost:8000/evaluate \
  -F "image=@test/screwdriver.jpeg" \
  -F "title=Phillips head screwdriver" \
  -F "description=Red and black handle screwdriver with cross-tip"
```

**Response**:
```json
{
  "image_title_similarity": 0.42,
  "image_description_similarity": 0.38,
  "title_description_similarity": 0.85,
  "llm_verdict": null,
  "flags": [],
  "decision": "pass",
  "reasons": ["all sims ≥ 0.4"]
}
```

#### 2. LLM-Only Evaluation

**Endpoint**: `POST /evaluate/llm-only`

Skips embeddings, directly uses Gemini for validation.

```bash
curl -X POST http://localhost:8000/evaluate/llm-only \
  -F "image=@test/screwdriver.jpeg" \
  -F "title=Masking tape" \
  -F "description=50mm adhesive tape"
```

**Response**:
```json
{
  "image_title_similarity": null,
  "image_description_similarity": null,
  "title_description_similarity": null,
  "llm_verdict": {
    "verdict": "fail",
    "conflicts": [
      {
        "attribute": "product_type",
        "source_pair": "image_title",
        "title_value": "masking tape",
        "image_value": "screwdriver",
        "severity": "major",
        "comment": "Product categories differ completely"
      }
    ],
    "pair_disagreements": ["image_title", "image_description"],
    "support": {...},
    "notes": ""
  },
  "flags": ["LLM_ONLY"],
  "decision": "fail",
  "reasons": ["llm verdict fail"]
}
```

### CLI Demo

Run the test script:

```bash
python test/test.py
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_CLOUD_PROJECT_ID` | - | GCP project ID (required) |
| `GOOGLE_CLOUD_LOCATION` | `us-central1` | Vertex AI region |
| `GOOGLE_CLIENT_EMAIL` | - | Service account email |
| `GOOGLE_PRIVATE_KEY` | - | Service account private key |
| `GEMINI_MODEL_ID` | `gemini-2.5-flash` | Gemini model for LLM validation |
| `EMBEDDING_DIM` | `512` | Embedding dimension (128, 256, 512, 1408) |
| `SIM_LOW` | `0.08` | Lower similarity threshold (auto-fail) |
| `SIM_HIGH` | `0.4` | Upper similarity threshold (auto-pass) |
| `DEBUG` | `0` | Enable debug logging (0 or 1) |
| `PORT` | `8000` | API server port |

### Tuning Thresholds

- **`SIM_LOW`**: If both `image↔title` and `image↔description` similarities are below this, auto-fail without LLM.
- **`SIM_HIGH`**: If all three similarities are above this, auto-pass without LLM.
- **Gray zone**: Between `SIM_LOW` and `SIM_HIGH`, the LLM is invoked for nuanced validation.

Recommended approach:
1. Collect a small labeled dataset (matches vs mismatches)
2. Plot similarity distributions
3. Choose thresholds that maximize F1 or minimize false positives

## Docker Deployment

Build the image:

```bash
docker build -t product-validator .
```

Run the container:

```bash
docker run -p 8000:8000 \
  -e GOOGLE_CLOUD_PROJECT_ID=your-project \
  -e GOOGLE_CLIENT_EMAIL=your-sa@project.iam.gserviceaccount.com \
  -e GOOGLE_PRIVATE_KEY="$(cat service-account-key.json | jq -r .private_key)" \
  product-validator
```

## Development

### Running Tests

```bash
# Run CLI demo
python test/test.py

# Test API endpoints
curl -X POST http://localhost:8000/evaluate \
  -F "image=@test/screwdriver.jpeg" \
  -F "title=Test product" \
  -F "description=Test description"
```

### Code Structure

- **Pure functions**: All LLM and evaluation logic uses pure functions (no classes)
- **Pydantic validation**: Strict schema validation for LLM responses
- **Env-driven config**: All settings via environment variables
- **Logging**: Structured logging with DEBUG flag

## Cost Optimization

- **Embeddings-first**: Fast, cheap cosine similarity checks filter most cases
- **LLM gray-zone only**: Gemini is called only when embeddings are ambiguous
- **Tunable thresholds**: Adjust `SIM_LOW`/`SIM_HIGH` to control LLM usage
- **Caching**: Consider adding embedding/verdict caching for repeated queries

## Troubleshooting

### Authentication Errors

```
google.auth.exceptions.DefaultCredentialsError
```

**Solution**: Ensure `GOOGLE_CLIENT_EMAIL` and `GOOGLE_PRIVATE_KEY` are set, or run `gcloud auth application-default login`.

### Model Not Found

```
404 Publisher Model `gemini-2.5-flash` was not found
```

**Solution**: 
- Ensure Vertex AI Generative AI is enabled in your project
- Check the model ID is correct: `GEMINI_MODEL_ID=gemini-2.5-flash`
- Verify your service account has `aiplatform.endpoints.predict` permission

### Low Similarity Scores

If all similarity scores are unexpectedly low:
- Try different embedding dimensions (256, 512, 1408)
- Ensure titles/descriptions are descriptive and match the image content
- Check image quality and format (JPEG/PNG)

## License

MIT

## Contributing

Pull requests welcome. For major changes, please open an issue first.

