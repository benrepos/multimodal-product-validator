# Multimodal Product Validator

A FastAPI service that validates product listings by comparing images against titles and descriptions using Google Vertex AI embeddings and Gemini LLM.

## Features

- **Embeddings-First Approach**: Fast cosine similarity checks using multimodal embeddings
  - **Cost-effective at scale**: Process hundreds of thousands of products cheaply
  - **Fast filtering**: Auto-pass/fail clear cases without expensive LLM calls
  - **Smart escalation**: Only invoke Gemini for ambiguous cases in the gray zone
- **Gray-Zone LLM Fallback**: Gemini-powered validation for uncertain cases
- **Dual Endpoints**: 
  - `/evaluate`: Hybrid embeddings + LLM approach (recommended for production)
  - `/evaluate/llm-only`: Direct LLM validation (bypass embeddings)
- **Structured Validation**: Returns conflicts across brand, product_type, color, and material
- **Pydantic Schema Validation**: Type-safe responses with automatic validation
- **Per-Category Tuning**: Adjust thresholds per product category to optimize accuracy and cost

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

# API Security
API_KEY=your-secret-api-key-here

# Optional
DEBUG=0
PORT=8000
```

**Note**: If `API_KEY` is not set, the API runs in dev mode without authentication (not recommended for production).

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

**Parameters**:
- `image` (file, required): Product image (JPEG/PNG)
- `title` (string, required): Product title
- `description` (string, required): Product description
- `embedding_dim` (int, optional): Embedding dimension (default: 512, options: 128, 256, 512, 1408)
- `sim_low` (float, optional): Lower similarity threshold for auto-fail (default: 0.08)
- `sim_high` (float, optional): Upper similarity threshold for auto-pass (default: 0.4)

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "x-api-key: your-secret-api-key-here" \
  -F "image=@test/screwdriver.jpeg" \
  -F "title=Phillips head screwdriver" \
  -F "description=Red and black handle screwdriver with cross-tip"
```

**With custom thresholds** (e.g., for apparel category):
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "x-api-key: your-secret-api-key-here" \
  -F "image=@test/screwdriver.jpeg" \
  -F "title=Phillips head screwdriver" \
  -F "description=Red and black handle screwdriver" \
  -F "embedding_dim=512" \
  -F "sim_low=0.10" \
  -F "sim_high=0.35"
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
  -H "x-api-key: your-secret-api-key-here" \
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
| `API_KEY` | - | API key for endpoint authentication (optional, dev mode if not set) |
| `DEBUG` | `0` | Enable debug logging (0 or 1) |
| `PORT` | `8000` | API server port |

### Tuning Thresholds

Thresholds are passed as **request parameters** (not environment variables) to allow per-category tuning:

- **`sim_low`** (default: 0.08): If both `image↔title` and `image↔description` similarities are below this, auto-fail without LLM.
- **`sim_high`** (default: 0.4): If all three similarities are above this, auto-pass without LLM.
- **`embedding_dim`** (default: 512): Embedding dimension (128, 256, 512, 1408).
- **Gray zone**: Between `sim_low` and `sim_high`, the LLM is invoked for nuanced validation.

**Recommended approach**:
1. Collect a small labeled dataset per product category (matches vs mismatches)
2. Plot similarity distributions
3. Choose thresholds that maximize F1 or minimize false positives
4. Pass category-specific thresholds in each API request

**Example**: Tools might use `sim_high=0.4`, while apparel uses `sim_high=0.35` due to higher visual variance.

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

### Why Embeddings-First?

When validating **hundreds of thousands of products**, LLM costs add up quickly. This service uses a hybrid approach to minimize expenses:

**Cost Comparison** (approximate):
- **Embedding similarity check**: ~$0.0001 per product (fast, cheap)
- **Gemini LLM call**: ~$0.001-0.01 per product (10-100x more expensive)

**At scale** (100,000 products):
- **LLM-only approach**: $100-1,000+ in API costs
- **Embeddings-first approach**: $10-50 (80-95% cost reduction)

### How It Works

1. **Auto-pass** (~60-70% of products): High similarity → immediate pass, no LLM call
2. **Auto-fail** (~10-20% of products): Very low similarity → immediate fail, no LLM call
3. **Gray zone** (~10-30% of products): Ambiguous cases → LLM validation only when needed

### Optimization Strategies

- **Embeddings-first**: Fast, cheap cosine similarity checks filter most cases
- **LLM gray-zone only**: Gemini is called only when embeddings are ambiguous
- **Tunable thresholds**: Adjust `sim_low`/`sim_high` per request to control LLM usage
  - Wider thresholds (e.g., `sim_low=0.05`, `sim_high=0.5`) → fewer LLM calls, lower cost
  - Tighter thresholds (e.g., `sim_low=0.10`, `sim_high=0.30`) → more LLM calls, higher accuracy
- **Per-category tuning**: Use stricter thresholds for high-confidence categories to reduce LLM calls
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

