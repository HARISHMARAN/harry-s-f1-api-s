# Harry's F1 API (Vercel)

This repo contains a Vercel-ready API for the Harry's F1 chatbot. It exposes:

- `GET /health`
- `POST /api/v1/chat`
- `POST /api/v1/chat/stream` (SSE)

The API uses:
- OpenAI for responses + embeddings
- Postgres + pgvector for the knowledge base
- A structured F1 SQL database (Ergast/FastF1 schema)

## Deploy on Vercel

1. Create a new Vercel project from this repo.
2. Add a Postgres database from the Vercel Marketplace (Neon/Supabase/etc.).
3. Set environment variables (Production + Preview):

```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://...
API_CORS_ORIGINS=https://harry-s-f1-data.vercel.app,https://harry-s-f1-data-git-main-harishmarans-projects.vercel.app
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
RAG_TOP_K=5
DATABASE_QUERY_TIMEOUT=5000
```

## Database setup

Run these SQL files on your Postgres instance (once):

- `scripts/db/schema.sql`
- `scripts/db/users.sql`
- `scripts/db/indexes.sql`

The knowledge base table requires `pgvector` enabled.

## Ingestion pipeline (optional, for knowledge base)

The ingestion pipeline is in `scripts/ingest`. It scrapes sources and loads
embeddings into Postgres. You will need Python 3.12 and an OpenAI API key.

```
cd scripts
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium

cp .env.example .env
# set OPENAI_API_KEY and DATABASE_URL in .env

python ingest/run_ingest.py
```

## Local dev

```
npm install
npm run dev
```

The API runs at `http://localhost:3000` (Vercel dev) or directly with Node.

