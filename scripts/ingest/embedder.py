"""
embedder.py — Generate embeddings for text chunks using OpenAI.

Batches chunks into groups of 100 to maximise throughput while staying
within the API's token-per-request limits. Retries on transient errors
with exponential backoff.
"""

import asyncio
import logging
import os
from typing import Sequence

from openai import AsyncOpenAI
from dotenv import load_dotenv

from chunker import Chunk

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

log = logging.getLogger(__name__)

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = 100
MAX_RETRIES = 5

# Cost per 1M tokens in USD — update if OpenAI pricing changes.
_COST_PER_1M_TOKENS: dict[str, float] = {
    "text-embedding-3-small": 0.020,
    "text-embedding-3-large": 0.130,
    "text-embedding-ada-002": 0.100,
}

def _client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return AsyncOpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

async def embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
    """
    Embed a list of Chunk objects.

    Returns:
        List of (chunk, embedding_vector) tuples in the same order as input.
    """
    if not chunks:
        return []

    client = _client()
    results: list[tuple[Chunk, list[float]]] = []
    total_tokens = 0
    num_batches = (len(chunks) - 1) // BATCH_SIZE + 1

    # Process in batches
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c.content for c in batch]
        batch_num = i // BATCH_SIZE + 1
        log.info("Embedding batch %d/%d (%d chunks)", batch_num, num_batches, len(batch))

        vectors, usage = await _embed_with_retry(client, texts)
        total_tokens += usage
        log.info("Batch %d/%d — %d tokens used", batch_num, num_batches, usage)
        results.extend(zip(batch, vectors))

    cost_per_1m = _COST_PER_1M_TOKENS.get(EMBEDDING_MODEL, 0.0)
    estimated_cost = total_tokens / 1_000_000 * cost_per_1m
    log.info(
        "Embedding complete — model: %s, total tokens: %d, estimated cost: $%.6f",
        EMBEDDING_MODEL, total_tokens, estimated_cost,
    )

    return results


async def _embed_with_retry(client: AsyncOpenAI, texts: list[str]) -> tuple[list[list[float]], int]:
    """Call the embeddings API with exponential backoff on failure.

    Returns a tuple of (embedding vectors, total tokens used).
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            return vectors, response.usage.total_tokens
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            log.warning("Embedding attempt %d failed: %s — retrying in %ds", attempt + 1, exc, wait)
            await asyncio.sleep(wait)

    raise RuntimeError("Embedding failed after all retries")
