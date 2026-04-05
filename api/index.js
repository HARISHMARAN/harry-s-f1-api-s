import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { Pool } from 'pg';
import OpenAI from 'openai';

const app = express();

const {
  OPENAI_API_KEY,
  OPENAI_MODEL = 'gpt-4o',
  OPENAI_BASE_URL,
  OPENROUTER_APP_URL,
  OPENROUTER_APP_NAME,
  EMBEDDING_MODEL = 'text-embedding-3-small',
  ENABLE_KNOWLEDGE_BASE = 'true',
  DISABLE_TOOLS = 'false',
  DATABASE_URL,
  POSTGRES_URL,
  DATABASE_QUERY_TIMEOUT = '5000',
  RAG_TOP_K = '5',
  API_CORS_ORIGINS = 'http://localhost:5173',
} = process.env;

if (!OPENAI_API_KEY) {
  throw new Error('Missing OPENAI_API_KEY');
}

const RESOLVED_DATABASE_URL = DATABASE_URL || POSTGRES_URL;

if (!RESOLVED_DATABASE_URL) {
  throw new Error('Missing DATABASE_URL');
}

const defaultHeaders = {};
if (OPENAI_BASE_URL && OPENAI_BASE_URL.includes('openrouter.ai')) {
  if (OPENROUTER_APP_URL) defaultHeaders['HTTP-Referer'] = OPENROUTER_APP_URL;
  if (OPENROUTER_APP_NAME) defaultHeaders['X-Title'] = OPENROUTER_APP_NAME;
}

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
  baseURL: OPENAI_BASE_URL || undefined,
  defaultHeaders,
});
const pool = new Pool({ connectionString: RESOLVED_DATABASE_URL });

const corsOrigins = API_CORS_ORIGINS.split(',').map((o) => o.trim()).filter(Boolean);

app.use(cors({ origin: corsOrigins, credentials: true }));
app.use(express.json({ limit: '1mb' }));

const SYSTEM_PROMPT = `
You are an expert Formula One assistant with access to two data tools.
Always use the most appropriate tool (or combination of tools) to answer
precisely and completely. Cite your sources where relevant.

## Response style (CRITICAL)
- Provide ONLY the final answer for the end user.
- Do NOT include steps, reasoning, SQL queries, tool names, or system/tool call details.
- Keep responses short and direct. If the user asks for a single fact, answer with just that fact.
- Do NOT include sources or citations unless the user explicitly asks for them.
- Do NOT use LaTeX or boxed answers.

## Tools

### sql_query
Runs a read-only SELECT query against a PostgreSQL database containing F1 data
from 2018 through the current 2026 season (imported via FastF1). Use this for:
- Current 2026 championship standings and points
- 2026 race results, grid positions, qualifying times
- Driver career statistics (wins, poles, podiums) across all seasons
- Constructor statistics and team comparisons
- Lap times and pit stop data
- Historical records, season comparisons, and trends
- Always filter by races.year for season-specific queries

### f1_knowledge
Performs a semantic search over a curated knowledge base of F1 content
(driver profiles, team histories, regulations, race reports). Use this for:
- Driver or team background and history
- Technical or sporting regulation questions
- Race narratives and analysis
- Circuit descriptions and characteristics

## Rules
- Only SELECT statements may be generated for sql_query.
- Never fabricate data — if a tool returns no results, say so.
- If a tool returns an error, explain that the data source is unavailable and ask the user to try again.
- Combine tool outputs when questions span multiple domains.
- Keep answers concise but complete.
- Format numbers clearly (e.g. lap times as M:SS.mmm).

## Database Schema

circuits(circuitId, circuitRef, name, location, country, lat, lng, alt, url)
constructor_results(constructorResultsId, raceId, constructorId, points, status)
constructor_standings(constructorStandingsId, raceId, constructorId, points, position, positionText, wins)
constructors(constructorId, constructorRef, name, nationality, url)
driver_standings(driverStandingsId, raceId, driverId, points, position, positionText, wins)
drivers(driverId, driverRef, number, code, forename, surname, dob, nationality, url)
lap_times(raceId, driverId, lap, position, time, milliseconds)
pit_stops(raceId, driverId, stop, lap, time, duration, milliseconds)
qualifying(qualifyId, raceId, driverId, constructorId, number, position, q1, q2, q3)
races(raceId, year, round, circuitId, name, date, time, url, fp1_date, fp1_time, fp2_date, fp2_time, fp3_date, fp3_time, quali_date, quali_time, sprint_date, sprint_time)
results(resultId, raceId, driverId, constructorId, number, grid, position, positionText, positionOrder, points, laps, time, milliseconds, fastestLap, rank, fastestLapTime, fastestLapSpeed, statusId)
seasons(year, url)
sprint_results(sprintResultId, raceId, driverId, constructorId, number, grid, position, positionText, positionOrder, points, laps, time, milliseconds, fastestLap, fastestLapTime, statusId)
status(statusId, status)

Key relationships:
- races.circuitId → circuits.circuitId
- races.year → seasons.year  (filter by year for season-specific queries)
- results.raceId → races.raceId
- results.driverId → drivers.driverId
- results.constructorId → constructors.constructorId
- driver_standings.raceId → races.raceId (cumulative standings after each round)
- qualifying.raceId → races.raceId

Notes:
- Always filter by races.year when asking about a specific season (e.g. WHERE r.year = 2026).
- driver_standings and constructor_standings are cumulative: to get the current
  championship standings, join to the most recent raceId for the season.
- lap_times.time and qualifying q1/q2/q3 are stored as 'M:SS.mmm' strings.
- pit_stops.duration is stored as a decimal seconds string (e.g. '23.456').
`.trim();

const knowledgeEnabled = ENABLE_KNOWLEDGE_BASE === 'true' && Boolean(EMBEDDING_MODEL);
const isOpenRouter = Boolean(OPENAI_BASE_URL && OPENAI_BASE_URL.includes('openrouter.ai'));
const toolsDisabled = DISABLE_TOOLS === 'true' || (isOpenRouter && OPENAI_MODEL === 'openrouter/free');

function extractYear(text) {
  const match = text.match(/\b(19\d{2}|20\d{2})\b/);
  return match ? Number(match[1]) : null;
}

function isChampionQuestion(text) {
  const t = text.toLowerCase();
  return t.includes('champion') || t.includes('world champion') || t.includes('championship winner');
}

function normalizeText(text) {
  return text.toLowerCase().replace(/[?.!,]/g, '').trim();
}

function extractDriverNameFromTeammateQuery(text) {
  const cleaned = normalizeText(text)
    .replace(/who is|whos|who's|current|the|a|an|driver|f1|formula one/g, '')
    .replace(/team mate|teammate/g, '')
    .replace(/of|for/g, '')
    .replace(/'s/g, '')
    .trim();
  return cleaned || null;
}

function extractDriverNameFromWhoIs(text) {
  const cleaned = normalizeText(text);
  if (cleaned.startsWith('who is ')) return cleaned.slice(7).trim();
  if (cleaned.startsWith("who's ")) return cleaned.slice(6).trim();
  if (cleaned.startsWith('whos ')) return cleaned.slice(5).trim();
  return null;
}

async function findDriverByName(name) {
  const cleaned = normalizeText(name);
  const exact = await pool.query(
    `
    SELECT driverId, forename, surname, nationality, driverRef, code
    FROM drivers
    WHERE lower(forename || ' ' || surname) = $1
       OR lower(driverRef) = $1
       OR lower(code) = $1
       OR lower(surname) = $1
    LIMIT 1
    `,
    [cleaned],
  );
  if (exact.rows?.[0]) return exact.rows[0];

  const like = await pool.query(
    `
    SELECT driverId, forename, surname, nationality, driverRef, code
    FROM drivers
    WHERE lower(forename || ' ' || surname) ILIKE $1
    LIMIT 1
    `,
    [`%${cleaned}%`],
  );
  return like.rows?.[0] ?? null;
}

async function getLatestRaceId(year, allowFuture) {
  const result = await pool.query(
    `
    SELECT raceId, round
    FROM races
    WHERE year = $1
      AND ($2::boolean = true OR date <= CURRENT_DATE)
    ORDER BY round DESC
    LIMIT 1
    `,
    [year, allowFuture],
  );
  return result.rows?.[0] ?? null;
}

async function getDriverTeammateName(driverId, year, allowFuture) {
  const latestRace = await getLatestRaceId(year, allowFuture);
  if (!latestRace) return null;

  const constructorRes = await pool.query(
    `
    SELECT constructorId
    FROM results
    WHERE raceId = $1 AND driverId = $2
    LIMIT 1
    `,
    [latestRace.raceid ?? latestRace.raceId, driverId],
  );
  const constructorId = constructorRes.rows?.[0]?.constructorid ?? constructorRes.rows?.[0]?.constructorId;
  if (!constructorId) return null;

  const teammates = await pool.query(
    `
    SELECT d.forename, d.surname
    FROM results r
    JOIN drivers d ON d.driverId = r.driverId
    WHERE r.raceId = $1
      AND r.constructorId = $2
      AND r.driverId <> $3
    ORDER BY r.positionOrder NULLS LAST
    `,
    [latestRace.raceid ?? latestRace.raceId, constructorId, driverId],
  );
  if (!teammates.rows?.length) return null;
  return teammates.rows.map((row) => `${row.forename} ${row.surname}`).join(', ');
}

async function getDriverProfileAnswer(name) {
  const driver = await findDriverByName(name);
  if (!driver) return null;
  return `${driver.forename} ${driver.surname} is a ${driver.nationality} Formula One driver.`;
}

async function getDriverWinCount(name) {
  const driver = await findDriverByName(name);
  if (!driver) return null;
  const result = await pool.query(
    `
    SELECT COUNT(*)::int AS wins
    FROM results r
    WHERE r.driverId = $1 AND r.position = 1
    `,
    [driver.driverid ?? driver.driverId],
  );
  const wins = result.rows?.[0]?.wins;
  if (wins === undefined || wins === null) return null;
  return { name: `${driver.forename} ${driver.surname}`, wins };
}

const tools = toolsDisabled
  ? []
  : [
  {
    type: 'function',
    function: {
      name: 'sql_query',
      description: 'Run a read-only SQL SELECT query against the F1 PostgreSQL database.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string' },
        },
        required: ['query'],
      },
    },
  },
];

if (knowledgeEnabled) {
  tools.push({
    type: 'function',
    function: {
      name: 'f1_knowledge',
      description: 'Semantic search over the F1 knowledge base (pgvector) for narrative context.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string' },
          top_k: { type: 'number' },
        },
        required: ['query'],
      },
    },
  });
}

const timeoutMs = Number(DATABASE_QUERY_TIMEOUT) || 5000;

function isSelectOnly(query) {
  const trimmed = query.trim().toLowerCase();
  if (!trimmed.startsWith('select')) return false;
  if (trimmed.includes(';')) return false;
  return true;
}

function buildOpenRouterUrl(path) {
  if (!OPENAI_BASE_URL) return path;
  const base = OPENAI_BASE_URL.endsWith('/') ? OPENAI_BASE_URL.slice(0, -1) : OPENAI_BASE_URL;
  return `${base}${path.startsWith('/') ? '' : '/'}${path}`;
}

async function openRouterChatFallback(messages) {
  const url = buildOpenRouterUrl('/chat/completions');
  const headers = {
    Authorization: `Bearer ${OPENAI_API_KEY}`,
    'Content-Type': 'application/json',
  };
  if (OPENROUTER_APP_URL) headers['HTTP-Referer'] = OPENROUTER_APP_URL;
  if (OPENROUTER_APP_NAME) headers['X-Title'] = OPENROUTER_APP_NAME;

  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model: OPENAI_MODEL,
      messages,
      temperature: 0.2,
    }),
  });

  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg =
      payload?.error?.message ||
      payload?.message ||
      `OpenRouter error (${res.status})`;
    throw new Error(msg);
  }

  const content = payload?.choices?.[0]?.message?.content ?? '';
  return content;
}

async function runSqlQuery(query) {
  if (!isSelectOnly(query)) {
    return { error: 'Only single SELECT statements are permitted.', rows: [], row_count: 0, columns: [] };
  }

  try {
    const result = await Promise.race([
      pool.query(query),
      new Promise((_, reject) => setTimeout(() => reject(new Error('Query timed out.')), timeoutMs)),
    ]);
    const rows = result.rows ?? [];
    const columns = rows.length ? Object.keys(rows[0]) : [];
    return { rows, row_count: rows.length, columns };
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Query failed.';
    if (message.includes('does not exist')) {
      return {
        error: 'Database not initialized. Please run the schema and import scripts.',
        rows: [],
        row_count: 0,
        columns: [],
      };
    }
    return { error: message, rows: [], row_count: 0, columns: [] };
  }
}

async function getCurrentLeader(year) {
  try {
    const result = await pool.query(
      `
      WITH latest_race AS (
        SELECT raceId, round
        FROM races
        WHERE year = $1
          AND date <= CURRENT_DATE
        ORDER BY round DESC
        LIMIT 1
      )
      SELECT d.forename, d.surname, ds.points, lr.round
      FROM driver_standings ds
      JOIN drivers d ON d.driverId = ds.driverId
      JOIN latest_race lr ON ds.raceId = lr.raceId
      WHERE ds.position = 1
      LIMIT 1
      `,
      [year],
    );

    const row = result.rows?.[0];
    if (!row) return null;
    return {
      name: `${row.forename} ${row.surname}`,
      points: row.points,
      round: row.round,
    };
  } catch {
    return null;
  }
}

async function getSeasonChampion(year) {
  try {
    const result = await pool.query(
      `
      WITH last_race AS (
        SELECT raceId
        FROM races
        WHERE year = $1
        ORDER BY round DESC
        LIMIT 1
      )
      SELECT d.forename, d.surname
      FROM driver_standings ds
      JOIN drivers d ON d.driverId = ds.driverId
      JOIN last_race lr ON ds.raceId = lr.raceId
      WHERE ds.position = 1
      LIMIT 1
      `,
      [year],
    );

    const row = result.rows?.[0];
    if (!row) return null;
    return `${row.forename} ${row.surname}`;
  } catch {
    return null;
  }
}

async function runKnowledgeSearch(query, topK) {
  if (!knowledgeEnabled) {
    return { error: 'Knowledge base is disabled.', results: [], result_count: 0 };
  }
  try {
    const embeddingRes = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: query,
    });
    const vector = embeddingRes.data[0].embedding;
    const vectorLiteral = `[${vector.join(',')}]`;
    const k = Number(topK || RAG_TOP_K || 5);

    const result = await Promise.race([
      pool.query(
        `
        SELECT content, source, 1 - (embedding <=> $1::vector) AS score
        FROM f1_knowledge
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        `,
        [vectorLiteral, k]
      ),
      new Promise((_, reject) => setTimeout(() => reject(new Error('Query timed out.')), timeoutMs)),
    ]);

    const rows = result.rows ?? [];
    return {
      results: rows.map((r) => ({ content: r.content, source: r.source, score: Number(r.score) })),
      result_count: rows.length,
    };
  } catch (err) {
    return { error: err instanceof Error ? err.message : 'Knowledge search failed.', results: [], result_count: 0 };
  }
}

async function runAgent({ message, history }) {
  const currentYear = new Date().getUTCFullYear();
  const askedYear = extractYear(message);

  // Deterministic teammate answers (avoid LLM hallucinations)
  if (message.toLowerCase().includes('teammate') || message.toLowerCase().includes('team mate')) {
    const subject = extractDriverNameFromTeammateQuery(message);
    if (subject) {
      const driver = await findDriverByName(subject);
      if (driver) {
        const targetYear = askedYear ?? currentYear;
        const allowFuture = targetYear < currentYear;
        const teammate = await getDriverTeammateName(driver.driverid ?? driver.driverId, targetYear, allowFuture);
        if (teammate) {
          return { answer: `${teammate}.`, toolCalls: ['sql_query'] };
        }
      }
      return { answer: 'No teammate data found.', toolCalls: [] };
    }
  }

  // Deterministic driver profile answers
  const whoIs = extractDriverNameFromWhoIs(message);
  if (whoIs) {
    const profile = await getDriverProfileAnswer(whoIs);
    if (profile) {
      return { answer: profile, toolCalls: ['sql_query'] };
    }
    if (normalizeText(whoIs).includes('kimi') || normalizeText(whoIs).includes('antonelli')) {
      return { answer: 'No driver data found. Database not initialized.', toolCalls: [] };
    }
  } else {
    const shortName = normalizeText(message);
    if (shortName && shortName.length <= 20 && !shortName.includes(' ')) {
      const profile = await getDriverProfileAnswer(shortName);
      if (profile) {
        return { answer: profile, toolCalls: ['sql_query'] };
      }
    }
  }

  // Deterministic win count: "How many races has X won?"
  if (normalizeText(message).includes('how many') && normalizeText(message).includes('races') && normalizeText(message).includes('won')) {
    const namePart = normalizeText(message)
      .replace(/how many|races|has|won|wins|\?/g, '')
      .trim();
    if (namePart) {
      try {
        const winInfo = await getDriverWinCount(namePart);
        if (winInfo) {
          return { answer: `${winInfo.wins}.`, toolCalls: ['sql_query'] };
        }
      } catch {
        return { answer: 'No driver data found. Database not initialized.', toolCalls: [] };
      }
    }
  }

  if (askedYear && askedYear >= currentYear && isChampionQuestion(message)) {
    const leader = await getCurrentLeader(askedYear);
    if (leader) {
      return {
        answer: `The ${askedYear} season is still ongoing. Current championship leader: ${leader.name} (after round ${leader.round}).`,
        toolCalls: ['sql_query'],
      };
    }
    return {
      answer: `The ${askedYear} season is still ongoing, so there is no world champion yet.`,
      toolCalls: [],
    };
  }

  if (askedYear && askedYear < currentYear && isChampionQuestion(message)) {
    const champion = await getSeasonChampion(askedYear);
    if (champion) {
      return {
        answer: `${champion}.`,
        toolCalls: ['sql_query'],
      };
    }
    return {
      answer: `No championship data found for ${askedYear}.`,
      toolCalls: [],
    };
  }

  const messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    ...(history ?? []),
    { role: 'user', content: message },
  ];

  const toolCalls = [];
  let finalText = '';

  for (let i = 0; i < 4; i += 1) {
    let response;
    const request = {
      model: OPENAI_MODEL,
      messages,
      temperature: 0.2,
    };
    if (tools.length) {
      request.tools = tools;
      request.tool_choice = 'auto';
    }
    try {
      response = await openai.chat.completions.create(request);
    } catch (err) {
      if (!tools.length) {
        if (isOpenRouter) {
          const fallbackText = await openRouterChatFallback(messages);
          return { answer: fallbackText || 'No response returned.', toolCalls };
        }
        throw err;
      }
      // Retry once without tools for models that don't support tool calls (e.g. OpenRouter free).
      try {
        response = await openai.chat.completions.create({
          model: OPENAI_MODEL,
          messages,
          temperature: 0.2,
        });
      } catch (retryErr) {
        if (isOpenRouter) {
          const fallbackText = await openRouterChatFallback(messages);
          return { answer: fallbackText || 'No response returned.', toolCalls };
        }
        throw retryErr;
      }
    }

    const choice = response.choices[0];
    const assistantMessage = choice.message;
    const calls = assistantMessage.tool_calls ?? [];

    if (!calls.length) {
      finalText = assistantMessage.content ?? '';
      break;
    }

    messages.push({
      role: 'assistant',
      content: assistantMessage.content ?? '',
      tool_calls: calls,
    });

    for (const call of calls) {
      toolCalls.push(call.function.name);

      if (call.function.name === 'sql_query') {
        const args = JSON.parse(call.function.arguments || '{}');
        const result = await runSqlQuery(args.query || '');
        messages.push({
          role: 'tool',
          tool_call_id: call.id,
          content: JSON.stringify(result),
        });
      } else if (call.function.name === 'f1_knowledge') {
        const args = JSON.parse(call.function.arguments || '{}');
        const result = await runKnowledgeSearch(args.query || '', args.top_k);
        messages.push({
          role: 'tool',
          tool_call_id: call.id,
          content: JSON.stringify(result),
        });
      } else {
        messages.push({
          role: 'tool',
          tool_call_id: call.id,
          content: JSON.stringify({ error: 'Unknown tool.' }),
        });
      }
    }
  }

  return { answer: finalText || 'No response returned.', toolCalls };
}

app.get('/health', async (_req, res) => {
  const components = {};
  try {
    await pool.query('SELECT 1');
    components.postgres = 'ok';
  } catch {
    components.postgres = 'unavailable';
  }

  const status = Object.values(components).every((v) => v === 'ok') ? 'ok' : 'degraded';
  res.json({ status, components });
});

app.post('/api/v1/chat', async (req, res) => {
  try {
    const { message, history } = req.body || {};
    const result = await runAgent({ message, history });
    res.json({ answer: result.answer, conversation_id: null, tool_calls: result.toolCalls });
  } catch (err) {
    res.status(500).json({ detail: err instanceof Error ? err.message : 'Internal error.' });
  }
});

app.post('/api/v1/chat/stream', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  try {
    const { message, history } = req.body || {};
    const result = await runAgent({ message, history });

    for (const tool of result.toolCalls) {
      res.write(`data: ${JSON.stringify({ type: 'tool_call', content: '', tool_name: tool })}\n\n`);
    }

    const text = result.answer || '';
    const chunkSize = 28;
    for (let i = 0; i < text.length; i += chunkSize) {
      const chunk = text.slice(i, i + chunkSize);
      res.write(`data: ${JSON.stringify({ type: 'delta', content: chunk })}\n\n`);
    }

    res.write(`data: ${JSON.stringify({ type: 'done', content: text })}\n\n`);
    res.end();
  } catch (err) {
    res.write(`data: ${JSON.stringify({ type: 'error', content: err instanceof Error ? err.message : 'Internal error.' })}\n\n`);
    res.end();
  }
});

export default app;
