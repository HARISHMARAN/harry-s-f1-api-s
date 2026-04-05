import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Pool } from 'pg';
import OpenAI from 'openai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const QUESTIONS_FILE = process.env.QUESTIONS_FILE || path.join(__dirname, '..', 'qa', 'questions_2018_2020.txt');
const OUTPUT_FILE = process.env.OUTPUT_FILE || path.join(__dirname, '..', 'qa', 'answers.json');

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL;
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'openrouter/free';

const DATABASE_URL = process.env.DATABASE_URL || process.env.POSTGRES_URL || '';

if (!OPENAI_API_KEY) {
  throw new Error('Missing OPENAI_API_KEY');
}
const useDb = Boolean(DATABASE_URL);

const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
  baseURL: OPENAI_BASE_URL || undefined,
});

const pool = useDb ? new Pool({ connectionString: DATABASE_URL }) : null;

function normalize(text) {
  return text.toLowerCase().replace(/[?.!,]/g, '').trim();
}

function parseQuestions(raw) {
  return raw
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line && !line.toLowerCase().includes('season'))
    .map((line) => line.replace(/^\d+\.\s*/, '').trim())
    .filter(Boolean);
}

async function answerWithLLM(question) {
  const prompt = `Answer the following F1 question in 1-2 concise sentences. If you are not sure, reply exactly: "Not enough data."\n\nQuestion: ${question}`;
  const res = await openai.chat.completions.create({
    model: OPENAI_MODEL,
    messages: [{ role: 'user', content: prompt }],
    temperature: 0.2,
  });
  return res.choices?.[0]?.message?.content?.trim() || 'Not enough data.';
}

async function answerWithSQL(question) {
  if (!useDb) return null;
  const q = normalize(question);

  if (q.startsWith('who is') && q.includes('champion')) {
    const yearMatch = question.match(/\b(19\d{2}|20\d{2})\b/);
    if (!yearMatch) return null;
    const year = Number(yearMatch[1]);
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
    return row ? `${row.forename} ${row.surname}.` : null;
  }

  if (q.includes('how many') && q.includes('races') && (q.includes('won') || q.includes('wins'))) {
    const namePart = q
      .replace('how many', '')
      .replace('races', '')
      .replace('has', '')
      .replace('won', '')
      .replace('wins', '')
      .trim();
    if (!namePart) return null;
    const driver = await pool.query(
      `
      SELECT driverId, forename, surname
      FROM drivers
      WHERE lower(forename || ' ' || surname) = $1
         OR lower(driverRef) = $1
         OR lower(code) = $1
         OR lower(surname) = $1
      LIMIT 1
      `,
      [namePart],
    );
    const d = driver.rows?.[0];
    if (!d) return null;
    const winsRes = await pool.query(
      `
      SELECT COUNT(*)::int AS wins
      FROM results
      WHERE driverId = $1 AND position = 1
      `,
      [d.driverid ?? d.driverId],
    );
    const wins = winsRes.rows?.[0]?.wins;
    return wins !== undefined ? `${wins}.` : null;
  }

  return null;
}

async function main() {
  const raw = await fs.readFile(QUESTIONS_FILE, 'utf8');
  const questions = parseQuestions(raw);

  let answers = [];
  try {
    const existing = await fs.readFile(OUTPUT_FILE, 'utf8');
    const parsed = JSON.parse(existing);
    if (Array.isArray(parsed)) answers = parsed;
  } catch {
    // ignore
  }

  const answeredSet = new Set(answers.map((a) => a.question));

  for (const question of questions) {
    if (answeredSet.has(question)) continue;
    let answer = null;
    let source = 'model';

    try {
      answer = await answerWithSQL(question);
      if (answer) {
        source = 'db';
      } else {
        answer = await answerWithLLM(question);
      }
    } catch {
      answer = 'Not enough data.';
      source = 'unknown';
    }

    answers.push({
      question,
      answer,
      source,
    });

    if (answers.length % 25 === 0) {
      await fs.writeFile(OUTPUT_FILE, JSON.stringify(answers, null, 2));
      console.log(`Progress: ${answers.length}/${questions.length}`);
      await new Promise((r) => setTimeout(r, 600));
    }
  }

  await fs.writeFile(OUTPUT_FILE, JSON.stringify(answers, null, 2));
  console.log(`Wrote ${answers.length} answers to ${OUTPUT_FILE}`);
  if (pool) await pool.end();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
