import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const QUESTIONS_FILE = process.env.QUESTIONS_FILE || path.join(__dirname, '..', 'qa', 'questions_mega.txt');
const OUTPUT_FILE = process.env.OUTPUT_FILE || path.join(__dirname, '..', 'qa', 'answers.json');
const API_BASE = process.env.API_BASE || 'https://harry-s-f1-api-s.vercel.app';
const BATCH_SLEEP_MS = Number(process.env.BATCH_SLEEP_MS || 700);

function parseQuestions(raw) {
  return raw
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line)
    .map((line) => line.replace(/^\d+\.\s*/, '').trim())
    .filter(Boolean);
}

async function callApi(question) {
  const res = await fetch(`${API_BASE}/api/v1/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: question, history: [] }),
  });

  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = payload?.detail || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return payload?.answer ?? payload?.message ?? 'Not enough data.';
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
    let answer = 'Not enough data.';
    let source = 'api';

    try {
      answer = await callApi(question);
    } catch {
      answer = 'Not enough data.';
      source = 'api_error';
    }

    answers.push({ question, answer, source });

    if (answers.length % 20 === 0) {
      await fs.writeFile(OUTPUT_FILE, JSON.stringify(answers, null, 2));
      console.log(`Progress: ${answers.length}/${questions.length}`);
      await new Promise((r) => setTimeout(r, BATCH_SLEEP_MS));
    }
  }

  await fs.writeFile(OUTPUT_FILE, JSON.stringify(answers, null, 2));
  console.log(`Wrote ${answers.length} answers to ${OUTPUT_FILE}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
