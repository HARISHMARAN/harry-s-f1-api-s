import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const OUTPUT_FILE = process.env.OUTPUT_FILE || path.join(__dirname, '..', 'qa', 'questions_mega.txt');
const TARGET_COUNT = Number(process.env.TARGET_COUNT || 2000);

const seasons = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026];

const races = [
  'Australian GP', 'Bahrain GP', 'Chinese GP', 'Azerbaijan GP', 'Spanish GP', 'Monaco GP',
  'Canadian GP', 'French GP', 'Austrian GP', 'British GP', 'German GP', 'Hungarian GP',
  'Belgian GP', 'Italian GP', 'Singapore GP', 'Russian GP', 'Japanese GP', 'United States GP',
  'Mexican GP', 'Brazilian GP', 'Abu Dhabi GP', 'Saudi Arabian GP', 'Emilia Romagna GP',
  'Dutch GP', 'Qatar GP', 'Las Vegas GP'
];

const teams = [
  'Mercedes', 'Ferrari', 'Red Bull', 'McLaren', 'Aston Martin', 'Alpine',
  'Williams', 'Haas', 'AlphaTauri', 'Racing Point', 'Sauber'
];

const drivers = [
  'Lewis Hamilton', 'Max Verstappen', 'Sebastian Vettel', 'Valtteri Bottas',
  'Charles Leclerc', 'Lando Norris', 'Sergio Perez', 'Daniel Ricciardo',
  'Carlos Sainz', 'George Russell', 'Kimi Raikkonen', 'Fernando Alonso',
  'Esteban Ocon', 'Pierre Gasly', 'Alex Albon', 'Oscar Piastri'
];

const templates = [
  (s, r) => `How did ${r} ${s} highlight the importance of tyre management?`,
  (s, r) => `What strategic decision defined the outcome of the ${r} ${s}?`,
  (s, r) => `Why did ${r} ${s} favor one team over another?`,
  (s, r) => `How did safety car timing influence the ${r} ${s} result?`,
  (s, r) => `What setup compromise was most important at the ${r} ${s}?`,
  (s, r) => `How did weather (or lack of it) shape the ${r} ${s}?`,
  (s, r) => `Why did overtaking feel difficult at the ${r} ${s}?`,
  (s, r) => `Which strategy would have won the ${r} ${s} with no pit stops?`,
  (s, r) => `How did ${r} ${s} expose reliability weaknesses?`,
  (s, r) => `What made the ${r} ${s} a turning point in the ${s} season?`,
  (s, r) => `How did DRS usage affect the ${r} ${s} outcome?`,
  (s, r) => `What role did track position play in the ${r} ${s}?`,
  (s, r) => `Why did the ${r} ${s} reward aggressive driving?`,
  (s, r) => `What made the ${r} ${s} strategically complex?`,
  (s, r) => `How did pit stop timing influence the ${r} ${s}?`,
  (s, r) => `Why did the ${r} ${s} amplify car strengths more than other races?`,
  (s, r) => `If the ${r} ${s} had rain, who benefits most and why?`,
  (s, r) => `What was the biggest tactical mistake at the ${r} ${s}?`,
  (s, r) => `How did qualifying results shape the ${r} ${s}?`,
  (s, r) => `Why did the ${r} ${s} feel like a momentum swing race?`,

  (s, r, t) => `Why did ${t} look stronger than rivals at the ${r} ${s}?`,
  (s, r, t) => `How did ${t}'s upgrades affect performance at the ${r} ${s}?`,
  (s, r, t) => `What problem limited ${t} at the ${r} ${s}?`,
  (s, r, t) => `How did ${t} adapt strategy during the ${r} ${s}?`,
  (s, r, t) => `What did the ${r} ${s} reveal about ${t}'s car concept?`,

  (s, r, d) => `What was the key to ${d}'s performance at the ${r} ${s}?`,
  (s, r, d) => `How did ${d} manage tyres at the ${r} ${s}?`,
  (s, r, d) => `What decision by ${d} mattered most at the ${r} ${s}?`,
  (s, r, d) => `Why did ${d} outperform teammates at the ${r} ${s}?`,
  (s, r, d) => `If ${d} had started on a different tyre, how would the ${r} ${s} change?`,

  (s) => `How did regulation changes affect performance trends in ${s}?`,
  (s) => `What was the biggest strategic trend of the ${s} season?`,
  (s) => `Which team maximized development during ${s}?`,
  (s) => `Which driver adapted best to the ${s} regulations?`,
  (s) => `What defined the championship narrative in ${s}?`,
  (s) => `How did tyre compounds shape the ${s} season?`,
  (s) => `Why did consistency matter more than raw pace in ${s}?`,
  (s) => `If all cars were equal in ${s}, who wins the title?`,
  (s) => `What was the most influential upgrade of ${s}?`,
  (s) => `Why did ${s} feel like a transition year?`,

  (s, d) => `How did ${d} influence the narrative of the ${s} season?`,
  (s, d) => `What was ${d}'s defining race in ${s}?`,
  (s, d) => `Why did ${d} struggle or shine in ${s}?`,

  (s) => `If ${s} had reverse grids, how would the top 3 change?`,
  (s) => `If refuelling returned in ${s}, what strategies change?`,
  (s) => `How would banning DRS have changed ${s}?`,
  (s) => `If team orders were banned in ${s}, who benefits most?`,

  (s) => `How did the cost cap influence team strategies in ${s}?`,
  (s) => `What role did simulator correlation play in ${s}?`,
  (s) => `How did aero correlation issues affect upgrades in ${s}?`,
  (s) => `What role did pit crew performance play in ${s}?`,

  (s) => `Why do fans remember chaos more than dominance in ${s}?`,
  (s) => `Which race in ${s} best balanced overtaking and strategy?`,
  (s) => `Which race in ${s} was secretly a strategic masterpiece?`,
  (s) => `Which driver improved the most across ${s}?`,
  (s) => `What was the biggest strategic blunder of ${s}?`,
];

function seededRandom(seed) {
  let x = seed % 2147483647;
  if (x <= 0) x += 2147483646;
  return () => (x = (x * 16807) % 2147483647) / 2147483647;
}

async function main() {
  const rand = seededRandom(20260405);
  const questions = [];

  let seasonIndex = 0;
  while (questions.length < TARGET_COUNT) {
    const season = seasons[seasonIndex % seasons.length];
    const race = races[Math.floor(rand() * races.length)];
    const team = teams[Math.floor(rand() * teams.length)];
    const driver = drivers[Math.floor(rand() * drivers.length)];

    const template = templates[Math.floor(rand() * templates.length)];
    let q = template.length === 3
      ? template(season, race, rand() > 0.5 ? team : driver)
      : template.length === 2
        ? template(season, rand() > 0.5 ? race : driver)
        : template(season, race);

    if (!q || typeof q !== 'string') {
      q = `What defined the ${season} season?`;
    }

    if (!questions.includes(q)) {
      questions.push(q);
    }

    seasonIndex += 1;
  }

  const lines = questions.map((q, i) => `${i + 1}. ${q}`);
  await fs.writeFile(OUTPUT_FILE, lines.join('\n'));
  console.log(`Wrote ${lines.length} questions to ${OUTPUT_FILE}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
