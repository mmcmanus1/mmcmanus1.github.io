import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync, readdirSync, existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {comparison, comparisonX, timeline, timelinePath, timeX, errorY} from '../../src/lib/set-study.mjs';
import {buyerPosterior} from '../../src/lib/explainers.mjs';

const read = path => readFileSync(new URL(path, import.meta.url));
const study = JSON.parse(read('../../src/data/set-market-study.json'));
const aggregate = JSON.parse(read('./results/holdout.json'));
const close = (a, b, tolerance = 1e-12) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);

test('figure data are traceable to the untouched holdout aggregates', () => {
  assert.equal(study.hashes.summarySha256, createHash('sha256').update(read('./results/holdout.json')).digest('hex'));
  assert.equal(study.protocol.rounds, 12000);
  assert.equal(study.protocol.seed, 'set-essay-holdout-v1');
  assert.equal(study.runs.length, 24);
  for (const run of study.runs) {
    const source = aggregate.runs.find(other => other.room === run.room && other.recall === run.recall && other.flow === run.flow);
    for (const key of Object.keys(run.metrics)) assert.deepEqual(run.metrics[key], source.metrics[key]);
  }
});

test('paired intervals independently reproduce from the retained per-board records', {skip: !existsSync(new URL('./results/holdout-rounds.json', import.meta.url))}, () => {
  const bytes = read('./results/holdout-rounds.json');
  assert.equal(createHash('sha256').update(bytes).digest('hex'), study.hashes.roundsSha256);
  const raw = JSON.parse(bytes);
  for (const pair of study.pairs) {
    const a = raw.find(run => run.room === pair.room && run.recall === pair.recall && run.flow === 'all');
    const b = raw.find(run => run.room === pair.room && run.recall === .65 && run.flow === 'none');
    assert.equal(a.rows.length, 12000);
    for (const key of ['pnl', 'mse']) {
      const differences = a.rows.map((row, i) => {
        assert.equal(row.value, b.rows[i].value);
        return row[key] - b.rows[i][key];
      });
      const mean = differences.reduce((sum, x) => sum + x, 0) / differences.length;
      const standardError = Math.sqrt(differences.reduce((sum, x) => sum + (x - mean) ** 2, 0) / (differences.length - 1) / differences.length);
      close(mean, pair.difference[key].mean);
      close(mean - 1.96 * standardError, pair.difference[key].low);
      close(mean + 1.96 * standardError, pair.difference[key].high);
    }
  }
});

test('primary difference, timing decomposition, and headline rounding agree', () => {
  const {listener: a, researcher: b, pair} = comparison(study);
  assert.equal(a.metrics.mse.mean.toFixed(3), '0.537');
  assert.equal(b.metrics.mse.mean.toFixed(3), '0.575');
  assert.equal(a.metrics.pnl.mean.toFixed(3), '0.541');
  assert.equal(b.metrics.pnl.mean.toFixed(3), '0.710');
  for (const key of ['mse', 'pnl']) {
    close(pair.difference[key].mean, a.metrics[key].mean - b.metrics[key].mean);
    assert.ok(pair.difference[key].high < 0);
  }
  close(pair.difference.openingPnl.mean + pair.difference.closingPnl.mean, pair.difference.pnl.mean);
  for (const run of [a, b]) {
    close(run.metrics.openingPnl.mean + run.metrics.closingPnl.mean, run.metrics.pnl.mean);
    close(run.metrics.pnlByQuote.reduce((sum, x) => sum + x.mean, 0), run.metrics.pnl.mean);
    close(run.metrics.beliefTimeline[9].mean, run.metrics.mse.mean);
  }
  // Index 5 is the second research pass; index 8 follows quote window 7.
  assert.ok(timeline(study, 7).values[0] > timeline(study, 7).values[1]);
  assert.ok(timeline(study, 8).values[0] < timeline(study, 8).values[1]);
});

test('every selectable comparison has finite, bounded, correctly ordered geometry', () => {
  for (const room of ['mixed', 'informed', 'prior']) {
    for (const recall of [.25, .5, .65, .85]) {
      const state = comparison(study, room, recall);
      for (const chart of state.charts) {
        assert.ok(chart.max > chart.min);
        for (const value of chart.values) {
          assert.ok(value.low <= value.mean && value.mean <= value.high);
          for (const number of [value.low, value.mean, value.high]) {
            const x = comparisonX(number, chart);
            assert.ok(Number.isFinite(x) && x >= 42 && x <= 512);
          }
        }
        chart.ticks.forEach((tick, index) => close(comparisonX(tick, chart), 42 + index * 470 / 4));
      }
      assert.ok(!/undefined|NaN/.test(state.result + state.interval));
    }
  }
  assert.throws(() => comparison(study, 'not-a-roster'));
});

test('the control settings change the conclusion rather than preserving a slogan', () => {
  const sameResearch = comparison(study, 'mixed', .65);
  assert.ok(sameResearch.pair.difference.pnl.low > 0);
  assert.ok(sameResearch.pair.difference.mse.high < 0);
  const priorOnly = comparison(study, 'prior', .5);
  assert.ok(priorOnly.pair.difference.mse.low > 0);
  const allResearchers = comparison(study, 'informed', .5);
  assert.ok(allResearchers.pair.difference.mse.high < 0);
  assert.ok(allResearchers.pair.difference.pnl.high < 0);
});

test('every timeline step is finite and the research-only forecast stays flat between searches', () => {
  const {listener, researcher} = comparison(study);
  for (const run of [listener, researcher]) {
    assert.ok(!/NaN|Infinity/.test(timelinePath(run.metrics.beliefTimeline)));
  }
  for (let i = 0; i < 10; i++) {
    assert.ok(timeX(i) >= 52 && timeX(i) <= 512);
    for (const value of timeline(study, i).values) assert.ok(errorY(value) >= 30 && errorY(value) <= 244);
  }
  for (let i = 1; i <= 4; i++) close(researcher.metrics.beliefTimeline[i].mean, researcher.metrics.beliefTimeline[0].mean);
  for (let i = 6; i <= 9; i++) close(researcher.metrics.beliefTimeline[i].mean, researcher.metrics.beliefTimeline[5].mean);
  assert.throws(() => timeline(study, -1));
  assert.throws(() => timeline(study, 10));
});

test('the analytic filled-sale example agrees with direct Bayes enumeration', () => {
  for (let i = 0; i <= 100; i++) {
    const alpha = i / 100;
    const highBuy = .5 * (alpha + (1 - alpha) / 2);
    const lowBuy = .5 * (1 - alpha) / 2;
    const probabilityHigh = highBuy / (highBuy + lowBuy);
    close(buyerPosterior(alpha), probabilityHigh);
    close(6 - (8 * probabilityHigh + 4 * (1 - probabilityHigh)), -2 * alpha);
  }
});

test('the article stays a draft and production contains neither it nor its study data', () => {
  const article = read('../../src/content/blog/a-market-hidden-in-a-card-game.mdx').toString();
  assert.match(article, /^draft: true$/m);
  const dist = new URL('../../dist/', import.meta.url);
  assert.ok(existsSync(dist), 'Run npm run build before the release guard');
  assert.ok(!existsSync(new URL('blog/a-market-hidden-in-a-card-game/index.html', dist)));
  const walk = url => readdirSync(url, {withFileTypes: true}).flatMap(entry => {
    const child = new URL(entry.name + (entry.isDirectory() ? '/' : ''), url);
    return entry.isDirectory() ? walk(child) : [child];
  });
  for (const file of walk(dist).filter(url => /\.(?:html|[cm]?js|css|json|xml|txt|md|map)$/.test(url.pathname))) {
    assert.doesNotMatch(readFileSync(file).toString(), /The Price of Learning Too Late|set-essay-holdout-v1|set-essay-overtime-v1|69153fc743f5fd799d7e075c51e38ec012c356e8|0\.5410833333333334/);
  }
});
