import assert from 'node:assert/strict';
import {readFileSync, writeFileSync, mkdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';

const root = new URL('./', import.meta.url);
const summaryBytes = readFileSync(new URL('results/holdout.json', root));
const rawBytes = readFileSync(new URL('results/holdout-rounds.json', root));
const summary = JSON.parse(summaryBytes);
const raw = JSON.parse(rawBytes);
function stats(values) {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const se = Math.sqrt(values.reduce((s, value) => s + (value - mean) ** 2, 0) / (values.length - 1) / values.length);
  return {mean, low: mean - 1.96 * se, high: mean + 1.96 * se};
}
const pairs = [];
for (const room of ['mixed', 'informed', 'prior']) {
  for (const recall of [.25, .5, .65, .85]) {
    const listener = raw.find(run => run.room === room && run.recall === recall && run.flow === 'all');
    const researcher = raw.find(run => run.room === room && run.recall === .65 && run.flow === 'none');
    const keys = ['pnl', 'mse', 'openingPnl', 'closingPnl', 'peakInventory', 'midpointMse'];
    const difference = Object.fromEntries(keys.map(key => [key, stats(listener.rows.map((row, i) => {
      assert.equal(row.value, researcher.rows[i].value);
      assert.equal(row.round, researcher.rows[i].round);
      return row[key] - researcher.rows[i][key];
    }))]));
    pairs.push({room, recall, difference});
  }
}
const data = {
  protocol: summary.protocol,
  hashes: {
    summarySha256: createHash('sha256').update(summaryBytes).digest('hex'),
    roundsSha256: createHash('sha256').update(rawBytes).digest('hex'),
  },
  runs: summary.runs.map(({room, recall, flow, metrics}) => ({room, recall, flow,
    metrics: Object.fromEntries(['pnl','mse','openingPnl','closingPnl','peakInventory','midpointMse','beliefTimeline','pnlByQuote'].map(key => [key, metrics[key]])),
  })),
  contrasts: summary.contrasts,
  pairs,
};
const target = new URL('../../src/data/set-market-study.json', root);
mkdirSync(fileURLToPath(new URL('./', target)), {recursive: true});
writeFileSync(target, JSON.stringify(data, null, 2) + '\n');
const primary = pairs.find(pair => pair.room === 'mixed' && pair.recall === .5);
assert.ok(primary.difference.mse.high < 0);
assert.ok(primary.difference.pnl.high < 0);
console.log('Saved measured study data. Primary paired differences:', primary.difference);
