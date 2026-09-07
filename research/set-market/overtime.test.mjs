import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync, existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {reopenClosingMarket} from './overtime-adapter.mjs';
import {overtimeModel, overtimePath, overtimeX, overtimeY} from '../../src/lib/overtime-study.mjs';
import {quoteOutcome} from '../../src/lib/learning-trade.mjs';

const read = name => readFileSync(new URL(name, import.meta.url));
const data = JSON.parse(read('./results/overtime.json'));
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-11, `${a} != ${b}`);

test('the continuation adapter preserves accounting and evidence-bearing state', () => {
  const state = Object.freeze({phase: 'SETTLEMENT_PENDING', stage: 'closing', version: 9,
    openingOrder: ['b', 'a'], makerIndex: 1, makerOrder: ['a', 'b'],
    boardHash: 'frozen', trades: [{id: 'already-executed'}],
    players: Object.freeze({a: Object.freeze({cashTicks: -10, position: 1, takerFillsRemaining: 1}), b: Object.freeze({cashTicks: 10, position: -1, takerFillsRemaining: 0})}),
  });
  const original = reopenClosingMarket(state, 'original');
  const refill = reopenClosingMarket(state, 'refill');
  assert.equal(state.phase, 'SETTLEMENT_PENDING');
  assert.equal(original.phase, 'QUOTE_OPEN');
  assert.equal(original.players, state.players);
  assert.equal(original.trades, state.trades);
  assert.equal(refill.trades, state.trades);
  assert.deepEqual(refill.makerOrder, ['a', 'b']);
  assert.equal(refill.version, 10);
  for (const id of ['a', 'b']) {
    assert.equal(refill.players[id].cashTicks, state.players[id].cashTicks);
    assert.equal(refill.players[id].position, state.players[id].position);
    assert.equal(refill.players[id].takerFillsRemaining, 4);
  }
  assert.throws(() => reopenClosingMarket({...state, phase: 'SETTLED'}));
  assert.throws(() => reopenClosingMarket({...state, settlement: {value: 8}}));
});

test('the fresh-sample protocol and all graph choices are bounded and labeled correctly', () => {
  assert.equal(data.protocol.rounds, 12000);
  assert.equal(data.protocol.seed, 'set-essay-overtime-v1');
  assert.deepEqual(data.protocol.extras, [0, 4, 8, 16]);
  for (const room of ['mixed', 'informed']) for (const extra of data.protocol.extras) {
    const model = overtimeModel(data, room, extra);
    assert.match(model.caption, new RegExp(`^${extra} extra`));
    for (const series of model.series) {
      assert.ok(!/NaN|undefined|Infinity/.test(series.result + overtimePath(series.points)));
      for (const point of series.points) {
        assert.ok(overtimeX(point.extra) >= 52 && overtimeX(point.extra) <= 512);
        for (const gap of [point.pnl.low, point.pnl.mean, point.pnl.high]) assert.ok(overtimeY(gap) >= 30 && overtimeY(gap) <= 244);
      }
    }
  }
  assert.match(overtimeModel(data, 'mixed', 16).series[0].result, /no clear winner/);
  assert.match(overtimeModel(data, 'mixed', 16).series[1].result, /Listener leads/);
  assert.match(overtimeModel(data, 'informed', 16).series[0].result, /Researcher leads/);
  assert.throws(() => overtimeModel(data, 'prior'));
  assert.throws(() => overtimeModel(data, 'mixed', 12));
});

test('raw trajectory records reproduce paired gaps and improvements independently', {skip: !existsSync(new URL('./results/overtime-rounds.json', import.meta.url))}, () => {
  const bytes = read('./results/overtime-rounds.json');
  assert.equal(createHash('sha256').update(bytes).digest('hex'), data.hashes.roundsSha256);
  const raw = JSON.parse(bytes);
  for (const contrast of data.contrasts) {
    const a = raw.find(run => run.room === contrast.room && run.capacity === contrast.capacity && run.policy === 'listener');
    const b = raw.find(run => run.room === contrast.room && run.capacity === contrast.capacity && run.policy === 'researcher');
    const baseA = raw.find(run => run.room === contrast.room && run.capacity === 'original' && run.policy === 'listener');
    const baseB = raw.find(run => run.room === contrast.room && run.capacity === 'original' && run.policy === 'researcher');
    a.rows.forEach((row, i) => {
      assert.equal(row.value, b.rows[i].value);
      assert.deepEqual(row.snapshots[0], baseA.rows[i].snapshots[0]);
      assert.deepEqual(b.rows[i].snapshots[0], baseB.rows[i].snapshots[0]);
      for (const s of row.snapshots) assert.ok(s.extraTrades >= 0 && s.exhaustedPlayers >= 0 && s.exhaustedPlayers <= 4);
    });
    for (let i = 0; i < 4; i++) for (const metric of ['pnl', 'improvement']) {
      const differences = a.rows.map((row, r) => {
        const gap = row.snapshots[i].pnl - b.rows[r].snapshots[i].pnl;
        return metric === 'pnl' ? gap : gap - (row.snapshots[0].pnl - b.rows[r].snapshots[0].pnl);
      });
      const mean = differences.reduce((sum, x) => sum + x, 0) / differences.length;
      const se = Math.sqrt(differences.reduce((sum, x) => sum + (x - mean) ** 2, 0) / (differences.length - 1) / differences.length);
      close(mean, contrast.deltas[i][metric].mean);
      close(mean - 1.96 * se, contrast.deltas[i][metric].low);
      close(mean + 1.96 * se, contrast.deltas[i][metric].high);
    }
  }
});

test('before overtime, the runner exactly matches the original holdout on a replayed prefix', {skip: !existsSync(new URL('./results/smoke-overtime-rounds.json', import.meta.url)) || !existsSync(new URL('./results/holdout-rounds.json', import.meta.url))}, () => {
  const original = JSON.parse(read('./results/holdout-rounds.json'));
  const replay = JSON.parse(read('./results/smoke-overtime-rounds.json'));
  for (const run of replay) {
    const reference = original.find(row => row.room === run.room && row.recall === (run.policy === 'listener' ? .5 : .65) && row.flow === (run.policy === 'listener' ? 'all' : 'none'));
    run.rows.forEach((row, i) => {
      assert.equal(row.value, reference.rows[i].value);
      assert.equal(row.snapshots[0].pnl, reference.rows[i].pnl);
      assert.equal(row.snapshots[0].mse, reference.rows[i].mse);
      assert.equal(row.snapshots[0].peakInventory, reference.rows[i].peakInventory);
    });
  }
});

test('the playable quote example correctly handles every price, value, and reveal state', () => {
  for (let ask = 3; ask <= 9; ask += .5) {
    const observations = [4, 8].map(value => quoteOutcome(ask, value));
    close((observations[0].pHigh + observations[1].pHigh) / 2, .5);
    for (const value of [4, 8]) {
      const response = quoteOutcome(ask, value);
      assert.equal(response.buys, value > ask);
      assert.equal(response.revealedValue, null);
      assert.equal(response.realizedProfit, null);
      assert.ok(response.mean >= 4 && response.mean <= 8);
      assert.ok(response.expectedProfit <= 0);
      const settled = quoteOutcome(ask, value, true);
      assert.equal(settled.mean, value);
      assert.equal(settled.pHigh, value === 8 ? 1 : 0);
      assert.equal(settled.realizedProfit, value > ask ? ask - value : 0);
      assert.equal(settled.revealedValue, value);
    }
  }
  assert.equal(quoteOutcome(6, 8).mean, 8);
  assert.equal(quoteOutcome(6, 8).expectedProfit, -2);
  assert.equal(quoteOutcome(6, 4).mean, 4);
  assert.equal(quoteOutcome(6, 4).expectedProfit, 0);
  assert.equal(quoteOutcome(3, 8).pHigh, .5);
  assert.equal(quoteOutcome(8, 8).pHigh, .5);
  assert.throws(() => quoteOutcome(6.1, 4));
  assert.throws(() => quoteOutcome(6, 7));
});
