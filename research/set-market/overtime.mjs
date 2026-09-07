/**
 * Follow-up protocol fixed before looking at results: extend trading by 0, 4,
 * 8, and 16 quote windows without additional research. Repeat with the original
 * taker cap and with four taker fills restored before each extra four-window
 * block. Same policies, beliefs, cash, inventory, and frozen board continue.
 * node research/set-market/overtime.mjs /path/to/set-mm-game overtime 12000
 */
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve, join} from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
import {reopenClosingMarket} from './overtime-adapter.mjs';

const gamePath = resolve(process.argv[2]);
const runName = process.argv[3] || 'overtime';
const rounds = Number(process.argv[4] || 12000);
assert.match(runName, /^[a-z0-9-]+$/);
assert.ok(Number.isInteger(rounds) && rounds >= 2);
const gameCommit = '69153fc743f5fd799d7e075c51e38ec012c356e8';
assert.equal(execFileSync('git', ['rev-parse', 'HEAD'], {cwd: gamePath, encoding: 'utf8'}).trim(), gameCommit);
assert.equal(execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], {cwd: gamePath, encoding: 'utf8'}).trim(), '');
const gameImport = path => import(pathToFileURL(join(gamePath, 'dist/src', path)));
const {createAgent} = await gameImport('bot/agent.js');
const {startResearch, continueResearch} = await gameImport('bot/research.js');
const {createRound, startOpeningMarket, startClosingMarket, getCurrentMaker, submitQuote, submitResponse, closeResponseWindow, settleRound} = await gameImport('engine/round.js');
const {generateOracleBoard} = await gameImport('sim/boards.js');
const {createRandom} = await gameImport('sim/random.js');
// Reuse the original independent calibration, not any evaluation-board counts.
const prior = JSON.parse(readFileSync(new URL('./results/holdout.json', import.meta.url))).prior;
const root = createRandom(process.env.SET_OVERTIME_SEED || `set-essay-${runName}-v1`);
const boardRandom = root.fork('boards');
const boards = Array.from({length: rounds}, () => generateOracleBoard(boardRandom));
const ids = ['target', 'a', 'b', 'c'];
const checkpoints = [8, 12, 16, 24];
const rooms = {mixed: ['informed', 'informed', 'prior'], informed: ['informed', 'informed', 'informed']};
const selectedRooms = (process.env.SET_OVERTIME_ROOMS || 'mixed,informed').split(',');
selectedRooms.forEach(room => assert.ok(rooms[room]));

function moments(values) {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const se = Math.sqrt(values.reduce((s, x) => s + (x - mean) ** 2, 0) / (n - 1) / n);
  return {n, mean, low: mean - 1.96 * se, high: mean + 1.96 * se};
}

function run(room, capacity, policy) {
  const strategies = ['smart', ...rooms[room]];
  const agents = ids.map((id, i) => createAgent(id, strategies[i], i === 0 ? {openingRecall: policy === 'listener' ? .5 : .65} : {}));
  if (policy === 'researcher') agents[0].observeTrade = () => {};
  const rows = [];
  for (let r = 0; r < rounds; r++) {
    const oracle = boards[r];
    const research = agents.map((agent, i) => startResearch(oracle.value, agent.config.openingRecall, root.fork(`research:${r}:${ids[i]}:opening`)));
    agents.forEach((agent, i) => agent.beginRound(prior.probabilities, research[i]));
    let state = startOpeningMarket(createRound({roundId: `r${r}`, boardHash: oracle.board.hash, playerIds: ids, openingSeat: r % 4}));
    let q = 0;
    const playMarket = () => {
      while (state.phase === 'QUOTE_OPEN') {
        const makerId = getCurrentMaker(state);
        const maker = agents[ids.indexOf(makerId)];
        const decision = maker.chooseQuote({stage: state.stage, ledger: state.players[makerId], opponentCount: 3}, root.fork(`quote:${r}:${q}:${makerId}`));
        state = submitQuote(state, {quoteId: `r${r}q${q}`, makerId, ...decision});
        const quote = state.quote;
        agents.forEach((agent, i) => {
          if (ids[i] === makerId) return;
          const action = agent.chooseAction({quote, ledger: state.players[ids[i]]}, root.fork(`action:${r}:${q}:${ids[i]}`));
          state = submitResponse(state, {playerId: ids[i], action, requestId: `r${r}q${q}:${ids[i]}`});
        });
        const previousTrades = state.trades.length;
        state = closeResponseWindow(state, root.fork(`fill:${r}:${q}`).next);
        for (const trade of state.trades.slice(previousTrades)) {
          for (const agent of agents) agent.observeTrade(trade, strategies[ids.indexOf(trade.takerId)]);
        }
        q++;
      }
    };
    playMarket();
    assert.equal(q, 4);
    assert.equal(state.phase, 'SECOND_RESEARCH');
    agents.forEach((agent, i) => {
      research[i] = continueResearch(oracle.value, research[i], agent.config.closingRecall, root.fork(`research:${r}:${ids[i]}:closing`));
      agent.addResearch(research[i]);
    });
    state = startClosingMarket(state);
    playMarket();
    const snapshots = [];
    const baseTrades = state.trades.length;
    for (let block = 0; block <= 4; block++) {
      assert.equal(state.phase, 'SETTLEMENT_PENDING');
      assert.equal(q, 8 + block * 4);
      if (checkpoints.includes(q)) {
        // Settle an immutable COPY for scoring only. Original state and agents
        // receive neither this result nor the oracle value; trading continues.
        const forecast = agents[0].snapshot();
        const evaluation = settleRound(state, oracle.value);
        assert.equal(evaluation.settlement.totalScoreTicks, 0);
        assert.equal(state.settlement, undefined);
        const ledger = evaluation.players.target;
        snapshots.push({extra: q - 8, pnl: ledger.scoreTicks / 2,
          mse: (forecast.mean - oracle.value) ** 2,
          trades: state.trades.length, extraTrades: state.trades.length - baseTrades,
          peakInventory: ledger.peakAbsoluteInventory,
          exhaustedPlayers: Object.values(state.players).filter(player => player.takerFillsRemaining === 0).length,
        });
      }
      if (block < 4) {
        state = reopenClosingMarket(state, capacity);
        playMarket();
      }
    }
    rows.push({round: r, value: oracle.value, snapshots});
  }
  const metrics = checkpoints.map((q, i) => ({extra: q - 8, ...Object.fromEntries(
    ['pnl', 'mse', 'trades', 'extraTrades', 'peakInventory', 'exhaustedPlayers'].map(key => [key, moments(rows.map(row => row.snapshots[i][key]))]),
  )}));
  console.log(`${room} ${capacity} ${policy}: ${metrics.map(row => `+${row.extra}: ${row.pnl.mean.toFixed(4)} pnl`).join('; ')}`);
  return {room, capacity, policy, metrics, rows};
}

const runs = selectedRooms.flatMap(room => ['original', 'refill'].flatMap(capacity => ['listener', 'researcher'].map(policy => run(room, capacity, policy))));
const contrasts = [];
for (const room of selectedRooms) for (const capacity of ['original', 'refill']) {
  const a = runs.find(run => run.room === room && run.capacity === capacity && run.policy === 'listener');
  const b = runs.find(run => run.room === room && run.capacity === capacity && run.policy === 'researcher');
  const deltas = checkpoints.map((q, i) => {
    const values = a.rows.map((row, r) => {
      assert.equal(row.value, b.rows[r].value);
      const baseDifference = row.snapshots[0].pnl - b.rows[r].snapshots[0].pnl;
      return {pnl: row.snapshots[i].pnl - b.rows[r].snapshots[i].pnl,
        mse: row.snapshots[i].mse - b.rows[r].snapshots[i].mse,
        improvement: row.snapshots[i].pnl - b.rows[r].snapshots[i].pnl - baseDifference};
    });
    return {extra: q - 8, ...Object.fromEntries(['pnl', 'mse', 'improvement'].map(key => [key, moments(values.map(row => row[key]))]))};
  });
  contrasts.push({room, capacity, deltas});
  // Both cap treatments must be bit-identical before the extension begins.
  const original = runs.find(run => run.room === room && run.capacity === 'original' && run.policy === 'listener');
  a.rows.forEach((row, i) => assert.deepEqual(row.snapshots[0], original.rows[i].snapshots[0]));
  const originalB = runs.find(run => run.room === room && run.capacity === 'original' && run.policy === 'researcher');
  b.rows.forEach((row, i) => assert.deepEqual(row.snapshots[0], originalB.rows[i].snapshots[0]));
}
const raw = JSON.stringify(runs.map(({room, capacity, policy, rows}) => ({room, capacity, policy, rows}))) + '\n';
const output = {
  protocol: {gameCommit, seed: root.seed, rounds, priorSeed: 'set-essay-prior-v1', priorSamples: prior.samples,
    extras: checkpoints.map(q => q - 8), rooms,
    change: 'Research-only adapter resumes the unchanged engine for extra closing blocks. No private research, settlement revelation, cash reset, inventory reset, or policy retuning. Completed trades still update Listener.',
    capacity: {original: 'Original four-taker-fill allowance lasts the whole extended round', refill: 'Before every extra four-window block, restore every player to four taker fills; unused allowances do not accumulate'},
    ordering: 'Every extra four-window block repeats the normal closing order. Opening seats still rotate across rounds.',
    evaluation: 'Endpoints share trajectory prefixes: these myopic policies do not use the announced horizon. Zero-sum evaluation occurs on immutable copies invisible to agents.',
    interval: 'Pointwise mean +/- 1.96 sample SE of within-board differences; no multiplicity correction. Improvement is a paired difference-in-differences relative to zero extra windows.',
  },
  hashes: {roundsSha256: createHash('sha256').update(raw).digest('hex')},
  runs: runs.map(({rows, ...run}) => run), contrasts,
};
const directory = new URL('./results/', import.meta.url);
mkdirSync(fileURLToPath(directory), {recursive: true});
writeFileSync(new URL(`${runName}.json`, directory), JSON.stringify(output, null, 2) + '\n');
writeFileSync(new URL(`${runName}-rounds.json`, directory), raw);
console.log(`Saved ${runName}: ${rounds} fresh paired boards, ${runs.length} trajectories per board.`);
