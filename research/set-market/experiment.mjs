/**
 * Local-only research for the SET // MARKET essay. Does not change the game.
 * Build mmcmanus1/set-mm-game at GAME_COMMIT, then:
 * node research/set-market/experiment.mjs /path/to/set-mm-game pilot 600
 * Outputs live under research/, never public/. Paired runs share exogenous RNGs.
 */
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdirSync, writeFileSync} from 'node:fs';
import {resolve, join} from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';

export const GAME_COMMIT = '69153fc743f5fd799d7e075c51e38ec012c356e8';
const gamePath = resolve(process.argv[2] || '../set-mm-game');
const runName = process.argv[3] || 'pilot';
const rounds = Number(process.argv[4] || 600);
assert.match(runName, /^[a-z0-9-]+$/);
assert.ok(Number.isInteger(rounds) && rounds >= 2);
assert.equal(execFileSync('git', ['rev-parse', 'HEAD'], {cwd: gamePath, encoding: 'utf8'}).trim(), GAME_COMMIT);
assert.equal(execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], {cwd: gamePath, encoding: 'utf8'}).trim(), '', 'Game tracked source must be clean');
const gameImport = path => import(pathToFileURL(join(gamePath, 'dist/src', path)));
const {createAgent} = await gameImport('bot/agent.js');
const {calibratePrior, updateFromResearch, summarize} = await gameImport('bot/belief.js');
const {startResearch, continueResearch} = await gameImport('bot/research.js');
const {createRound, startOpeningMarket, startClosingMarket, getCurrentMaker, submitQuote, submitResponse, closeResponseWindow, settleRound} = await gameImport('engine/round.js');
const {generateOracleBoard} = await gameImport('sim/boards.js');
const {createRandom} = await gameImport('sim/random.js');

// One independently seeded calibration, shared across every evaluation run.
const priorSeed = 'set-essay-prior-v1';
const prior = calibratePrior(50000, createRandom(priorSeed));
const root = createRandom(`set-essay-${runName}-v1`);
const boardRandom = root.fork('boards');
const boards = Array.from({length: rounds}, () => generateOracleBoard(boardRandom));
const ids = ['target', 'a', 'b', 'c'];
const rooms = {
  mixed: ['informed', 'informed', 'prior'],
  informed: ['informed', 'informed', 'informed'],
  prior: ['prior', 'prior', 'prior'],
  smart: ['smart', 'smart', 'smart'],
};

function moments(values) {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1);
  const se = Math.sqrt(variance / n);
  return {n, mean, se, low: mean - 1.96 * se, high: mean + 1.96 * se};
}

function run({room, recall, flow}) {
  const strategies = ['smart', ...rooms[room]];
  const agents = ids.map((id, i) => createAgent(id, strategies[i], i === 0 ? {openingRecall: recall} : {}));
  const target = agents[0];
  const originalObserve = target.observeTrade.bind(target);
  target.observeTrade = (trade, strategy) => {
    if (flow === 'none') return;
    if (flow === 'informed-only' && strategy !== 'informed') return;
    originalObserve(trade, strategy);
  };
  const rows = [];
  for (let r = 0; r < rounds; r++) {
    const oracle = boards[r];
    const research = agents.map((agent, i) => startResearch(oracle.value, agent.config.openingRecall, root.fork(`research:${r}:${ids[i]}:opening`)));
    agents.forEach((agent, i) => agent.beginRound(prior.probabilities, research[i]));
    let state = startOpeningMarket(createRound({roundId: `r${r}`, boardHash: oracle.board.hash, playerIds: ids, openingSeat: r % ids.length}));
    let q = 0;
    let flowTrades = 0;
    let targetMidpointSquaredError = 0;
    const beliefTimeline = [(target.snapshot().mean - oracle.value) ** 2];
    const pnlByQuote = [];
    let openingPnl = 0;
    let closingPnl = 0;
    let openingMakerPnl = 0;
    let closingMakerPnl = 0;
    const playMarket = () => {
      while (state.phase === 'QUOTE_OPEN') {
        const makerId = getCurrentMaker(state);
        const maker = agents[ids.indexOf(makerId)];
        const decision = maker.chooseQuote({stage: state.stage, ledger: state.players[makerId], opponentCount: 3}, root.fork(`quote:${r}:${q}:${makerId}`));
        state = submitQuote(state, {quoteId: `r${r}q${q}`, makerId, ...decision});
        if (makerId === 'target') targetMidpointSquaredError += ((decision.bidTicks + decision.askTicks) / 4 - oracle.value) ** 2 / 2;
        const quote = state.quote;
        agents.forEach((agent, i) => {
          if (ids[i] === makerId) return;
          const action = agent.chooseAction({quote, ledger: state.players[ids[i]]}, root.fork(`action:${r}:${q}:${ids[i]}`));
          state = submitResponse(state, {playerId: ids[i], action, requestId: `r${r}q${q}:${ids[i]}`});
        });
        const previousTrades = state.trades.length;
        state = closeResponseWindow(state, root.fork(`fill:${r}:${q}`).next);
        let quotePnl = 0;
        for (const trade of state.trades.slice(previousTrades)) {
          const takerPnl = trade.takerAction === 'BUY' ? oracle.value - trade.priceTicks / 2 : trade.priceTicks / 2 - oracle.value;
          const ourPnl = trade.takerId === 'target' ? takerPnl : trade.makerId === 'target' ? -takerPnl : 0;
          quotePnl += ourPnl;
          if (q < 4) {
            openingPnl += ourPnl;
            if (trade.makerId === 'target') openingMakerPnl += ourPnl;
          } else {
            closingPnl += ourPnl;
            if (trade.makerId === 'target') closingMakerPnl += ourPnl;
          }
          if (trade.takerId !== 'target') flowTrades++;
          for (const agent of agents) agent.observeTrade(trade, strategies[ids.indexOf(trade.takerId)]);
        }
        pnlByQuote.push(quotePnl);
        beliefTimeline.push((target.snapshot().mean - oracle.value) ** 2);
        q++;
      }
    };
    playMarket();
    assert.equal(state.phase, 'SECOND_RESEARCH');
    agents.forEach((agent, i) => {
      research[i] = continueResearch(oracle.value, research[i], agent.config.closingRecall, root.fork(`research:${r}:${ids[i]}:closing`));
      agent.addResearch(research[i]);
    });
    beliefTimeline.push((target.snapshot().mean - oracle.value) ** 2);
    state = startClosingMarket(state);
    playMarket();
    assert.equal(state.phase, 'SETTLEMENT_PENDING');
    // Beliefs are measured before settlement. No oracle value enters an agent.
    const snapshot = target.snapshot();
    const privateOnly = summarize(updateFromResearch(prior.probabilities, research[0].discoveredCount, research[0].cumulativeRecall));
    state = settleRound(state, oracle.value);
    assert.equal(Object.values(state.players).reduce((sum, player) => sum + player.scoreTicks, 0), 0);
    assert.equal(q, 8);
    const ledger = state.players.target;
    assert.equal(openingPnl + closingPnl, ledger.scoreTicks / 2);
    assert.equal(openingMakerPnl + closingMakerPnl, ledger.makerScoreTicks / 2);
    rows.push({
      round: r, value: oracle.value, discovered: research[0].discoveredCount,
      pnl: ledger.scoreTicks / 2, makerPnl: ledger.makerScoreTicks / 2,
      mse: (snapshot.mean - oracle.value) ** 2,
      privateMse: (privateOnly.mean - oracle.value) ** 2,
      mae: Math.abs(snapshot.mean - oracle.value), variance: snapshot.variance,
      midpointMse: targetMidpointSquaredError, peakInventory: ledger.peakAbsoluteInventory,
      flowTrades, evidence: snapshot.evidenceCount,
      openingPnl, closingPnl, openingMakerPnl, closingMakerPnl, beliefTimeline, pnlByQuote,
    });
  }
  const metrics = Object.fromEntries(['pnl','makerPnl','mse','privateMse','mae','variance','midpointMse','peakInventory','flowTrades','evidence','openingPnl','closingPnl','openingMakerPnl','closingMakerPnl'].map(key => [key, moments(rows.map(row => row[key]))]));
  metrics.beliefTimeline = Array.from({length: 10}, (_, i) => moments(rows.map(row => row.beliefTimeline[i])));
  metrics.pnlByQuote = Array.from({length: 8}, (_, i) => moments(rows.map(row => row.pnlByQuote[i])));
  return {room, recall, flow, metrics, rows};
}

const selectedRooms = (process.env.SET_ESSAY_ROOMS || 'mixed,informed,prior').split(',');
const recalls = (process.env.SET_ESSAY_RECALLS || '0,0.25,0.5,0.65,0.85,1').split(',').map(Number);
const flows = (process.env.SET_ESSAY_FLOWS || 'none,all').split(',');
selectedRooms.forEach(room => assert.ok(rooms[room]));
recalls.forEach(recall => assert.ok(recall >= 0 && recall <= 1));
flows.forEach(flow => assert.ok(['none', 'all', 'informed-only'].includes(flow)));
const configs = selectedRooms.flatMap(room => recalls.flatMap(recall => flows.map(flow => ({room, recall, flow}))));
const runs = configs.map(config => {
  const result = run(config);
  console.log(`${config.room} recall=${config.recall} flow=${config.flow}: pnl=${result.metrics.pnl.mean.toFixed(4)}, mse=${result.metrics.mse.mean.toFixed(4)}`);
  return result;
});
const contrasts = [];
for (const room of selectedRooms) {
  for (const recall of recalls) {
    const base = runs.find(run => run.room === room && run.recall === recall && run.flow === 'none');
    if (!base) continue;
    for (const flow of flows.filter(flow => flow !== 'none')) {
      const treatment = runs.find(run => run.room === room && run.recall === recall && run.flow === flow);
      const metrics = Object.fromEntries(['pnl','mse','midpointMse','peakInventory'].map(key => [key, moments(treatment.rows.map((row, i) => row[key] - base.rows[i][key]))]));
      contrasts.push({room, recall, treatment: flow, baseline: 'none', metrics});
    }
  }
}
const output = {
  protocol: {gameCommit: GAME_COMMIT, runName, seed: root.seed, rounds, priorSeed, priorSamples: prior.samples,
    target: 'smart, default risk and action threshold; opening recall varied; closing incremental recall fixed at 0.35',
    rooms, flows: {none: 'ignore all trades', all: 'unchanged game update', 'informed-only': 'unchanged update but ignore takers whose strategy is not informed'},
    pairing: 'Same independently forked boards, research draws, maker seats, action and fill seeds. Endogenous trades may differ.',
    interval: 'Mean +/- 1.96 sample SD / sqrt(n); contrasts use within-round paired differences; pointwise, conditional on fixed prior and model; no multiplicity adjustment',
  },
  prior,
  runs: runs.map(({rows, ...run}) => run), contrasts,
};
const outputDirectory = join(fileURLToPath(new URL('.', import.meta.url)), 'results');
mkdirSync(outputDirectory, {recursive: true});
writeFileSync(join(outputDirectory, `${runName}.json`), JSON.stringify(output, null, 2) + '\n');
writeFileSync(join(outputDirectory, `${runName}-rounds.json`), JSON.stringify(runs.map(({room, recall, flow, rows}) => ({room, recall, flow, rows}))) + '\n');
console.log(`Saved ${runName}: ${configs.length} configurations x ${rounds} paired boards.`);
