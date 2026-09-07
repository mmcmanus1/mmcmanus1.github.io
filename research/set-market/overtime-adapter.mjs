import assert from 'node:assert/strict';

/** Research-only extension. No patch is made to the production game engine. */
export function reopenClosingMarket(state, capacity = 'original') {
  assert.equal(state.phase, 'SETTLEMENT_PENDING');
  assert.equal(state.stage, 'closing');
  assert.equal(state.settlement, undefined, 'Never continue a revealed/settled round');
  assert.equal(state.quote, undefined);
  assert.ok(['original', 'refill'].includes(capacity));
  const players = capacity === 'original' ? state.players : Object.freeze(Object.fromEntries(
    Object.entries(state.players).map(([id, ledger]) => [id, Object.freeze({...ledger, takerFillsRemaining: 4})]),
  ));
  return Object.freeze({
    ...state, players, phase: 'QUOTE_OPEN', makerIndex: 0,
    makerOrder: Object.freeze([...state.openingOrder].reverse()), version: state.version + 1,
  });
}
