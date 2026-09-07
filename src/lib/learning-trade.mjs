/** Exact two-value toy, not the SET game's trade policy. Ties always pass. */
export function quoteOutcome(ask, hiddenValue, revealed = false) {
  if (!Number.isFinite(ask) || ask < 3 || ask > 9 || ask * 2 % 1) throw new Error('Ask must be a half-point from 3 to 9');
  if (![4, 8].includes(hiddenValue)) throw new Error('Toy value must be four or eight');
  const buys = hiddenValue > ask;
  const likelihoods = [4, 8].map(value => Number((value > ask) === buys));
  const total = likelihoods[0] + likelihoods[1];
  const pHigh = likelihoods[1] / total;
  const mean = 4 * (1 - pHigh) + 8 * pHigh;
  const cash = buys ? ask : 0;
  const position = buys ? -1 : 0;
  const expectedProfit = cash + position * mean;
  const realizedProfit = revealed ? cash + position * hiddenValue : null;
  let explanation;
  if (buys && pHigh === 1) explanation = `A buyer who knows the value would not buy at ${ask.toFixed(1)} in the four-point world. You now know it is eight—but you owe one eight-point claim.`;
  else if (buys) explanation = 'Both values are above your ask, so either world produces a purchase. You sold cheaply without learning which world you are in.';
  else if (pHigh === 0) explanation = 'An eight-point buyer would have bought. The pass tells you the value is four, and you learned that without taking a position.';
  else explanation = 'Neither value is above your ask. Passing was certain, so it tells you nothing. You also avoided taking a position.';
  return {buys, pHigh: revealed ? Number(hiddenValue === 8) : pHigh,
    mean: revealed ? hiddenValue : mean, cash, position, expectedProfit, realizedProfit,
    revealedValue: revealed ? hiddenValue : null, explanation};
}

export const pointString = value => `${value < 0 ? '−' : value > 0 ? '+' : ''}${Math.abs(value).toFixed(2)}`;
