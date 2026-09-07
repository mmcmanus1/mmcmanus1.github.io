export const roomLabels = {
  mixed: 'Two researchers + one prior-only player',
  informed: 'Three researchers',
  prior: 'Three prior-only players',
};
export const timelineLabels = [
  'Before any trading', 'After opening quote 1', 'After opening quote 2',
  'After opening quote 3', 'After opening quote 4', 'After the second research pass',
  'After closing quote 1', 'After closing quote 2', 'After closing quote 3',
  'After closing quote 4 — no trades left',
];
export const timelineX = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10];
export const fixed = value => value.toFixed(3);
export const signed = value => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(3)}`;

export function comparison(data, room = 'mixed', recall = .5) {
  const listener = data.runs.find(run => run.room === room && run.recall === recall && run.flow === 'all');
  const researcher = data.runs.find(run => run.room === room && run.recall === .65 && run.flow === 'none');
  const pair = data.pairs.find(pair => pair.room === room && pair.recall === recall);
  if (!listener || !researcher || !pair) throw new Error('Unknown study comparison');
  const series = [listener, researcher];
  const charts = ['mse', 'pnl'].map(key => {
    const values = series.map(run => run.metrics[key]);
    const min = Math.min(0, ...values.map(value => value.low));
    const max = Math.max(0, ...values.map(value => value.high));
    const pad = (max - min || 1) * .08;
    const lo = min < 0 ? min - pad : 0;
    const hi = max + pad;
    return {key, values, min: lo, max: hi,
      ticks: Array.from({length: 5}, (_, i) => lo + i * (hi - lo) / 4),
    };
  });
  const error = pair.difference.mse;
  const profit = pair.difference.pnl;
  const errorWord = error.high < 0 ? 'lower final error' : error.low > 0 ? 'higher final error' : 'no clear difference in final error';
  const profitWord = profit.high < 0 ? 'lower profit' : profit.low > 0 ? 'higher profit' : 'no clear difference in profit';
  return {listener, researcher, pair, charts,
    result: `With ${Math.round(100 * recall)}% opening recall, Listener has ${errorWord} and ${profitWord} than Researcher.`,
    interval: `Paired Listener − Researcher: error ${signed(error.mean)} [${signed(error.low)}, ${signed(error.high)}] points²; profit ${signed(profit.mean)} [${signed(profit.low)}, ${signed(profit.high)}] points per round.`,
  };
}

export const comparisonX = (value, chart) => 42 + (value - chart.min) / (chart.max - chart.min) * 470;
export const timeX = index => 52 + timelineX[index] / 10 * 460;
export const errorY = value => 244 - value / 1.3 * 210;
export const timelinePath = values => values.map((value, i) => `${i ? 'L' : 'M'}${timeX(i).toFixed(2)},${errorY(value.mean).toFixed(2)}`).join(' ');

export function timeline(data, index = 0) {
  if (!Number.isInteger(index) || index < 0 || index > 9) throw new Error('Timeline index outside 0–9');
  const {listener, researcher} = comparison(data);
  const a = listener.metrics.beliefTimeline[index].mean;
  const b = researcher.metrics.beliefTimeline[index].mean;
  return {
    values: [a, b],
    result: `${timelineLabels[index]}: Listener error ${fixed(a)}; Researcher error ${fixed(b)} points². ${a < b ? 'Listener is now more accurate on average.' : 'Researcher is still more accurate on average.'}`,
  };
}
