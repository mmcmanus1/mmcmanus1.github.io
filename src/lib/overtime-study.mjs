export const overtimeX = extra => 52 + extra / 16 * 460;
export const overtimeY = gap => 244 - (gap + .25) / .55 * 210;
export const overtimePath = points => points.map((point, i) => `${i ? 'L' : 'M'}${overtimeX(point.extra).toFixed(2)},${overtimeY(point.pnl.mean).toFixed(2)}`).join(' ');
const signed = value => `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(3)}`;

export function overtimeModel(data, room = 'mixed', extra = 16) {
  const series = ['original', 'refill'].map(capacity => {
    const result = data.contrasts.find(row => row.room === room && row.capacity === capacity);
    if (!result) throw new Error('Unknown overtime room');
    const selected = result.deltas.find(row => row.extra === extra);
    if (!selected) throw new Error('Unknown overtime checkpoint');
    const interval = selected.pnl;
    const verdict = interval.low > 0 ? 'Listener leads' : interval.high < 0 ? 'Researcher leads' : 'no clear winner';
    return {capacity, points: result.deltas, selected,
      label: capacity === 'original' ? 'Keep the original allowance' : 'Refresh the allowance each block',
      result: `${signed(interval.mean)} points per round [${signed(interval.low)}, ${signed(interval.high)}]; ${verdict}.`,
    };
  });
  return {series, caption: `${extra} extra quote windows · Listener’s profit minus Researcher’s profit`,
    result: `After ${extra} extra quote windows: original allowance ${series[0].result} Refreshed allowance ${series[1].result}`};
}
