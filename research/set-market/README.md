# SET essay experiment — local and unpublished

Question: can the game's trade updates substitute for better private research?
The specific forecast/profit rank reversal was selected after an exploratory
600-board pilot, then evaluated without policy tuning on a fresh 12,000-board
holdout seed. It is an original experiment in this game, not a claim to new
market-microstructure theory or an equilibrium result.

## Reproduce

Use Node 22 or later. Build a clean checkout of
https://github.com/mmcmanus1/set-mm-game at
`69153fc743f5fd799d7e075c51e38ec012c356e8` with `npm ci && npm run build`.
From this website checkout:

```sh
SET_ESSAY_RECALLS=0.25,0.5,0.65,0.85 node research/set-market/experiment.mjs /absolute/path/to/set-mm-game holdout 12000
node research/set-market/summarize.mjs
node --test research/set-market/study.test.mjs
```

The runner verifies the game commit and tracked-source cleanliness. Build the
game first: generated dist files are not covered by the Git cleanliness check.
No source in the game repository is modified by the experiment.

To reproduce the pilot, omit SET_ESSAY_RECALLS and use `pilot 600`.
The pilot predates the timing instrumentation; its summary does not contain
timing fields, but its forecasts and profit numbers are reproducible.

## Design

- Target policy is the unchanged `smart` agent (risk aversion .065, action
  threshold .04). Disable flow by suppressing only its `observeTrade` call.
  Do not compare the game's informed and smart agents as a flow-only ablation:
  they also differ in quote optimization and risk settings.
- Research is simulated by independent per-SET Bernoulli discovery, not human
  visual search. Opening recall varies over .25, .50, .65, .85. In all cases a
  second pass discovers each remaining SET with probability .35. Total recall
  is therefore .35 + .65 times opening recall. Higher recall is not a measured
  expenditure of time or money.
- Each round has one target and three opponents. Main roster: two informed
  agents (.65 initial, .35 incremental recall) plus one prior-only agent.
  Robustness rosters: three informed, or three prior-only. Opponents do not
  learn from trades. No random trader is used.
- Fixed opening spread 2, closing spread 1, four taker fills maximum, no maker
  fill cap, uniformly randomized allocation among same-side takers. Opening
  maker seats rotate by round; closing reverses their order. Eight quote
  windows per round. Cash and inventory reset every round; reported profit
  is in settlement points, before any hypothetical research cost.
- Independent prior calibration: 50,000 boards, seed `set-essay-prior-v1`.
  Add .5 pseudocount to each possible count 0–14, as in the game.
- All variants share boards and stable labeled random substreams for private
  research, action choices, and fills. Endogenous quotes and trades may differ.
  Agents see neither exact value, board cards, nor board RNG seeds.
- Final MSE is measured after quote 8's trades but *before* settlement. Timeline
  samples: opening research; after quotes 1–4; second research; after quotes 5–8.
  Stage profit marks trades to their eventual settlement value for evaluation
  only. This future value is never used in an agent's decision.
- Intervals are pointwise normal-approximation 95% Monte Carlo intervals,
  conditional on the simulator and calibrated prior. Comparisons use paired
  per-board differences, not subtraction of marginal intervals. No multiple-
  comparison correction. The headline pair was chosen in the pilot, not by
  searching the holdout for the most favorable pair.

## Interpretation boundaries

The headline compares Listener (.50 opening recall, flow on) with Researcher
(.65, flow off). The same-recall on/off experiment is the causal ablation of
this particular flow update. The headline changes two information channels.
Neither identifies the return to an optimal information-acquisition policy.

The listener's better closing profit and late MSE crossover are consistent
with learning too late. They are a decomposition, not a causal isolation of
timing: quote choices, inventory, and counterparties also change. Risk, flow
weights, fill likelihoods, horizon, and opponent types are fixed; no claim is
made that the phenomenon holds for every policy or real market.

The original flow likelihood is a heuristic tempered logistic update, not the
exact generative likelihood of this tournament. It uses taker strategy labels,
ignores passes and unfilled responses, and skips the target's own taker fills.

## Files and release gate

### Overtime follow-up

Before observing follow-up results, fix extra quote windows at 0, 4, 8, and 16;
use both original and replenished taker budgets, and both mixed and informed
rosters. Run on the fresh seed `set-essay-overtime-v1` (12,000 boards). The target
pair is unchanged: .50 opening recall with flow versus .65 without flow.

```sh
node research/set-market/overtime.mjs /absolute/path/to/set-mm-game overtime 12000
node --test research/set-market/overtime.test.mjs
```

`overtime-adapter.mjs` is an experiment-only transition from an un-revealed
settlement-pending state back to four closing quotes. It never modifies game
source. Repeated blocks use the normal closing order, with balanced opening
seats across rounds. Original capacity lasts the full extended round. Refilled
capacity restores all players to four taker fills before each extra block;
unused allowance does not accumulate. Cash, positions and evidence persist.
There is no more private research, but trade-based inference continues. Scoring
is on immutable copies so the oracle value never enters a continuing agent.

Each treatment is one trajectory with endpoint snapshots, not independent
random runs for each ending time. The policies are myopic and do not use the
announced horizon. Confidence intervals use paired differences across policies;
the improvement interval uses within-board changes in that difference. No
multiple-comparison adjustment is applied. The follow-up changes both future
trade opportunities and the future evidence stream, not information timing
alone. Expanded capacity also changes risk and both sides' opportunities.

On the mixed roster, the original-cap gap moves from -0.169 to +0.006 points
at +16 windows (interval includes zero). Refilled capacity moves it to +0.210
(interval excludes zero). With three informed opponents the corresponding
endpoints are -0.112 and +0.146. The article reports the boundary instead of
claiming that waiting alone always produces a win.

For a prefix-regression check against the retained original holdout:

```sh
SET_OVERTIME_SEED=set-essay-holdout-v1 SET_OVERTIME_ROOMS=mixed node research/set-market/overtime.mjs /absolute/path/to/set-mm-game smoke-overtime 40
node --test research/set-market/overtime.test.mjs
```

The optional seed override is for reproducibility checks; the reported follow-up
uses the default fresh seed. Tests compare the first eight windows exactly
against the earlier holdout records before the adapter can have any effect.

### Local-only artifacts

`results/holdout.json` and `results/pilot.json` contain aggregates. The large
per-round files are retained locally and gitignored; they can be regenerated.
`summarize.mjs` derives the website data and paired contrasts from those files,
recording SHA-256 digests. No data live in `public/`.

Keep the article `draft: true`. Do not stage, commit, push, or publish without
Matt's explicit approval. Before release: review the first-person voice with
Matt, choose a real publication date, test browser interaction/responsiveness,
and make the reproducibility files available alongside the released article.

### Playable opening example

`LearningTrade.astro` uses the pure exact model in `src/lib/learning-trade.mjs`.
Value 4 or 8 is sampled with equal probability when the reader posts a price.
The informed buyer buys only strictly below value (ties pass). Bayes updates
use the observed action, including passes, at the actual chosen price. Quotes
are locked until a new round; settlement is a separate reveal. Values outside
the informative price interval illustrate uninformative purchases/passes.
This is a teaching example, not the full game or an empirical experiment.
