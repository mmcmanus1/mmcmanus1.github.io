# Local writing series

## Publication approved

Matt explicitly requested publishing both retained articles: The Price of Learning Too Late and A Neural Network with a Combination Lock. Both now use draft: false. This approval supersedes the historical draft-only and no-push notes below; the four archived posts remain excluded from the website.

## Current selection (supersedes the original series below)

Matt removed The Other Side Is Red, What Does It Mean to Be Calibrated?, When Is a Probability a Price?, and Research Notes: Neural ODEs and Inertial Drift from the website. Their source files are preserved in archive/blog/, outside Astro's content loader and public assets; restore them to src/content/blog/ if requested. Only The Price of Learning Too Late and A Neural Network with a Combination Lock remain listed in the local blog, both unpublished drafts. Series numbering and previous/next series links have been removed. The old welcome route remains unlisted.

All new writing and supporting implementation remain unstaged, uncommitted, and unpushed. Do not publish these drafts until Matt requests it. All six articles use draft: true. The blog lists and renders drafts only in development; production excludes their routes and listings. The old welcome post remains at its original route but is unlisted in the index.

## Jane Street puzzle explainer

Standalone draft: /blog/a-neural-network-with-a-combination-lock/. Added at Matt's request, separate from the original five-part sequence. Credits Jane Street, Thomas Brownback, and Shreyas Mahimkar; the attached user notes supplied the solution outline. Does not claim independent discovery or original-model execution.

Two additional interactive figures reconstruct the ReLU equality tent and the shrinking nonzero region when k comparisons share an error. Original explanatory extensions derive the L1 error budget, its geometric volume ratio, and a random-byte null model for misleading partial matches. These are analytic constructions, not empirical model results. The answer and digest verifier appear inside a collapsed spoiler reveal.

Teaching revision: the article now replaces the finished-tent figure with a three-step ramp construction. ByteLock.astro asks for a prediction at 15/16 matches before unlocking individual comparison toggles; HashBytes.astro compares real, build-time MD5 digests for four small text variants. Both use native controls and no-JavaScript explanations. Prerequisites now introduce layers, weights, biases, bits/bytes, hexadecimal, local gradient information, and binomial counting before relying on them. Original source attribution and model-execution limitations are unchanged. The older equality model remains available for the independent checks.

Connected walkthrough revision: LockWalkthrough.astro links short phrases to real standard-library MD5 bytes, target comparisons, match totals, and the reconstructed gate. The winning phrase enters the visible selector only after an explicit spoiler reveal (a reading convenience, not a security boundary). SlopeWalk.astro steps along the k=4 comparator with explicit peak-capping, outside-flat-region behavior, and reduced-motion support. An evidence checkpoint separates clues, testable predictions, and boundary checks. Pure trace and slope-policy checks are included in research/jane-street/check.mjs. Browser interaction testing has not been performed.

Final editorial/browser pass (supersedes earlier untested status for this post): trimmed repeated setup and conclusion; moved the general unequal-error formula into the collapsed geometry section. In the in-app browser, tested 390×844 and 320×740 layouts with no horizontal document overflow. Inspected mobile screenshots of the ramp, lock, slope, and walkthrough. Long equations scroll within their containers at 320px. Verified keyboard prediction, byte toggle to 16/16, reset focus, construction slider, inside/outside slope steps, hash selector changes, spoiler trace to output 1, and expandable geometry. No console errors were captured. Fixed prediction focus skipping the tiles by focusing the unmatched sixteenth tile. Math checks remain separate from browser checks.

Before publication, ask an actual unfamiliar reader (not an author or simulated persona), without coaching: (1) Why do fifteen matching bytes still produce zero? (2) Why won't changing one letter reliably fix only the sixteenth? Ask where they first had to reread a passage. No external reader was contacted, and that test remains pending. Nothing in this review approves publication.

Run `node research/jane-street/check.mjs` to check every integer byte difference, all 65,536 match/mismatch patterns, continuous tent values, plot bounds, the reported digest, and numerical claims. No model pickle is downloaded or executed. The interrupted SET research remains exploratory in research/set-market/ and has not been presented as a finished result or used to rewrite that post.

## Read the series

Start the normal Astro development server and visit /blog/. The navigation now includes Blog, and the articles link forward and backward through this order:

1. /blog/the-other-side-is-red/ — sample-space enumeration, priors, likelihoods, Bayes' rule, and an interactive change to the observation procedure.
2. /blog/calibration-is-not-enough/ — calibration, information, a finite-sample reliability diagram, a Brier-loss derivation and explorable, evaluation pitfalls, and the AIA benchmark connection.
3. /blog/a-market-hidden-in-a-card-game/ — **The Price of Learning Too Late**, now the focused flagship draft: original paired game experiments, a forecast/profit rank reversal, a measured learning timeline, interactive control rosters and research settings, an adverse-selection derivation, and full methods/code/data.
4. /blog/when-is-a-probability-a-price/ — a one-dollar binary claim, discounting, executable prices and fees, state-dependent value, physical versus pricing probabilities, and settlement details.
5. /blog/neural-odes-and-inertial-drift/ — double integration, bias drift, orientation, ODEs, the documented thesis architecture, a normalized reported result, and limitations.

## Teaching approach

Use a concrete question before terminology, derive one small case before stating a general equation, and place a plot next to the passage it explains. Each article has an optional check-your-understanding question. The style is original, pursuing visual intuition and accessible explanations without copying another creator's wording or persona.

The publication is called Blog under Matt's name. No newsletter account, mailing list, paid product, or new brand was created. “Prices & Priors” remains an earlier unapproved suggestion.

## Implementation and figures

MDX allows plots inside the text. remark-math and rehype-katex render equations at build time; KaTeX assets are bundled locally. The installed MDX major supports the existing Astro 4 project.

Explorable.astro renders SVG figures and native labeled range controls. src/lib/explainers.mjs supplies deterministic mathematical models for initial rendering and client updates. These include card posteriors, synthetic calibration, expected Brier loss, informed-buyer selection, inventory payoff (now unused), discounting, constant-bias drift, and normalized thesis RMSE. Motion follows direct input and respects reduced-motion preferences. SetStudy.astro adds the measured comparison and step-through timeline, backed by src/data/set-market-study.json. The data are embedded only in rendered draft HTML, never imported into the production client bundle. SetReproduction.astro makes the runner, dataset-generation script, and numerical results inspectable within the article.

## SET flagship experiment

The earlier article was a primer with a proposed experiment. It has been rewritten around a measured result: at 50% opening recall plus trade updates, the smart agent has lower final MSE but lower gross profit than the same agent at 65% recall without trade updates. The prior, quoting rule, inventory settings, and opponent roster are controlled. The headline pair was selected after a 600-board pilot and then confirmed on 12,000 independently seeded holdout boards. Twenty-four holdout settings cover four recalls, two flow modes, and three opponent rosters. Same-recall ablations and a prior-only control are included; there is no claim to a new general market theorem.

The game checkout remains unchanged. research/set-market/README.md documents reproduction, assumptions, statistical intervals, and limitations. The large per-round files are retained locally and gitignored, not deleted or placed in public/. Article and all research work remain local, unstaged, uncommitted, and unpushed.

### Overtime follow-up and playable introduction

The SET draft now includes a second experiment on 12,000 fresh boards: 0, 4, 8, or 16 extra quote windows, without additional private research. A research-only adapter preserves the original game engine and continues cash, inventory, and beliefs. Original taker capacity is compared with restoring all players to four taker fills before each extra block. Extra time alone yields no clear winner in the mixed roster at +16 windows and leaves Listener behind in the all-informed roster; replenished capacity produces a positive Listener advantage in both. The article and figure state these boundaries explicitly rather than treating more time as a universal cure.

OvertimeStudy.astro renders paired intervals and opponent/endpoint controls. LearningTrade.astro adds a one-off quote puzzle with a fairly sampled hidden value, a price-dependent informed response, exact Bayesian updating from buys or passes, a locked quote, and a separate settlement reveal. It uses the existing site's understated styling and is not a change to the standalone SET game.

The production build succeeds; 13 study tests cover raw-record reproduction, confidence intervals, original-engine prefix parity, the continuation adapter, all plot choices, every toy price/value/reveal combination, and exclusion of the unpublished article/data from production. Local HTTP rendering also succeeds. Browser interaction and visual QA have not been performed in this pass.

Synthetic and analytic examples are labeled beside their figures. The thesis comparison uses Table 6.1's 516.55 and 191.21 values, normalized to baseline = 100%, because the table does not print units on those entries. Do not relabel the toy drift curves as EKF or trained-network output. The broader neural-ODE equation is a conceptual connection, not a claim that the thesis implemented that exact architecture.

## Sources checked

- Thesis Chapters 4–6 and rendered Table 6.1 on page 70: public/files/MIT-McManus-Thesis.pdf.
- SET game README and rules: https://github.com/mmcmanus1/set-mm-game
- AIA report abstract: https://arxiv.org/abs/2511.07678
- Gneiting and Raftery: https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
- Wolfers and Zitzewitz: https://www.nber.org/papers/w12200
- Cochrane: https://www.johnhcochrane.com/asset-pricing
- Chen et al.: https://arxiv.org/abs/1806.07366

## Validation

Production build and whitespace checks pass. Local requests verify all five pages return HTTP 200, contain rendered KaTeX without error markers, and include their figures and valid contents anchors. The local index lists them in the intended order. Model checks cover endpoint values, Bayesian normalization, independently expanded Brier expectations, drift calculations, and finite plot bounds at every control's minimum/default/maximum. Production output is checked separately for absence of all five draft routes and titles. Browser interaction and screenshot review have not been performed in this pass.

Before release, Matt should review the voice and attribution, choose actual publication dates, and explicitly approve any post for publication. No personal work anecdotes or unreported experimental successes are invented in these drafts.
