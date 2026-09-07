// Pure, deterministic teaching models. No market or thesis time-series data.
const pct = x => `${(100 * x).toFixed(1)}%`;
export const expectedBrier = (p, q) => (p - q) ** 2 + q * (1 - q);
export const cardPosterior = s => 1 / (1.5 + .5 * s);
export const buyerPosterior = a => (1 + a) / 2;
export const binaryPrice = (p, r, t) => p * Math.exp(-r * t);
export const drift = (b, t, correction = 0) => .5 * b * (1 - correction) * t * t;
export const reluEquality = d => Math.max(0,d-1)-2*Math.max(0,d)+Math.max(0,d+1);
export const reluLock = differences => Math.max(0, differences.reduce((sum,d)=>sum+reluEquality(d),0)-differences.length+1);
export const rampBuild = (d,stage) => Math.max(0,d+1) - (stage>=2?2*Math.max(0,d):0) + (stage>=3?Math.max(0,d-1):0);
const line = (fn, n = 100) => Array.from({length:n+1}, (_,i) => [i/n, fn(i/n)]);
export const definitions = {
  'relu-build': {
    title: 'Build a detector in three moves',
    hint: 'Start at step 1. Before moving to step 2, predict what subtracting a steeper ramp will do to the right-hand slope.',
    controls: [{key:'stage',label:'Construction step',min:1,max:3,step:1,value:1,unit:''},{key:'difference',label:'Probe the difference d',min:-2,max:2,step:.25,value:0,unit:''}],
    note: 'Step 1: R(d + 1). Step 2: subtract 2R(d). Step 3: add R(d − 1). Blue is the running sum; the dashed orange line is the piece added at this step. At step 1 the two lines coincide.',
  },
  'relu-equality': {
    title: 'Three ramps make one exact-match detector',
    hint: 'Move the computed value away from its target. The line includes fractional differences; actual byte differences are integers.',
    controls: [{key:'difference',label:'Computed byte − target byte',min:-2,max:2,step:.05,value:0,unit:''}],
    note: 'Analytic reconstruction of the comparator, not a run of Jane Street’s model. Only d = 0 scores 1; every other integer scores 0.',
  },
  'relu-lock': {
    title: 'How much room is there to be almost right?',
    hint: 'Give k byte comparisons the same error d, keeping the other comparisons exactly correct. Increase k and watch the nonzero region shrink.',
    controls: [{key:'count',label:'Comparisons sharing the error (k)',min:1,max:16,step:1,value:16,unit:''},{key:'difference',label:'Error in each of those comparisons (d)',min:-1,max:1,step:.01,value:.1,unit:''}],
    note: 'Continuous thought experiment in digest coordinates, not text-input space. The original circuit has 16 byte comparisons. Fractional digest values need not be reachable from text.',
  },
  cards: {
    title: 'Same red face. Different selection rule.',
    hint: 'Move from a randomly revealed side to a helper who always shows red when possible.',
    controls: [{key:'selection',label:'Chance the helper chooses the side',min:0,max:100,step:1,value:0,unit:'%'}],
    note: 'Exact probabilities. The three card types are equally likely before observing a red side.',
  },
  calibration: {
    title: 'Does 80% behave like 80%?',
    hint: 'Turn up confidence, then increase the sample size. Confidence = 1 means the forecasts equal the true probabilities.',
    controls: [{key:'confidence',label:'Confidence multiplier',min:.25,max:3,step:.05,value:1,unit:'×'}, {key:'samples',label:'Observations per forecast group',min:20,max:1000,step:20,value:100,unit:''}],
    note: 'Synthetic example: nine equally common groups with true probabilities 10% through 90%. Seeded Bernoulli outcomes; all points have the same sample count. The reported score is a population expectation.',
  },
  brier: {
    title: 'Find the bottom of the bowl',
    hint: 'The event has a true probability of 20%. Choose the probability you would report.',
    controls: [{key:'forecast',label:'Your forecast',min:0,max:100,step:1,value:50,unit:'%'}],
    note: 'Exact expected loss for a Bernoulli event with q = 0.2. The curve shows an expectation over outcomes, not a single realized score.',
  },
  flow: {
    title: 'Who accepts your offer?',
    hint: 'The count is 4 or 8, each initially equally likely. You offer to sell at 6. An informed buyer buys only at value 8; an uninformed buyer buys with probability 1/2 regardless of value.',
    controls: [{key:'informed',label:'Fraction of potential buyers who know the count',min:0,max:100,step:1,value:50,unit:'%'}],
    note: 'Illustrative two-value model, not the SET game’s implemented agent policy. Buyer type is independent of the count.',
  },
  inventory: {
    title: 'Your position changes the slope',
    hint: 'Choose a signed position acquired at 6 per claim. Long is positive; short is negative.',
    controls: [{key:'position',label:'Position in claims',min:-5,max:5,step:1,value:2,unit:''}],
    note: 'Hypothetical fills at one price, no fees. Profit = position × (final count − 6). Real game fills can occur at different prices.',
  },
  price: {
    title: 'One dollar later is worth less today',
    hint: 'Change the belief, the wait, or the continuously compounded rate. The blue line is the present value of the expected payoff.',
    controls: [{key:'probability',label:'Probability of yes',min:0,max:100,step:1,value:60,unit:'%'},{key:'rate',label:'Annual rate',min:0,max:10,step:.25,value:5,unit:'%'},{key:'years',label:'Years until payment',min:0,max:5,step:.25,value:1,unit:''}],
    note: 'An idealized risk-neutral valuation with a certain payment time, reliable settlement, and no fees. It is not a live quote or an investment recommendation.',
  },
  drift: {
    title: 'A small bias, integrated twice',
    hint: 'Change the acceleration bias and the fraction you remove. Try 100% correction, then go slightly past it.',
    controls: [{key:'bias',label:'Constant acceleration bias',min:0,max:.05,step:.001,value:.01,unit:' m/s²'},{key:'correction',label:'Bias removed',min:0,max:120,step:1,value:80,unit:'%'}],
    note: 'One-dimensional analytic toy model with zero initial error and a constant bias. The orange curve is an assumed correction, not a trained neural network or an EKF result.',
  },
  thesis: {
    title: 'The reported 3D RMSE comparison', hint: 'Lower is better. Both bars use the baseline result as 100%.', controls: [],
    note: 'Re-expressed from Table 6.1, p. 70 of the thesis: baseline 516.55, SciML 191.21, in the table’s reported scale. These bars are dimensionless; no uncertainty interval or raw trajectory is inferred.',
  },
};
export function model(kind, v = {}) {
  const base = {xRange:[0,1],yRange:[0,1],xTicks:[[0,'0'],[.5,'0.5'],[1,'1']],yTicks:[[0,'0'],[.5,'0.5'],[1,'1']],lines:[],dots:[],bars:[],result:''};
  if(kind==='relu-build') {
    const piece=d=>v.stage===1?Math.max(0,d+1):v.stage===2?-2*Math.max(0,d):Math.max(0,d-1);
    const xs=[-2,-1,0,1,2];
    const explanation=['The ramp starts at −1 and keeps rising.','Subtracting twice the ramp at 0 folds the rising line downward. It still falls below zero to the right.','Adding a ramp at 1 stops the fall. Only a tent centered at 0 remains.'][v.stage-1];
    return {...base,xRange:[-2,2],yRange:[-4,4],xTicks:xs.map(x=>[x,String(x)]),yTicks:[[-4,'−4'],[-2,'−2'],[0,'0'],[2,'2'],[4,'4']],xLabel:'Difference d = computed value − target',yLabel:'Height of the running sum and added piece',lines:[{name:'Running sum',points:xs.map(d=>[d,rampBuild(d,v.stage)])},{name:'Piece added at this step',dashed:true,points:xs.map(d=>[d,piece(d)])}],dots:[[v.difference,rampBuild(v.difference,v.stage)]],result:`${explanation} At d = ${v.difference.toFixed(2)}, the sum is ${rampBuild(v.difference,v.stage).toFixed(2)}.`};
  }
  if(kind==='relu-equality') {
    const score=reluEquality(v.difference);
    return {...base,xRange:[-2,2],xTicks:[[-2,'−2'],[-1,'−1'],[0,'0'],[1,'1'],[2,'2']],xLabel:'Difference d',yLabel:'Equality score E(d)',lines:[{name:'Three-ReLU comparator',points:[[-2,0],[-1,0],[0,1],[1,0],[2,0]]}],dots:[[v.difference,score]],result:`At d = ${v.difference.toFixed(2)}, E(d) = ${score.toFixed(2)}. ${Number.isInteger(v.difference)?'This is an integer byte difference.':'This fractional score belongs to the continuous relaxation, not an ordinary byte comparison.'}`};
  }
  if(kind==='relu-lock') {
    const width=1/v.count, score=reluLock(Array(v.count).fill(v.difference));
    return {...base,xRange:[-1,1],xTicks:[[-1,'−1'],[-.5,'−0.5'],[0,'0'],[.5,'0.5'],[1,'1']],xLabel:'Shared error d',yLabel:'Final output',lines:[{name:'One imperfect comparison',points:[[-1,0],[0,1],[1,0]],dashed:true},{name:'Selected k comparisons',points:[[-1,0],[-width,0],[0,1],[width,0],[1,0]]}],dots:[[v.difference,score]],result:`With k = ${v.count}, output is positive only when |d| < ${width.toFixed(4)}. At d = ${v.difference.toFixed(2)}, output = ${score.toFixed(2)}.`};
  }
  if(kind==='cards') {
    const s= v.selection/100, p=cardPosterior(s);
    return {...base,xRange:[0,3],xTicks:[[.5,'RR'],[1.5,'RB'],[2.5,'BB']], yLabel:'Probability after seeing red',xLabel:'Card in your hand',bars:[[.5,p],[1.5,1-p],[2.5,0]],result:`RR: ${pct(p)} · RB: ${pct(1-p)} · BB: 0%. The probability the reverse is red is ${pct(p)}.`};
  }
  if(kind==='calibration') {
    const dots=[]; let expected=0;
    for(let j=1;j<=9;j++) {
      const q=j/10, p=1/(1+Math.exp(-v.confidence*Math.log(q/(1-q))));
      let seed=17+j*7919, yes=0;
      for(let i=0;i<v.samples;i++) { seed=(1664525*seed+1013904223)>>>0; yes+=Number(seed/4294967296<q); }
      dots.push([p,yes/v.samples]); expected+=expectedBrier(p,q)/9;
    }
    return {...base,xLabel:'Forecast probability',yLabel:'Observed fraction of yes outcomes',lines:[{name:'Perfect calibration',points:[[0,0],[1,1]],dashed:true}],dots,result:`${9*v.samples} synthetic outcomes. Expected Brier loss: ${expected.toFixed(3)}; always predicting 50% gives 0.250.`};
  }
  if(kind==='brier') {
    const p=v.forecast/100, loss=expectedBrier(p,.2);
    return {...base,xLabel:'Reported probability',yLabel:'Expected Brier loss',lines:[{name:'Expected loss',points:line(p=>expectedBrier(p,.2))}],dots:[[p,loss]],result:`At ${pct(p)}, expected loss is ${loss.toFixed(3)} = ${((p-.2)**2).toFixed(3)} from the probability error + 0.160 from the outcome’s uncertainty.`};
  }
  if(kind==='flow') {
    const p=buyerPosterior(v.informed/100), value=4+4*p;
    return {...base,xRange:[0,2],xTicks:[[.5,'Count = 4'],[1.5,'Count = 8']],xLabel:'Settlement value',yLabel:'Probability, given a purchase',bars:[[.5,1-p],[1.5,p]],result:`After a buy, expected count is ${value.toFixed(2)}. Selling at 6 has expected profit ${(6-value).toFixed(2)} per filled claim.`};
  }
  if(kind==='inventory') {
    return {...base,xRange:[0,12],yRange:[-30,30],xTicks:[[0,'0'],[6,'6'],[12,'12']],yTicks:[[-30,'−30'],[0,'0'],[30,'30']],xLabel:'Final SET count',yLabel:'Profit (points)',lines:[{name:'Profit at settlement',points:Array.from({length:13},(_,i)=>[i,v.position*(i-6)])}],dots:[[6,0]],result:`Position ${v.position}: at count 4, profit is ${v.position*(-2)}; at count 8, profit is ${v.position*2}. Each extra SET changes profit by ${v.position} points.`};
  }
  if(kind==='price') {
    const p=v.probability/100, price=binaryPrice(p,v.rate/100,v.years);
    return {...base,xLabel:'Probability of yes',yLabel:'Present value ($)',lines:[{name:'Immediate payment',points:[[0,0],[1,1]],dashed:true},{name:'Payment after the wait',points:line(q=>binaryPrice(q,v.rate/100,v.years))}],dots:[[p,price]],result:`A ${pct(p)} belief gives an idealized value of $${price.toFixed(3)} today, versus $${p.toFixed(3)} with immediate payment.`};
  }
  if(kind==='drift') {
    return {...base,xRange:[0,60],yRange:[-20,100],xTicks:[[0,'0'],[30,'30'],[60,'60']],yTicks:[[-20,'−20'],[0,'0'],[50,'50'],[100,'100']],xLabel:'Time (seconds)',yLabel:'Signed position error (meters)',lines:[{name:'Uncorrected bias',points:line(t=>drift(v.bias,t*60)).map(([x,y])=>[x*60,y])},{name:'Assumed bias correction',dashed:true,points:line(t=>drift(v.bias,t*60,v.correction/100)).map(([x,y])=>[x*60,y])}],result:`After 60 s: ${drift(v.bias,60).toFixed(2)} m uncorrected; ${drift(v.bias,60,v.correction/100).toFixed(2)} m after correction.`};
  }
  if(kind==='thesis') return {...base,xRange:[0,2],xTicks:[[.5,'Baseline'],[1.5,'SciML']],xLabel:'Reported system',yLabel:'3D RMSE relative to baseline',bars:[[.5,1],[1.5,191.21/516.55]],result:`SciML is ${(100*191.21/516.55).toFixed(2)}% of baseline RMSE: a ${(100*(1-191.21/516.55)).toFixed(2)}% reduction in this reported simulation comparison.`};
  throw new Error(`Unknown explainer: ${kind}`);
}
export const scaleX=(x,m)=>55+(x-m.xRange[0])/(m.xRange[1]-m.xRange[0])*465;
export const scaleY=(y,m)=>240-(y-m.yRange[0])/(m.yRange[1]-m.yRange[0])*200;
export const pathFor=(points,m)=>points.map(([x,y],i)=>`${i?'L':'M'}${scaleX(x,m).toFixed(2)},${scaleY(y,m).toFixed(2)}`).join(' ');
