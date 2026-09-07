// Independent checks of the essay's math. Does not load or run the puzzle model.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {compareDigest,slopeScore,slopeStep} from '../../src/lib/lock-walkthrough.mjs';
import {reluEquality, reluLock, rampBuild, definitions, model, scaleX, scaleY} from '../../src/lib/explainers.mjs';

const close = (a,b) => assert.ok(Math.abs(a-b)<1e-10, `${a} != ${b}`);
for (let d=-255;d<=255;d++) close(reluEquality(d),Number(d===0));
for (let i=-400;i<=400;i++) {
  const d=i/100;
  close(reluEquality(d),Math.max(0,1-Math.abs(d)));
  close(rampBuild(d,1),Math.max(0,d+1));
  close(rampBuild(d,2),Math.max(0,d+1)-2*Math.max(0,d));
  close(rampBuild(d,3),reluEquality(d));
  for(let k=1;k<=16;k++) {
    const errors=Array.from({length:16},(_,j)=>j<k?d:0);
    close(reluLock(errors),Math.max(0,1-k*Math.abs(d)));
  }
}
for(let mask=0;mask<65536;mask++) {
  const errors=Array.from({length:16},(_,i)=>(mask>>i)&1);
  close(reluLock(errors),Number(mask===0));
}
for(const kind of ['relu-build','relu-equality','relu-lock']) {
  const controls=definitions[kind].controls;
  const combinations=controls.reduce((rows,c)=>rows.flatMap(row=>[c.min,c.value,c.max].map(v=>({...row,[c.key]:v}))),[{}]);
  for(const values of combinations) {
    const m=model(kind,values);
    for(const [x,y] of [...m.lines.flatMap(l=>l.points),...m.dots]) {
      assert.ok(Number.isFinite(scaleX(x,m))&&Number.isFinite(scaleY(y,m)));
      assert.ok(x>=m.xRange[0]&&x<=m.xRange[1]);
      assert.ok(y>=m.yRange[0]-1e-10&&y<=m.yRange[1]+1e-10);
    }
  }
}
const target='c7ef65233c40aa32c2b9ace37595fa7c';
const md5=text=>createHash('md5').update(text,'ascii').digest('hex');
assert.equal(md5('bitter lesson'),target);
for(const phrase of ['vegetable dog','cat','bitter lesson']) {
  const digest=md5(phrase), trace=compareDigest(digest);
  assert.equal(trace.bytes.length,16);
  assert.equal(trace.output,Number(digest===target));
  close(trace.output,reluLock(trace.bytes.map((byte,i)=>parseInt(byte,16)-parseInt(trace.targets[i],16))));
}
assert.throws(()=>compareDigest('invalid'));
let inside=.15;
for(let i=0;i<4;i++) { const next=slopeStep(inside);assert.ok(slopeScore(next)>slopeScore(inside));inside=next; }
assert.equal(inside,0);
assert.equal(slopeStep(0),0);
assert.equal(slopeStep(.6),.6);
assert.equal(slopeStep(-.6),-.6);
assert.equal(slopeStep(.25),.25);
assert.ok(slopeStep(-.15)>-.15);
assert.equal(md5('cat'),'d077f244def8a70e5ea758bd8352fcd8');
assert.equal(parseInt('00101010',2),42);
assert.equal((42).toString(16),'2a');
for(const variant of ['Bitter lesson','bitter  lesson','bitter lesson\n']) assert.notEqual(md5(variant),target);
const chance=1-(255/256)**16;
const factorial=Array.from({length:16},(_,i)=>i+1).reduce((a,b)=>a*b,1);
assert.equal(Math.round(chance*1e6),60702);
console.log({checks:'passed',digest:target,atLeastOneMatch:chance,expectedPerMillion:chance*1e6,volumeFraction:1/factorial});
