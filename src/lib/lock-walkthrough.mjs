export const targetDigest = 'c7ef65233c40aa32c2b9ace37595fa7c';
export function compareDigest(digest) {
  if (!/^[0-9a-f]{32}$/.test(digest)) throw new Error('Expected 16 hexadecimal bytes');
  const bytes=digest.match(/../g);
  const targets=targetDigest.match(/../g);
  const matches=bytes.map((byte,i)=>byte===targets[i]);
  const count=matches.filter(Boolean).length;
  return {bytes,targets,matches,count,raw:count-15,output:Math.max(0,count-15)};
}
export const slopeScore=d=>Math.max(0,1-4*Math.abs(d));
// A deliberately simple teaching policy: stop at the peak/boundary, and cap
// each uphill step at the known peak rather than oscillating across its cusp.
export function slopeStep(d) {
  if(d===0 || Math.abs(d)>=.25) return d;
  const distance=Math.max(0,Math.abs(d)-.04);
  return distance<1e-12?0:Math.sign(d)*distance;
}
