#version 420

layout(location = 0) out vec4 tempAccum;
layout(location = 1) out vec4 den1;
layout(location = 2) out vec4 var1;

in vec2 texCoord;

//uniform sampler2D color;
uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
//uniform sampler2D secondpos;
//uniform sampler2D weigth;
//uniform sampler2D outgoingr;
uniform sampler2D weightS;
uniform sampler2D outgoingrS;
uniform sampler2D prevN;
uniform sampler2D prevAcc;
//uniform sampler2D reflAlb;
//uniform sampler2D prevPosition;
//uniform sampler2D prevSecondPosition;
uniform sampler2D prevvar;
//uniform sampler2D prevden;

//uniform mat4 view;
//uniform mat4 projection;
//uniform mat4 invview;
//uniform mat4 invproj;
uniform mat4 prevview;
uniform mat4 prevproj;

//uniform vec2 wh;
//uniform float time;
//uniform vec3 viewPos;
//uniform vec3 lastViewPos;
//uniform vec3 ldir;

// NOT MY CODE///////////////
uint wang_hash(inout uint seed) {
  seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
  seed *= uint(9);
  seed = seed ^ (seed >> 4);
  seed *= uint(0x27d4eb2d);
  seed = seed ^ (seed >> 15);
  return seed;
}

float rndf(inout uint state) { return float(wang_hash(state)) / 4294967296.0; }
///////////////////////////
vec3 lerp(vec3 a, vec3 b, float t) { return mix(a, b, t); }
float lerp(float a, float b, float t) { return mix(a, b, t); }

vec3 interpolateHistory(vec3 prev, vec3 current, float antilagAlpha,
                        float invHistLen, float minAlpha) {
  float alpha = lerp(max(minAlpha, invHistLen), 1., antilagAlpha);
  return lerp(prev, current, alpha);
}
float lum(vec3 c) {
  return sqrt(0.299 * c.x * c.x + 0.587 * c.y * c.y + 0.114 * c.z * c.z);
}

void main() {
  //vec2 iResolution = wh;
  if (texture2D(albedo, texCoord).w > 0.5) {
    return;
  }

  //vec3 cameraOffset = viewPos - lastViewPos;
  vec3 View = texture2D(position, texCoord).xyz;
  vec4 Projected = vec4(View.xyz, 1.); // + vec4(cameraOffset, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;
  float RW = texture2D(weightS, texCoord).z;
  vec3 col = texture2D(outgoingrS, texCoord).xyz * clamp(RW, 0., 200.);
  //float roughness = texture2D(color, texCoord).w;
  bool isthesky = false;

  bool outsc = false;
  //float isSpec = texture2D(reflAlb, texCoord).w;

  if (ProjectedCoordinates.x > 1. || ProjectedCoordinates.x < 0. ||
      ProjectedCoordinates.y > 1. || ProjectedCoordinates.y < 0.) {
    ProjectedCoordinates = texCoord;
    outsc = true;
  }

  vec4 n = texture2D(normal, texCoord);

  vec4 nDij = texture2D(prevN, ProjectedCoordinates);
  bool outScreen =
      (ProjectedCoordinates.x > 1.0 || ProjectedCoordinates.x < 0. ||
       ProjectedCoordinates.y > 1.0 || ProjectedCoordinates.y < 0.0);

  float depthWeight = pow(exp(-abs((nDij.w) - (n.w)) * 0.01), 12.);

  float normalWeight = pow(max(dot(nDij.xyz, n.xyz), 0.), 5.);
  float totalWeight = (1.0 - float(outScreen)) * normalWeight * depthWeight;

  float accumulation = texture2D(prevAcc, ProjectedCoordinates).w;

  accumulation = clamp(accumulation + 1.0, 0.0, 42.);
  accumulation *= totalWeight;

  float frameWeight = (1.0 / max(1.0, accumulation));

  //

  vec3 prevFrame = texture2D(prevAcc, ProjectedCoordinates).xyz;
  // vec3 prevFrame = texture2D(prevden, ProjectedCoordinates).xyz;
  /*
  vec3 kk = vec3(0.);
  vec3 minCol = (vec3(9999.));
  vec3 maxCol = (vec3(-9999.));
  for (int i = 0; i < 9; i++) {
    vec2 coords = vec2(float(i % 3) - 1., float(i / 3) - 1.) *
                  min(1.0 - min(accumulation * 0.1, 1.), 1.);
    vec2 newc = ((ProjectedCoordinates)*iResolution + coords) / iResolution;
    vec3 currSample = texture2D(prevAcc, newc).xyz;
    minCol = min(minCol, currSample);
    maxCol = max(maxCol, currSample);
    kk += currSample / 9.;
  }
  prevFrame = ClipAABB(prevFrame, minCol, maxCol);
  */

  // if(true){
  // if( isSpec < 0.5){
  col = col * (frameWeight) + (1.0 - frameWeight) * prevFrame;
  //}

  den1 = vec4(col, 7.);
  vec2 nm = clamp(vec2(lum(col), pow(lum(col), 2.)), 0., 100.);
  vec2 prevnm = clamp(texture2D(prevvar, ProjectedCoordinates).yz, 0., 100.);
  nm = prevnm * (1.0 - frameWeight) + nm * frameWeight;

  float variance = abs(nm.g - nm.r * nm.r);
  tempAccum = vec4(vec3(col), accumulation);
  variance = variance * frameWeight + (1. - frameWeight);
  var1 = vec4(variance, nm, 1.);

  vec2 TC = texCoord;
  
}