#version 420

layout(location = 0) out vec4 tempAccum;
layout(location = 1) out vec4 den1;
layout(location = 2) out vec4 var1;

in vec2 texCoord;

uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D weightS;
uniform sampler2D outgoingrS;
uniform sampler2D prevN;
uniform sampler2D prevAcc;
uniform sampler2D prevvar;
uniform sampler2D prevden;

uniform mat4 prevview;
uniform mat4 prevproj;
uniform vec2 wh;
uniform float time;
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

vec3 retCol(vec2 texCoord){
  float RW = texture2D(weightS, texCoord).z;
  vec3 col = texture2D(outgoingrS, texCoord).xyz * clamp(RW, 0., 200.);
  return col;
}

void main() {
  vec2 iResolution = wh;

vec2 fragCoord = texCoord * wh;

  if(int(fragCoord.x) % 2 == int(time)%2 && int(fragCoord.y) % 2 == int(time)%2){
      //discard;
   }

  if (texture2D(albedo, texCoord).w > 0.5) {
    return;
  }

  // vec3 cameraOffset = viewPos - lastViewPos;
  vec3 View = texture2D(position, texCoord).xyz;
  vec4 Projected = vec4(View.xyz, 1.); // + vec4(cameraOffset, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;

  uint seedCamPrev = uint(max(time, 0));
  vec2 smallOffset2 =
      ((vec2(rndf(seedCamPrev), rndf(seedCamPrev))) * 2. - 1.) / iResolution;
  // smallOffset2 = vec2(0.);

  ProjectedCoordinates -= smallOffset2 * 1.;
  vec3 col = retCol(texCoord);
  
  // float roughness = texture2D(color, texCoord).w;
  bool isthesky = false;

  bool outsc = false;
  // float isSpec = texture2D(reflAlb, texCoord).w;

  if (ProjectedCoordinates.x > 1. || ProjectedCoordinates.x < 0. ||
      ProjectedCoordinates.y > 1. || ProjectedCoordinates.y < 0.) {
    ProjectedCoordinates = texCoord;
    outsc = true;
  }

  vec4 n = texture2D(normal, texCoord);

  vec4 nDij = texture2D(prevN, ProjectedCoordinates);
  bool outScreen = outsc;

  float depthWeight = pow(exp(-abs((nDij.w) - (n.w)) * 0.001), 12.);

  float normalWeight = pow(max(dot(nDij.xyz, n.xyz), 0.), 5.);
  float totalWeight = (1.0 - float(outScreen)) * normalWeight * depthWeight;

  float accumulation = texture(prevAcc, ProjectedCoordinates).w;

  accumulation = clamp(accumulation + 1.0, 0.0, 42.);
  accumulation *= totalWeight;

  float frameWeight = (1.0 / max(1.0, accumulation));
 // frameWeight *= clamp(1.-texture2D(prevvar, ProjectedCoordinates).x,0., 1.);

  //

  vec3 prevFrame = texture(prevAcc, ProjectedCoordinates).xyz;
  // vec3 prevFrame = texture2D(prevden, ProjectedCoordinates).xyz;

  // if( isSpec < 0.5){
    vec2 nm = clamp(vec2(lum(col), pow(lum(col), 2.)), 0., 100.);
  vec2 prevnm = clamp(texture2D(prevvar, ProjectedCoordinates).yz, 0., 100.);
  nm = prevnm * (1.0 - frameWeight) + nm * frameWeight;

  col = col * (frameWeight) + (1.0 - frameWeight) * prevFrame;

  //}

  den1 = vec4(col, 7.);
  
  float variance = abs(nm.r - nm.g * nm.g);
    variance = variance * frameWeight + clamp(texture2D(prevvar, ProjectedCoordinates).w, 0., 100.) * (1. - frameWeight);
  float isout = 1.;
  
  tempAccum = vec4(vec3(col), accumulation);
  var1 = vec4(variance, nm, variance);

  vec2 TC = texCoord;
}