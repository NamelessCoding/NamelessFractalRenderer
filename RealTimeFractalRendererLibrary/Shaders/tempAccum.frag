#version 420

layout(location = 0) out vec4 tempAccum;
layout(location = 1) out vec4 den1;
layout(location = 2) out vec4 var1;

in vec2 texCoord;

/*

_swapShader.SetInt("tempWeights", 0);
            _swapShader.SetInt("tempOutL", 1);
            _swapShader.SetInt("tempPos", 2);
            _swapShader.SetInt("spatWe", 3);
            _swapShader.SetInt("spatLO", 4);
*/
/*
_tempAccumShader.SetInt("color", 0);
            _tempAccumShader.SetInt("position", 1);
            _tempAccumShader.SetInt("normal", 2);
            _tempAccumShader.SetInt("albedo", 3);
            _tempAccumShader.SetInt("secondpos", 4);
            _tempAccumShader.SetInt("weigth", 5);
            _tempAccumShader.SetInt("outgoingr", 6);
            _tempAccumShader.SetInt("weightS", 7);
            _tempAccumShader.SetInt("outgoingrS", 8);
            _tempAccumShader.SetInt("prevN", 9);
            _tempAccumShader.SetInt("prevAcc", 10);
*/
uniform sampler2D color;
uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D secondpos;
uniform sampler2D weigth;
uniform sampler2D outgoingr;
uniform sampler2D weightS;
uniform sampler2D outgoingrS;
uniform sampler2D prevN;
uniform sampler2D prevAcc;
uniform sampler2D reflAlb;
uniform sampler2D prevPosition;
uniform sampler2D prevSecondPosition;
uniform sampler2D prevvar;
uniform sampler2D prevden;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 invview;
uniform mat4 invproj;
uniform mat4 prevview;
uniform mat4 prevproj;

uniform vec2 wh;
uniform float time;
uniform vec3 viewPos;
uniform vec3 lastViewPos;
uniform vec3 ldir;

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

vec3 ClipAABB(vec3 q, vec3 aabb_min, vec3 aabb_max) {
  vec3 p_clip = 0.5 * (aabb_max + aabb_min);
  vec3 e_clip = 0.5 * (aabb_max - aabb_min) + 0.00000001;

  vec3 v_clip = q - vec3(p_clip);
  vec3 v_unit = v_clip.xyz / e_clip;
  vec3 a_unit = abs(v_unit);
  float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

  if (ma_unit > 1.0)
    return vec3(p_clip) + v_clip / ma_unit;
  else
    return q;
}

vec2 motionVectorSpecular(vec3 pos, vec2 motionpos, vec2 texcord,
                          vec3 prevpos) {
  vec2 iResolution = wh;

  float stepSize = 4.;
  vec2 bestPos = texcord;
  vec2 centerPos = texcord;
  float bestDist = length(pos - prevpos);
  // vec3 pm = texture2D(color, texCoord - motionpos).xyz;
  vec2 m = texcord - motionpos;
  vec3 prevposm = texture2D(prevSecondPosition, m).xyz;
  if (length(pos - prevposm) < bestDist) {
    bestDist = length(pos - prevposm);
    bestPos = m;
    centerPos = m;
  }

  for (int i = 0; i < 20; i++) {
    vec2 q =
        (centerPos * iResolution + (vec2(1., 0.) * stepSize)) / iResolution;
    vec3 prevposq = texture2D(prevSecondPosition, q).xyz;
    if (length(pos - prevposq) < bestDist) {
      bestDist = length(pos - prevposq);
      bestPos = q;
    }

    q = (centerPos * iResolution + (vec2(-1., 0.) * stepSize)) / iResolution;
    prevposq = texture2D(prevSecondPosition, q).xyz;
    if (length(pos - prevposq) < bestDist) {
      bestDist = length(pos - prevposq);
      bestPos = q;
    }

    q = (centerPos * iResolution + (vec2(0., 1.) * stepSize)) / iResolution;
    prevposq = texture2D(prevSecondPosition, q).xyz;
    if (length(pos - prevposq) < bestDist) {
      bestDist = length(pos - prevposq);
      bestPos = q;
    }

    q = (centerPos * iResolution + (vec2(0., -1.) * stepSize)) / iResolution;
    prevposq = texture2D(prevSecondPosition, q).xyz;
    if (length(pos - prevposq) < bestDist) {
      bestDist = length(pos - prevposq);
      bestPos = q;
    }

    q = (centerPos * iResolution + (vec2(0., 0.) * stepSize)) / iResolution;
    prevposq = texture2D(prevSecondPosition, q).xyz;
    if (length(pos - prevposq) < bestDist) {
      bestDist = length(pos - prevposq);
      bestPos = q;
    }

    if (length(bestPos * iResolution - centerPos * iResolution) < 0.001) {
      if (stepSize == 1.) {
        break;
      }
      stepSize *= 0.5;
    }
    centerPos = bestPos;
  }

  for (int i = 0; i < 9; i++) {
    // if(i == 4){continue;}
    vec2 offset = vec2(float(i % 3) - 1., float(i / 3) - 1.);
    vec3 prevposq = texture2D(prevSecondPosition,
                              (centerPos * iResolution + offset) / iResolution)
                        .xyz;
    if (length(pos - prevposq) < bestDist) {
      bestDist = length(pos - prevposq);
      bestPos = (centerPos * iResolution + offset) / iResolution;
    }
  }
  return bestPos;
}

vec3 proj_point_in_plane(vec3 p, vec3 v0, vec3 n, out float d) {
  d = dot(n, p - v0);
  return p - (n * d);
}

vec3 find_reflection_incident_point(vec3 p0, vec3 p1, vec3 v0, vec3 n) {
  float d0 = 0;
  float d1 = 0;
  vec3 proj_p0 = proj_point_in_plane(p0, v0, n, d0);
  vec3 proj_p1 = proj_point_in_plane(p1, v0, n, d1);

  if (d1 < d0)
    return (proj_p0 - proj_p1) * d1 / (d0 + d1) + proj_p1;
  else
    return (proj_p1 - proj_p0) * d0 / (d0 + d1) + proj_p0;
}

vec3 median5(vec2 uv, vec2 iResolution, float offsetMult) {
  // vec3 arrayCol[9];
  vec3 currentCol = texture2D(reflAlb, uv).xyz;
  // return currentCol;
  vec3 colArray[9] = vec3[](vec3(0.), vec3(0.), vec3(0.), vec3(0.), vec3(0.),
                            vec3(0.), vec3(0.), vec3(0.), vec3(0.));

  float minLum = 9999.;
  int minID = -1;
  int maxID = -1;
  float maxLum = -9999.;
  for (int i = 0; i < 9; i++) {
    vec2 offset = vec2(float(i % 3) - 1., float(i / 3) - 1.) * offsetMult;
    if (i == 4) {
      continue;
    }
    vec2 coords = (uv * iResolution.xy + offset);
    // vec3 currCol = texelFetch(tex, ivec2(coords*0.5),0).rgb;
    // vec3 currCol = texture2D(tex, (coords/iResolution)*0.5).rgb;
    vec3 currCol = texture2D(reflAlb, coords / iResolution).xyz;
    minLum = min(minLum, lum(currCol));
    if (abs(minLum - lum(currCol)) < 0.0001) {
      minID = (i);
    }
    maxLum = max(maxLum, lum(currCol));
    if (abs(maxLum - lum(currCol)) < 0.0001) {
      maxID = (i);
    }
    colArray[i] = currCol;
  }

  if (lum(currentCol) > maxLum && maxID > -1) {
    return colArray[maxID];
  }
  if (lum(currentCol) < minLum && minID > -1) {
    return colArray[minID];
  }

  return currentCol;
  // return pickbetween3(first, second, third);
}

void main() {
  vec2 iResolution = wh;
  if (texture2D(albedo, texCoord).w > 0.5) {
    return;
  }

  vec3 cameraOffset = viewPos - lastViewPos;
  vec3 View = texture2D(position, texCoord).xyz;
  vec4 Projected = vec4(View.xyz, 1.); // + vec4(cameraOffset, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;
  float RW = texture2D(weightS, texCoord).z;
  vec3 col = texture2D(outgoingrS, texCoord).xyz * clamp(RW, 0., 200.);
  float roughness = texture2D(color, texCoord).w;
  bool isthesky = false;

  bool outsc = false;
  float isSpec = texture2D(reflAlb, texCoord).w;

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
  // prevFrame = ClipAABB(prevFrame, minCol, maxCol);

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
  /*
  vec3 currShad = median5(TC, iResolution, 1.);
  vec3 shads = vec3(0.);
  const float atrous_kernel_weights[25] = {
    1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0,
    4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
    6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0,
    4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
    1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0 };

  vec3 currN = texture2D(normal, TC).xyz;
  float wgthSun = 0.;
  for(int i = 0; i < 25; i++){
                  vec2 coords = vec2(float(i % 5) - 2., float(i / 5) - 2.);
          vec2 sampCord = (TC*iResolution + coords)/iResolution;
          vec3 nextN = texture2D(normal, sampCord).xyz;

          //vec3 nextshads = texture2D(reflAlb, sampCord).xyz;
          vec3 nextshads = median5(sampCord, iResolution, 1.);
          float wp = max(pow(max(dot(currN.xyz, nextN.xyz), 0.), 16.), 0.0);
          float wrp = exp(-(abs((lum(currShad)) - lum(nextshads)) / (12.72 +
  0.00001)));

          float currWgth = atrous_kernel_weights[i]*wp*wrp;
          shads += currWgth*nextshads;
          wgthSun += currWgth;

  }

  vec3 sunN = shads/max(wgthSun,0.001);

  colortexsecalb = vec4(sunN, texture2D(reflAlb, texCoord).w);
  */
}