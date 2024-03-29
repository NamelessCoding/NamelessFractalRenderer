#version 420
// colortexFogPrev

layout(binding = 0, Rgba16f) uniform image3D rcpos;

layout(location = 0) out vec4 TAA;
layout(location = 1) out vec4 prevPosition;
layout(location = 2) out vec4 prevSecondPosition;
layout(location = 3) out vec4 colortexFogPrev;

// prevSecondPosition
in vec2 texCoord;

uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D secondpos;

uniform sampler2D Acc;
uniform sampler2D den1;
uniform sampler2D prevTAA;
uniform sampler2D reflAlb;
uniform sampler2D inf; // TODO: inf and holdinfo sample the same texture
uniform sampler2D colorfog;

uniform sampler2D holdinfo;
uniform sampler2D sunTex;
uniform sampler2D skyTex;
uniform sampler2D watpos;
uniform sampler2D watnorm;
uniform sampler3D worl;

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
uniform vec3 lpos;
uniform mat4 lightproj;
uniform mat4 lightview;
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

vec3 angledircos(vec3 n, inout uint r) {
  float r1 = rndf(r);
  float r2 = rndf(r);

  float x = cos(2. * 3.14159 * r1) * sqrt(1. - r2);
  float y = sin(2. * 3.14159 * r1) * sqrt(1. - r2);
  float z = sqrt(r2);

  vec3 W = (abs(n.x) > 0.99) ? vec3(0., 1., 0.) : vec3(1., 0., 0.);
  vec3 N = n;
  vec3 T = normalize(cross(N, W));
  vec3 B = cross(T, N);
  return normalize(x * T + y * B + z * N);
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

#define RESOLUTION 0.5
const float phi2 = 1.32471795724474602596;
const vec2 a = vec2(1.0 / phi2, 1.0 / (phi2 * phi2));

vec2 R2(float n) { return fract(a * n + 0.5); }

vec2 jitter() {
  return (R2(float(int(time) % 1000)) * 1.2 - 0.1) / (wh * 2.0 * RESOLUTION);
}
float lum(vec3 c) { return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z; }

vec3 pickbetween3(vec3 a, vec3 b, vec3 c) {
  float lm1 = lum(a);
  float lm2 = lum(b);
  float lm3 = lum(c);

  float lowest = min(lm1, min(lm2, lm3));
  float maxest = max(lm1, max(lm2, lm3));

  if (lowest == lm1) {
    if (maxest == lm2) {
      return c;
    } else {
      return b;
    }
  }

  if (lowest == lm2) {
    if (maxest == lm1) {
      return c;
    } else {
      return a;
    }
  }

  if (lowest == lm3) {
    if (maxest == lm1) {
      return b;
    } else {
      return a;
    }
  }

  return a;
}

vec3 median3(vec2 uv, vec2 iResolution, float offsetMult) {
  // vec3 arrayCol[9];
  vec3 currentCol = texture2D(colorfog, uv).xyz;
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
    vec3 currCol = texture2D(colorfog, coords / iResolution).xyz;
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

vec3 median4(vec2 uv, vec2 iResolution, float offsetMult) {
  // vec3 arrayCol[9];
  vec3 currentCol = texture2D(den1, uv).xyz;
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
    vec3 currCol = texture2D(den1, coords / iResolution).xyz;
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

vec3 rgbtohsv(vec3 col) {
  float cmax = max(max(col.x, col.y), col.z);
  float cmin = min(min(col.x, col.y), col.z);
  float delta = cmax - cmin;
  float H = 0.;
  if (delta < 0.0001) {
    H = 0.;
  } else if (cmax == col.x) {
    H = 60. * mod((col.g - col.b) / delta, 6.);
  } else if (cmax == col.g) {
    H = 60. * ((col.b - col.r) / delta + 2.);
  } else if (cmax == col.b) {
    H = 60. * ((col.r - col.g) / delta + 4.);
  }
  float S = 0.;
  if (cmax > 0.) {
    S = delta / cmax;
  }
  float V = cmax;

  return vec3(clamp(H, 0., 360.), clamp(S, 0., 1.), clamp(V, 0., 1.));
}

vec3 hsvtorgb(vec3 hsv) {

  float C = hsv.b * hsv.g;
  float X = C * (1. - abs(mod(hsv.r / 60., 2.) - 1.));
  float m = hsv.b - C;

  vec3 rgb = vec3(0.);
  float H = hsv.r;
  if (H < 60.) {
    rgb = vec3(C, X, 0.);
  } else if (H < 120.) {
    rgb = vec3(X, C, 0.);
  } else if (H < 180.) {
    rgb = vec3(0., C, X);
  } else if (H < 240.) {
    rgb = vec3(0., X, C);
  } else if (H < 300.) {
    rgb = vec3(X, 0., C);
  } else if (H < 360.) {
    rgb = vec3(C, 0., X);
  }

  return rgb + m;
}

vec4 rgbtoCMYK(vec3 col) {
  float K = max(1. - max(col.r, max(col.g, col.b)), 0.);
  float C = (1. - col.r - K) / (1. - K);
  float M = (1. - col.g - K) / (1. - K);
  float Y = (1. - col.b - K) / (1. - K);
  return vec4(C, M, Y, K);
}

vec3 CMYKtorgb(vec4 col) {
  col = clamp(col, 0., 1.);
  float R = (1. - col.x) * (1. - col.w);
  float G = (1. - col.g) * (1. - col.w);
  float B = (1. - col.b) * (1. - col.w);
  return vec3(R, G, B);
}

vec3 fresnelSchlick(vec3 F0, vec3 F90, float LdotH) {
  return F0 + (max(F0, F90) - F0) * pow(1. - LdotH, 5.);
}

float random3d(vec3 p) {
  return fract(sin(p.x * 214. + p.y * 241. + p.z * 123.) * 100. +
               cos(p.x * 42. + p.y * 41.2 + p.z * 32.) * 10.);
}

vec3 shadowS(vec3 pos) {
  vec4 sspace = lightproj * lightview * vec4(pos, 1.);
  sspace.xyz /= sspace.w;

  vec3 coords = sspace.xyz * 0.5 + 0.5;

  // float currDepth = texelFetch(sunTex, ivec2(2048.*coords), 0).r;
  float currDepth = texture2D(sunTex, coords.xy).r;
  // vec3 currPos = texture2D(sunTex, coords.xy).xyz;
  vec3 col = vec3(1.);
  if (coords.z > currDepth + 0.001 && coords.x >= 0. && coords.x <= 1. &&
      coords.y >= 0. && coords.y <= 1.) {
    col = vec3(0.);
  }

  if (length(pos.xz - viewPos.xz) > 90.) {
    //  col = vec3(1.);
  }

  return col;
}
float PM(float cost, float g) {
  float a = 3. / (8. * 3.14159);
  float b = (1.0 - g * g) * (1.0 + cost * cost);
  float c = (2.0 + g * g) * pow(1.0 + g * g - 2. * g * cost, 3. / 2.);
  return a * (b / c);
}
float PR(float cost) { return (3. / (16. * 3.14159)) * (1.0 + cost * cost); }

vec4 fog(vec3 pos, vec3 dir, vec2 TC, inout uint r) {
  vec3 cam = viewPos;
  vec3 finPos = cam;
  float mov = 1.;

  if (texture2D(albedo, TC).w > 0.5) {
    pos = cam + dir * 20.;
  }

  vec3 accum = vec3(0.);
  vec3 direction = (pos - cam) / 20.;
  vec3 reversePos = pos;
  vec3 volCol = vec3(0.), volAbs = vec3(1.);
  vec3 stepAbs = vec3(1.) * exp(-0.05 * length(direction));

  vec3 stepCol = (vec3(1.) - stepAbs);
  vec3 stepCol2 = (vec3(1.) - stepAbs);
  float octave = 0.;

  float absorb = 1.;

  vec3 rayDirection = normalize(direction.xzy);
  rayDirection.z = abs(rayDirection.z);
  rayDirection = normalize(rayDirection);
  vec2 TX = vec2((atan(rayDirection.z, rayDirection.x) /
                  6.283185307179586476925286766559) +
                     0.5,
                 acos(rayDirection.y) / 3.1415926535897932384626433832795);
  //                albedos = texture2D(skytex, TX).rgb*4.;
  float ldirsss = max(1. - ldir.y, 0.00000001);
  vec3 skyCol = textureLod(skyTex, TX, ceil(log2(max(wh.x, wh.y)) * 0.5)).xyz;
  float acci = 0.;
  for (int i = 0; i < 20; i++) {
    finPos += direction * max(rndf(r), 0.5);
    reversePos -= direction * max(rndf(r), 0.5);
    float pm2 = (PM(max(dot(dir, ldir), 0.), 0.76) +
                 PM(max(dot(dir, ldir), 0.), pow(0.9, octave))) *
                1.;
    // float pm2 = PM(max(dot(d,lig),0.),0.76);
    float dens = max(texture(worl, reversePos * .01).x - 0.7, 0.);
    dens = clamp(dens, 0., 1.);
    dens = 1.0 - pow(1.0 - dens, 4.);
    acci += dens;
    float pr2 = PR(max(dot(dir, ldir), 0.));
    vec3 shadow = shadowS(finPos);
    // accum += shadowS(finPos);
    if (length(shadow) > 0.01) {
      octave += 1.;

      accum += octave * 0.9 * stepCol * volAbs * shadow * pm2 * pr2 * 0.0001 *
               mix(vec3(0.9, 0.5, 0.2), vec3(0.8, 0.8, 0.7),
                   pow(max(dot(ldir, vec3(0., 1., 0.)), 0.), 1.));
    }
    float accum2 = length(cam - finPos);
    float heigh = exp(-max(reversePos.y - (mov + 1.), 0.) * .4);
    accum += stepCol * volAbs * .004 * dens * max(shadow, 0.5) * 1.5 *
             exp(-acci * 5.);

    accum +=
        stepCol * volAbs * .8 * max(shadow, 0.5) * dens * 1.5 * heigh * skyCol;

    // volAbs *= exp(-length(finPos - cam)*.025);
    volAbs *= exp(-max(dens, 0.05) * heigh * length(finPos - cam) * .01);

    absorb *= 1.0 - exp(-0.7 * (1.0 - max(dot(ldir, vec3(0., 1., 0.)), 0.)) *
                        length(finPos - cam));
  }

  return vec4(accum * pow(1. - ldir.y, 4.), volAbs);
}

void main() {
  vec2 iResolution = wh;
  vec2 fragCoord = texCoord * wh;

  uint r = uint(uint(fragCoord.x) * uint(1973) +
                uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) |
           uint(1);

  vec2 TC = floor((texCoord * iResolution.xy) * RESOLUTION) /
            (RESOLUTION * iResolution.xy);
  //  TC += jitter();
  TC = texCoord;

  vec4 wp = texture2D(watpos, TC);
  vec4 wn = texture2D(watnorm, TC);
  vec2 Albdiff = TC;
  if (wp.w > 0.5) {
    Albdiff += wn.xz * 0.05;
  }

  uint seedCam = uint(time);
  vec2 smallOffset =
      ((vec2(rndf(seedCam), rndf(seedCam))) * 2. - 1.) / iResolution;

  // TC -= smallOffset*4.;
  // TC += jitter();
  // vec2 TC = texCoord*1.;
  vec3 View = texture2D(position, TC).xyz;

  vec4 p22 = vec4(((texCoord)*2.0 - 1.0), 0.0, 1.0);
  vec3 dir = (invproj * p22).xyz / (invproj * p22).w;
  dir = normalize(mat3(invview) * dir);
  vec3 wi = dir;
  if (texture2D(albedo, TC).w > 0.5) {
    // View = viewPos + dir*180.;
  }

  // uint seedCam = uint(time);
  // vec3 off = (vec3(rndf(seedCam),rndf(seedCam),rndf(seedCam))*2.0-1.0)*0.002;
  // vec3 off = (vec3(rndf(seedCam),rndf(seedCam),rndf(seedCam))*2.0-1.0)*0.02;
  vec4 Projected = vec4(View.xyz, 1.); // - vec4(off, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  // Projected.xy -= smallOffset;

  vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;

  uint seedCamPrev = uint(max(time, 0));
  vec2 smallOffset2 =
      ((vec2(rndf(seedCamPrev), rndf(seedCamPrev))) * 2. - 1.) / iResolution;
  // smallOffset2 = vec2(0.);

  ProjectedCoordinates -= smallOffset2 * 1.;

  ProjectedCoordinates *= 1.0;
  ProjectedCoordinates = max(ProjectedCoordinates, 0.);
  if (texture2D(albedo, TC).w > 0.5) {
    // View = viewPos + dir*180.;
    ProjectedCoordinates = texCoord;
  }

  // ProjectedCoordinates  = floor( (ProjectedCoordinates * iResolution.xy) *
  // RESOLUTION ) / (RESOLUTION * iResolution.xy); ProjectedCoordinates +=
  // jitter();
  vec3 Albedo = pow(texture2D(albedo, Albdiff).xyz, vec3(2.2));

  Albedo = rgbtohsv(Albedo);
  Albedo.g *= 1.;
  // Albedo.r *= 1.2;
  Albedo = hsvtorgb(vec3(clamp(Albedo.x, 0., 360.), clamp(Albedo.y, 0., 1.),
                         clamp(Albedo.z, 0., 1.)));

  vec3 nrm = texture2D(normal, TC).xyz;
  vec3 F2 = fresnelSchlick(vec3(0.5), vec3(1.), max(dot(-dir, nrm), 0.));
  // Albedo = vec3(1.);
  vec3 col = texture2D(den1, TC).xyz; //*(Albedo/3.14159);
  // vec3 median(vec2 uv, vec2 iResolution, sampler2D tex, float offsetMult){
  // vec3 ind = median(TC, iResolution, 1.);
  vec3 ind = col;
  vec3 diff = median4(Albdiff, iResolution, 1.);
  col = diff * (Albedo / 3.14159);

  // if(texture2D(reflnorm, TC).w > 0.5){
  //  col = texture2D(den1, TC).xyz*(Albedo/3.14159);
  //}

  // SHADOWS
  col += clamp(texture2D(reflAlb, TC).xyz, 0., 10.);
  // vec3 currShad = texture2D(reflAlb, TC).xyz;
  vec3 spec = median3(TC, iResolution, 1.);
  col += clamp(spec * F2, 0., 1000.);

  col *= vec3(texture2D(colorfog, TC).w);

  vec3 watPOS = View;
  if (wp.w > 0.5) {
    // do water
    vec3 F22 = clamp(fresnelSchlick(vec3(0.04), vec3(0.99),
                                    max(dot(normalize(wn.xyz), -wi), 0.)),
                     0., 1.);
    vec3 posWat = wp.xyz;
    watPOS = posWat;
    vec3 posNorm = texture2D(position, TC).xyz;

    float DP = length(posWat - posNorm);
    vec3 watCol =
        mix(vec3(0.05, 0.1, 0.2), vec3(0.2, 0.6, 0.9), exp(-DP * 0.5));
    col = (diff * (Albedo / 3.14159) * 2. +
           clamp(texture2D(reflAlb, TC).xyz, 0., 10.) * 0.1) *
              (1. - F22) * exp(-DP * 2.5) * 2. * watCol +
          clamp(spec * F22, 0., 1000.);

    if (wn.w > 0.5) {
      col = vec3(0.);
    }
  }

  if (texture2D(albedo, TC - smallOffset).w > 0.5 ||
      texture2D(position, TC - smallOffset).w > 0.5) {
    col = texture2D(albedo, TC - smallOffset).xyz;
  }
  ind = col;

  // vec4 fog(vec3 pos, vec3 dir, vec2 TC, inout uint r){

  vec4 fogs = fog(watPOS, wi, TC, r);
  col = col * fogs.w + pow(fogs.xyz * max(ldir.y * ldir.y, 0.5), vec3(1.));

  /// Keep for possible emissive volumetrics
  /*
  vec3 PP = texture2D(position, TC).xyz;
  vec3 cam = viewPos;
  if(texture2D(albedo, TC).w > 0.5){
      PP = cam + wi * 100.;
  }

  vec3 div = (PP-cam)/50.;

  vec3 volCol = vec3(0.), volAbs = vec3(1.);

  vec3 accum = vec3(0.);
  for(int i = 0; i < 50; i++){
      vec3 OFFS = vec3(rndf(r), rndf(r), rndf(r))*2.-1.;

      cam += div*max(random3d(cam),0.5);
      vec3 sampPos = ((cam+OFFS*0.5)+90.0 - floor(viewPos));
      //accum += imageLoad(rcpos, ivec3(sampPos)).xyz;
      float dens = (imageLoad(rcpos, ivec3(sampPos)).w);

      accum += volAbs*(imageLoad(rcpos, ivec3(sampPos)).xyz)*0.3*exp(
  -4.73*max(1.-dens/40.,0.001)) + length(cam - viewPos)*0.0000001; volAbs *=
  exp(-length(cam - viewPos)*0.1*max(dens,0.1)*length(cam - viewPos)*.001);
      //rcpos
  }
  col = col*volAbs.x + accum*0.2;
  */

  float depth = texture2D(normal, TC).w;

  vec3 prevCol = clamp(texture2D(prevTAA, ProjectedCoordinates).xyz, 0., 10.);
  vec3 minCol = (vec3(9999.));
  vec3 maxCol = (vec3(0.));
  for (int i = 0; i < 9; i++) {
    vec2 coords = vec2(float(i % 3) - 1., float(i / 3) - 1.);
    vec3 currSample =
        clamp(texture2D(prevTAA, (ProjectedCoordinates * iResolution + coords) /
                                     iResolution)
                  .xyz,
              0., 10.);
    minCol = min(minCol, currSample);
    maxCol = max(maxCol, currSample);
  }
  prevCol = ClipAABB(prevCol, minCol, maxCol);

  // if(ProjectedCoordinates.x >= 0. && ProjectedCoordinates.x <= 1. &&
  // ProjectedCoordinates.y >= 0. && ProjectedCoordinates.y <= 1.0){
  vec2 velocity = (TC - ProjectedCoordinates.xy) * iResolution;

  // blend only if we are inside of bounds
  float blendFactor =
      float(ProjectedCoordinates.x > 0.0 && ProjectedCoordinates.x < 1.0 &&
            ProjectedCoordinates.y > 0.0 && ProjectedCoordinates.y < 1.0);
  blendFactor *= exp(-length(velocity) * 0.9) * 0.6 + 0.3;

  // if(texture2D(color, TC).w > 0.5){
  if (texture2D(albedo, TC - smallOffset).w < 0.5 &&
      texture2D(position, TC - smallOffset).w < 0.5) {
    col = mix(col, prevCol, clamp(blendFactor, 0.0, 1.));
  }
  //}

  //}

  TAA = vec4(col, clamp(sqrt(lum(ind)) * 0.1 +
                            texture2D(prevTAA, ProjectedCoordinates).w * 0.9,
                        0., 0.99));

  prevPosition = texture2D(position, TC);
  prevSecondPosition = texture2D(secondpos, TC);
}