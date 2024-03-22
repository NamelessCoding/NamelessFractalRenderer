
out vec4 outputColor;
in vec2 texCoord;

layout(binding = 0, Rgba16f) uniform image3D rcpos;

uniform sampler2D color;
uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D secondpos;
/*
_finalShader.SetInt("weigth", 5);
            _finalShader.SetInt("outgoingr", 6);
*/
uniform sampler2D weigth;
uniform sampler2D outgoingr;
uniform sampler2D weightS;
uniform sampler2D outgoingrS;
uniform sampler2D Acc;
uniform sampler2D den1;
uniform sampler2D var1;
uniform sampler2D TAA;
uniform sampler2D colorfog;
// colortexFog
uniform sampler2D colortexFog;

uniform sampler2D fogw;
uniform sampler2D fogLO;
uniform sampler2D sunTex;
uniform sampler2D watpos;
/*
      _finalShader.SetInt("fogw", 16);
            _finalShader.SetInt("fogLO", 17);
*/

uniform mat4 view;
uniform mat4 projection;
uniform mat4 invview;
uniform mat4 invproj;
uniform vec2 wh;
uniform float time;
uniform vec3 viewPos;

uniform float brightness;
uniform vec3 ldir;
uniform mat4 lightproj;
uniform mat4 lightview;
uniform vec3 lpos;

uniform mat4 linvview;
uniform mat4 linvproj;

vec3 tonemap_uchimura2(vec3 v) {
  const float P = 1.0;  // max display brightness
  const float a = 1.;   // contrast
  const float m = 0.1;  // linear section start
  const float l = 0.0;  // linear section length
  const float c = 1.33; // black
  const float b = 0.0;  // pedestal

  float l0 = ((P - m) * l) / a;
  float L0 = m - m / a;
  float L1 = m + (1.0 - m) / a;
  float S0 = m + l0;
  float S1 = m + a * l0;
  float C2 = (a * P) / (P - S1);
  float CP = -C2 / P;

  vec3 w0 = 1.0 - smoothstep(0.0, m, v);
  vec3 w2 = step(m + l0, v);
  vec3 w1 = 1.0 - w0 - w2;

  vec3 T = m * pow(v / m, vec3(c)) + vec3(b);
  vec3 S = P - (P - S1) * exp(CP * (v - S0));
  vec3 L = m + a * (v - vec3(m));

  return T * w0 + L * w1 + S * w2;
}
//////////////////////////////////

float L(float x, float a) {
  if (x == 0.0) {
    return 1.0;
  }
  if (x != 0.0 && x < a && x > -a) {
    return (a * sin(3.14159 * x) * sin((3.14159 * x) / a)) /
           (pow(3.14159, 2.) * x * x);
  }
  return 0.0;
}

vec3 upscaleIndirect(vec2 iResolution, vec2 texcc) {
  vec3 col = vec3(0.);
  float weight = 0.;
  for (int i = 0; i < 25; i++) {
    vec2 coords = vec2(float(i % 5) - 2., float(i / 5) - 2.);
    vec2 newCords = (texcc * iResolution + coords) / iResolution;

    float L_W = L(length(coords) / 1., 2.0);
    col += texture2D(TAA, newCords).xyz * L_W;
    // weight += L_W;
  }
  return col / max(1., 0.001);
}

vec3 blur2(vec2 p, float dist, vec2 iResolution) {
  // vec2 iResolution = vec2(viewWidth, viewHeight);

  p *= iResolution.xy;
  vec3 s = vec3(0.);

  vec3 div = vec3(0.);
  // vec2 off = vec2(0.0, r);
  float k = 0.61803398875;
  for (int i = 0; i < 150; i++) {
    float m = float(i) * 0.01;
    float r = 2. * 3.14159 * k * float(i);
    vec2 coords = vec2(m * cos(r), m * sin(r)) * dist;
    // vec4 c2 = texture2D(iChannel0, (p+coords)/iResolution.xy).xyzw;
    vec2 cir = (p + coords) / iResolution.xy;
    vec3 ccc = texture2D(albedo, cir).xyz;

    vec3 c = upscaleIndirect(iResolution, cir);
    /*vec3 c = texture2D(TAA, cir).xyz*5.;
    vec3 c1 =texture2D(TAA, (cir*iResolution +
    vec2(1.,0.))/iResolution).xyz*-1.; vec3 c2 =texture2D(TAA, (cir*iResolution
    + vec2(-1.,0.))/iResolution).xyz*-1.; vec3 c3 =texture2D(TAA,
    (cir*iResolution + vec2(0.,1.))/iResolution).xyz*-1.; vec3 c4
    =texture2D(TAA, (cir*iResolution + vec2(0.,-1.))/iResolution).xyz*-1.;
    c = c + c1 + c2 + c3 + c4;*/
    // texture2D(albedo, cir).w > 0.5 ||
    // if(texture2D(position, cir).w > 0.5){
    //      c = ccc;
    // }

    // c = c*c *1.8;
    // vec3 bok = pow(c,vec3(4.));
    vec3 bok = vec3(1.);
    s += c * bok;
    div += bok;
  }

  s /= div;

  return s;
}

/*
CoC = abs(aperture * (focallength * (objectdistance - planeinfocus)) /
(objectdistance * (planeinfocus - focallength)))

*/
// https://developer.nvidia.com/gpugems/gpugems/part-iv-image-processing/chapter-23-depth-field-survey-techniques
float CoC(float aperture, float focallength, float objectdistance,
          float planeinfocus) {
  return abs(aperture * (focallength * (objectdistance - planeinfocus)) /
             max(objectdistance * (planeinfocus - focallength), 0.0001));
}

float newCOC(float As, float focalLength, float depth, float depthCenter) {

  return As * (abs(depth - depthCenter) / max(depth, 0.01)) *
         (focalLength / max(depthCenter - focalLength, 0.001));
}

const vec2 pos[12] = {vec2(0.326212, 0.40581),  vec2(0.840144, 0.07358),
                      vec2(0.695914, 0.457137), vec2(0.203345, 0.620716),
                      vec2(0.96234, 0.194983),  vec2(0.473434, 0.480026),
                      vec2(0.519456, 0.767022), vec2(0.185461, 0.893124),
                      vec2(0.507431, 0.064425), vec2(0.89642, 0.412458),
                      vec2(0.32194, 0.932615),  vec2(0.791559, 0.59771)};

float getBlurSize(float depth, float focusPoint, float focusScale,
                  float maxmult) {
  float coc = clamp((1.0 / focusPoint - 1.0 / depth) * focusScale, -1.0, 1.0);
  return abs(coc) * maxmult;
}

vec3 blur3(vec2 p, float centerDepth, float focalLength, vec2 iResolution,
           inout uint r, bool mn) {
  // vec2 iResolution = vec2(viewWidth, viewHeight);

  float TotalContribution = 0.;
  vec3 ColorSum = vec3(0.);
  p *= iResolution.xy;
  float mm = iResolution.y / iResolution.x;

  float currentDepth = texture2D(normal, p / iResolution).w;
  /// float cocSizeS =  clamp(CoC(mm, focalLength, currentDepth, centerDepth),
  /// 0.0001, 1. );
  // float newCOC(float As, float focalLength, float depth, float depthCenter){
  // float cocSizeS = clamp(newCOC(mm, focalLength, currentDepth, centerDepth),
  // 0., 1.);

  float multw = currentDepth;
  float maxmult = 24;
  float maxfin = 13;

  if (mn) {
    multw = 1.;
    maxmult = 94.;
    maxfin = 24.;
  }

  float cocSizeS = clamp(
      getBlurSize(currentDepth, centerDepth, focalLength, maxmult) * multw, 0.,
      maxfin);
  float keepCOC = cocSizeS;
  float CoCSize = cocSizeS;
  vec3 cul = texture2D(fogLO, p / iResolution).rgb;

  for (int i = 0; i < 150; i++) {
    float k = 0.73;
    // for(int i = 0; i < 150; i++){
    float m = sqrt(float(i) / 150.);
    float r = 2. * 3.14159 * k * float(i);
    vec2 coords = vec2(m * cos(r), m * sin(r)) * CoCSize;
    // float round = (float(i)/11.)*2.-1.;
    // vec2 coords = vec2(cos(round),
    // sin(round))*(floor(float(i/11.))/4.)*cocSizeS;
    // vec2 cir = (p + coords) / iResolution.xy;

    vec2 sampCoords = (p + coords) / iResolution;

    float Depth = texture2D(normal, sampCoords).w;

    vec2 uv2 = sampCoords;
    vec3 rad2 = vec3(0.);
    vec2 offset2 = (sampCoords * iResolution - iResolution.xy / 2.) * 1.;
    float dist = CoCSize * 0.08;
    for (int is = 0; is < 10; is++) {
      vec2 offset =
          sampCoords * iResolution +
          offset2 *
              smoothstep(0., 15. - length(uv2 * 2.0 - 1.) * 1.5 + dist,
                         float(is) / 20.) *
              1.;
      rad2.x +=
          texture2D(fogLO, (offset + offset2 * 0.0064 * dist) / iResolution.xy)
              .x;
      rad2.y += texture2D(fogLO, (offset) / iResolution.xy).y;
      rad2.z +=
          texture2D(fogLO, (offset - offset2 * 0.0064 * dist) / iResolution.xy)
              .z;
    }
    rad2 /= 10.;

    vec3 currCol = rad2;
    // vec3 currCol = texture2D(fogLO, sampCoords).rgb;

    // float CoC(float aperture, float focallength, float objectdistance, float
    // planeinfocus){ float CoCSize = clamp(CoC(mm, focalLength, Depth,
    // centerDepth),0.0001,1.); float CoCSize = clamp(newCOC(mm, focalLength,
    // Depth, centerDepth),0.,1.);

    multw = Depth;
    maxmult = 24;
    maxfin = 13;

    if (mn) {
      multw = 1.;
      maxmult = 94.;
      maxfin = 24.;
    }

    CoCSize =
        clamp(getBlurSize(Depth, centerDepth, focalLength, maxmult) * multw, 0.,
              maxfin);

    // if sample depth > center depth then
    //	sample size = clamp(sample size, 0, center size)
    // end
    if (Depth > centerDepth) {
      CoCSize = clamp(CoCSize, 0., keepCOC);
    }

    float SampleContribution = currentDepth > Depth ? 1.0 : CoCSize;

    ColorSum += mix(cul, currCol, clamp(SampleContribution, 0., 1.));
    TotalContribution += 1.;
  }

  vec3 finC = ColorSum / max(TotalContribution, 0.001);
  if (mn) {
    return finC;
  }
  // return mix(finC, cul,  clamp(exp(-getBlurSize(currentDepth, centerDepth,
  // focalLength)*currentDepth*centerDepth*0.01),0.,1.));
  float diff = exp(-abs(currentDepth - centerDepth));
  return mix(finC, cul, diff);
}

vec3 blur23(vec2 p, float dist, vec2 iResolution) {

  p *= iResolution.xy;
  vec3 s = vec3(0.);

  vec3 div = vec3(0.);
  // vec2 off = vec2(0.0, r);
  float k = 0.61803398875;
  for (int i = 0; i < 20; i++) {
    float m = 1.;
    float r = 2. * 3.14159 * k * float(i);
    vec2 coords = vec2(m * cos(r), m * sin(r)) * dist;
    // vec4 c2 = texture2D(iChannel0, (p+coords)/iResolution.xy).xyzw;
    vec2 cir = (p + coords) / iResolution.xy;

    // vec3 c = texture2D(colortex5, cir).xyz;
    vec2 uv2 = cir;
    vec3 rad2 = vec3(0.);
    vec2 offset2 = (cir * iResolution - iResolution.xy / 2.) * 1.;
    for (int i = 0; i < 20; i++) {
      vec2 offset =
          cir * iResolution +
          offset2 *
              smoothstep(0., 15. - length(uv2 * 2.0 - 1.) * 1.5 + dist,
                         float(i) / 20.) *
              1.;
      rad2.x +=
          texture2D(TAA, (offset + offset2 * 0.0064 * dist) / iResolution.xy).x;
      rad2.y += texture2D(TAA, (offset) / iResolution.xy).y;
      rad2.z +=
          texture2D(TAA, (offset - offset2 * 0.0064 * dist) / iResolution.xy).z;
    }
    rad2 /= 16.;

    vec3 c = rad2 * 0.8;
    // c = c*c *1.8;
    // vec3 bok = pow(c,vec3(4.));
    vec3 bok = vec3(1.);
    s += c * bok;
    div += bok;
  }

  s /= div;

  return s;
}

// NOT MY CODE//////////////////////
vec3 ACESFilm(vec3 x) {
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0., 1.);
}

/***** RCAS *****/
#define FSR_RCAS_LIMIT (0.25 - (1.0 / 16.0))
//#define FSR_RCAS_DENOISE

// Input callback prototypes that need to be implemented by calling shader
vec4 FsrRcasLoadF(vec2 p);
//------------------------------------------------------------------------------------------------------------------------------
void FsrRcasCon(out float con,
                // The scale is {0.0 := maximum, to N>0, where N is the number
                // of stops (halving) of the reduction of sharpness}.
                float sharpness) {
  // Transform from stops to linear value.
  con = exp2(-sharpness);
}

vec3 FsrRcasF(vec2 ip, // Integer pixel position in output.
              float con) {
  // Constant generated by RcasSetup().
  // Algorithm uses minimal 3x3 pixel neighborhood.
  //    b
  //  d e f
  //    h
  vec2 sp = vec2(ip);
  vec3 b = FsrRcasLoadF(sp + vec2(0, -1)).rgb;
  vec3 d = FsrRcasLoadF(sp + vec2(-1, 0)).rgb;
  vec3 e = FsrRcasLoadF(sp).rgb;
  vec3 f = FsrRcasLoadF(sp + vec2(1, 0)).rgb;
  vec3 h = FsrRcasLoadF(sp + vec2(0, 1)).rgb;
  // Luma times 2.
  float bL = b.g + .5 * (b.b + b.r);
  float dL = d.g + .5 * (d.b + d.r);
  float eL = e.g + .5 * (e.b + e.r);
  float fL = f.g + .5 * (f.b + f.r);
  float hL = h.g + .5 * (h.b + h.r);
  // Noise detection.
  float nz = .25 * (bL + dL + fL + hL) - eL;
  nz = clamp(abs(nz) / (max(max(bL, dL), max(eL, max(fL, hL))) -
                        min(min(bL, dL), min(eL, min(fL, hL)))),
             0., 1.);
  nz = 1. - .5 * nz;
  // Min and max of ring.
  vec3 mn4 = min(b, min(f, h));
  vec3 mx4 = max(b, max(f, h));
  // Immediate constants for peak range.
  vec2 peakC = vec2(1., -4.);
  // Limiters, these need to be high precision RCPs.
  vec3 hitMin = mn4 / (4. * mx4);
  vec3 hitMax = (peakC.x - mx4) / (4. * mn4 + peakC.y);
  vec3 lobeRGB = max(-hitMin, hitMax);
  float lobe =
      max(-FSR_RCAS_LIMIT, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.)) *
      con;
// Apply noise removal.
#ifdef FSR_RCAS_DENOISE
  lobe *= nz;
#endif
  // Resolve, which needs the medium precision rcp approximation to avoid
  // visible tonality changes.
  return (lobe * (b + d + h + f) + e) / (4. * lobe + 1.);
}

vec4 FsrRcasLoadF(vec2 p) { return texture2D(TAA, p / (wh)); }
float lum(vec3 c) {
  return sqrt(0.299 * c.x * c.x + 0.587 * c.y * c.y + 0.114 * c.z * c.z);
}

vec3 sharpn(vec2 iResolution) {
  float str = 1.;
  float strength = mix(-1. / 9., -1. / 6., str);
  // vec3 col = texture2D(TAA, texCoord).xyz;
  vec3 minCol = (vec3(9999.));
  vec3 maxCol = (vec3(-9999.));
  float minPixel = 9999.;
  float maxPixel = -9999.;
  for (int i = 0; i < 9; i++) {
    vec2 offset = vec2(float(i % 3), float(i / 3)) - 1.;
    vec2 newCords = (texCoord * iResolution + offset) / iResolution;
    // if(i != 4){
    vec3 currSample = (texture2D(TAA, newCords).xyz);
    minCol = min(minCol, currSample);
    minPixel =
        min(min(currSample.x, min(currSample.y, currSample.z)), minPixel);
    maxPixel =
        max(max(currSample.x, max(currSample.y, currSample.z)), maxPixel);

    maxCol = max(maxCol, currSample);
    //}
  }
  // float amplitude = min(lum(minCol), 2.0-lum(maxCol))/max(lum(maxCol),0.001);
  float amplitude =
      min((minPixel), max(2.0 - (maxPixel), 0.)) / max((maxPixel), 0.001);
  float amp = amplitude * strength;

  vec3 c1 = texture2D(TAA, texCoord).xyz * 1.;
  vec3 c2 =
      texture2D(TAA, (texCoord * iResolution + vec2(1., 0.)) / iResolution)
          .xyz *
      amp;
  vec3 c3 =
      texture2D(TAA, (texCoord * iResolution + vec2(-1., 0.)) / iResolution)
          .xyz *
      amp;
  vec3 c4 =
      texture2D(TAA, (texCoord * iResolution + vec2(0., 1.)) / iResolution)
          .xyz *
      amp;
  vec3 c5 =
      texture2D(TAA, (texCoord * iResolution + vec2(0., -1.)) / iResolution)
          .xyz *
      amp;

  vec3 c = c1 + c2 + c3 + c4 + c5;

  return c / max(1.0 + 4. * amp, 0.001);
}

vec3 sharpn2(vec2 iResolution) {
  float amp = -1.;

  vec3 c1 = texture2D(TAA, texCoord).xyz * 5.;
  vec3 c2 =
      texture2D(TAA, (texCoord * iResolution + vec2(1., 0.)) / iResolution)
          .xyz *
      amp;
  vec3 c3 =
      texture2D(TAA, (texCoord * iResolution + vec2(-1., 0.)) / iResolution)
          .xyz *
      amp;
  vec3 c4 =
      texture2D(TAA, (texCoord * iResolution + vec2(0., 1.)) / iResolution)
          .xyz *
      amp;
  vec3 c5 =
      texture2D(TAA, (texCoord * iResolution + vec2(0., -1.)) / iResolution)
          .xyz *
      amp;

  vec3 c = c1 + c2 + c3 + c4 + c5;

  return c;
}

vec3 sRGB(vec3 t) {
  return mix(1.055 * pow(t, vec3(1. / 2.4)) - 0.055, 12.92 * t,
             step(t, vec3(0.0031308)));
}

#define ROT(a) mat2(cos(a), sin(a), -sin(a), cos(a))

#define PI 3.141592654
#define TAU (2.0 * PI)

const mat2 brot = ROT(2.399);
// License: Unknown, author: Dave Hoskins, found: Forgot where
vec3 dblur(vec2 q, float rad) {
  vec2 RESOLUTION = wh;
  vec3 acc = vec3(0);
  const float m = 0.0025;
  vec2 pixel = vec2(m * RESOLUTION.y / RESOLUTION.x, m);
  vec2 angle = vec2(0, rad);
  rad = 1.;
  const int iter = 30;
  for (int j = 0; j < iter; ++j) {
    rad += 1. / rad;
    angle *= brot;
    vec4 col = texture2D(TAA, q + pixel * (rad - 1.) * angle);
    acc += clamp(col.xyz, 0.0, 10.0);
  }
  return acc * (1.0 / float(iter));
}

#define FXAA_SPAN_MAX 8.0
#define FXAA_REDUCE_MUL (1.0 / FXAA_SPAN_MAX)
#define FXAA_REDUCE_MIN (1.0 / 128.0)
#define FXAA_SUBPIX_SHIFT (1.0 / 4.0)

vec3 FxaaPixelShader(vec4 uv, sampler2D tex, vec2 rcpFrame) {

  vec3 rgbNW = texture2D(tex, uv.zw, 0.0).xyz;
  vec3 rgbNE = texture2D(tex, uv.zw + vec2(1, 0) * rcpFrame.xy, 0.0).xyz;
  vec3 rgbSW = texture2D(tex, uv.zw + vec2(0, 1) * rcpFrame.xy, 0.0).xyz;
  vec3 rgbSE = texture2D(tex, uv.zw + vec2(1, 1) * rcpFrame.xy, 0.0).xyz;
  vec3 rgbM = texture2D(tex, uv.xy, 0.0).xyz;

  vec3 luma = vec3(0.299, 0.587, 0.114);
  float lumaNW = dot(rgbNW, luma);
  float lumaNE = dot(rgbNE, luma);
  float lumaSW = dot(rgbSW, luma);
  float lumaSE = dot(rgbSE, luma);
  float lumaM = dot(rgbM, luma);

  float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
  float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

  vec2 dir;
  dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
  dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));

  float dirReduce =
      max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL),
          FXAA_REDUCE_MIN);
  float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);

  dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
            max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX), dir * rcpDirMin)) *
        rcpFrame.xy;

  vec3 rgbA =
      (1.0 / 2.0) * (texture2D(tex, uv.xy + dir * (1.0 / 3.0 - 0.5), 0.0).xyz +
                     texture2D(tex, uv.xy + dir * (2.0 / 3.0 - 0.5), 0.0).xyz);
  vec3 rgbB =
      rgbA * (1.0 / 2.0) +
      (1.0 / 4.0) * (texture2D(tex, uv.xy + dir * (0.0 / 3.0 - 0.5), 0.0).xyz +
                     texture2D(tex, uv.xy + dir * (3.0 / 3.0 - 0.5), 0.0).xyz);

  float lumaB = dot(rgbB, luma);

  if ((lumaB < lumaMin) || (lumaB > lumaMax))
    return rgbA;

  return rgbB;
}

// AgX Settings
const float MIDDLE_GREY = 0.18f;
const float SLOPE = 2.3f;
const float TOE_POWER = 1.9f;
const float SHOULDER_POWER = 3.1f;
const float COMPRESSION = 0.15;

//"Look" Settings
// Try 1.2 for a more saturated look. There's nothing wrong with intentionally
// skewing to develop a look you like, because intention is the entire point.
// That's why we should separate grading from compression, rather than combining
// them and forcing an artist
const float SATURATION = 1.;

// Demo Settings
const float EXPOSURE = -1.0;
const float MIN_EV = -10.0f;
const float MAX_EV = 6.5f;
const float AGX_LERP = 1.0;

mat3 InverseMat(mat3 m) {
  float d = m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) -
            m[0].y * (m[1].x * m[2].z - m[1].z * m[2].x) +
            m[0].z * (m[1].x * m[2].y - m[1].y * m[2].x);

  float id = 1.0f / d;

  mat3 c = mat3(1, 0, 0, 0, 1, 0, 0, 0, 1);

  c[0].x = id * (m[1].y * m[2].z - m[2].y * m[1].z);
  c[0].y = id * (m[0].z * m[2].y - m[0].y * m[2].z);
  c[0].z = id * (m[0].y * m[1].z - m[0].z * m[1].y);
  c[1].x = id * (m[1].z * m[2].x - m[1].x * m[2].z);
  c[1].y = id * (m[0].x * m[2].z - m[0].z * m[2].x);
  c[1].z = id * (m[1].x * m[0].z - m[0].x * m[1].z);
  c[2].x = id * (m[1].x * m[2].y - m[2].x * m[1].y);
  c[2].y = id * (m[2].x * m[0].y - m[0].x * m[2].y);
  c[2].z = id * (m[0].x * m[1].y - m[1].x * m[0].y);

  return c;
}

vec3 xyYToXYZ(vec3 xyY) {
  if (xyY.y == 0.0f) {
    return vec3(0, 0, 0);
  }

  float Y = xyY.z;
  float X = (xyY.x * Y) / xyY.y;
  float Z = ((1.0f - xyY.x - xyY.y) * Y) / xyY.y;

  return vec3(X, Y, Z);
}

vec3 Unproject(vec2 xy) { return xyYToXYZ(vec3(xy.x, xy.y, 1)); }

mat3 PrimariesToMatrix(vec2 xy_red, vec2 xy_green, vec2 xy_blue,
                       vec2 xy_white) {
  vec3 XYZ_red = Unproject(xy_red);
  vec3 XYZ_green = Unproject(xy_green);
  vec3 XYZ_blue = Unproject(xy_blue);

  vec3 XYZ_white = Unproject(xy_white);

  mat3 temp = mat3(XYZ_red.x, XYZ_green.x, XYZ_blue.x, 1.0f, 1.0f, 1.0f,
                   XYZ_red.z, XYZ_green.z, XYZ_blue.z);

  mat3 inverse = InverseMat(temp);
  vec3 scale = XYZ_white * inverse;

  return mat3(scale.x * XYZ_red.x, scale.y * XYZ_green.x, scale.z * XYZ_blue.x,
              scale.x * XYZ_red.y, scale.y * XYZ_green.y, scale.z * XYZ_blue.y,
              scale.x * XYZ_red.z, scale.y * XYZ_green.z, scale.z * XYZ_blue.z);
}

mat3 ComputeCompressionMatrix(vec2 xyR, vec2 xyG, vec2 xyB, vec2 xyW,
                              float compression) {
  float scale_factor = 1.0f / (1.0f - compression);
  vec2 R = ((xyR - xyW) * scale_factor) + xyW;
  vec2 G = ((xyG - xyW) * scale_factor) + xyW;
  vec2 B = ((xyB - xyW) * scale_factor) + xyW;
  vec2 W = xyW;

  return PrimariesToMatrix(R, G, B, W);
}

vec3 OpenDomainToNormalizedLog2(vec3 openDomain, float minimum_ev,
                                float maximum_ev) {
  float total_exposure = maximum_ev - minimum_ev;

  vec3 output_log =
      clamp(log2(openDomain / MIDDLE_GREY), minimum_ev, maximum_ev);

  return (output_log - minimum_ev) / total_exposure;
}

float AgXScale(float x_pivot, float y_pivot, float slope_pivot, float power) {
  return pow(pow((slope_pivot * x_pivot), -power) *
                 (pow((slope_pivot * (x_pivot / y_pivot)), power) - 1.0),
             -1.0 / power);
}

float AgXHyperbolic(float x, float power) {
  return x / pow(1.0 + pow(x, power), 1.0f / power);
}

float AgXTerm(float x, float x_pivot, float slope_pivot, float scale) {
  return (slope_pivot * (x - x_pivot)) / scale;
}

float AgXCurve(float x, float x_pivot, float y_pivot, float slope_pivot,
               float toe_power, float shoulder_power, float scale) {
  if (scale < 0.0f) {
    return scale * AgXHyperbolic(AgXTerm(x, x_pivot, slope_pivot, scale),
                                 toe_power) +
           y_pivot;
  } else {
    return scale * AgXHyperbolic(AgXTerm(x, x_pivot, slope_pivot, scale),
                                 shoulder_power) +
           y_pivot;
  }
}

float AgXFullCurve(float x, float x_pivot, float y_pivot, float slope_pivot,
                   float toe_power, float shoulder_power) {
  float scale_x_pivot = x >= x_pivot ? 1.0f - x_pivot : x_pivot;
  float scale_y_pivot = x >= x_pivot ? 1.0f - y_pivot : y_pivot;

  float toe_scale =
      AgXScale(scale_x_pivot, scale_y_pivot, slope_pivot, toe_power);
  float shoulder_scale =
      AgXScale(scale_x_pivot, scale_y_pivot, slope_pivot, shoulder_power);

  float scale = x >= x_pivot ? shoulder_scale : -toe_scale;

  return AgXCurve(x, x_pivot, y_pivot, slope_pivot, toe_power, shoulder_power,
                  scale);
}

#define AGX_LOOK 2

// AgX
// ->

// Mean error^2: 3.6705141e-06
vec3 agxDefaultContrastApprox(vec3 x) {
  vec3 x2 = x * x;
  vec3 x4 = x2 * x2;

  return +15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x +
         0.4298 * x2 + 0.1191 * x - 0.00232;
}

vec3 agx(vec3 val) {
  const mat3 agx_mat =
      mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
           0.0784335999999992, 0.878468636469772, 0.0784336, 0.0792237451477643,
           0.0791661274605434, 0.879142973793104);

  const float min_ev = -12.47393f;
  const float max_ev = 4.026069f;

  // Input transform
  val = agx_mat * val;

  // Log2 space encoding
  val = clamp(log2(val), min_ev, max_ev);
  val = (val - min_ev) / (max_ev - min_ev);

  // Apply sigmoid function approximation
  val = agxDefaultContrastApprox(val);

  return val;
}

vec3 agxEotf(vec3 val) {
  const mat3 agx_mat_inv =
      mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
           -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
           -0.0990297440797205, -0.0989611768448433, 1.15107367264116);

  // Undo input transform
  val = agx_mat_inv * val;

  // sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
  // val = pow(val, vec3(2.2));

  return val;
}

vec3 agxLook(vec3 val) {
  const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
  float luma = dot(val, lw);

  // Default
  vec3 offset = vec3(0.0);
  vec3 slope = vec3(1.0);
  vec3 power = vec3(1.0);
  float sat = 1.0;

#if AGX_LOOK == 1
  // Golden
  slope = vec3(1.0, 0.9, 0.5);
  power = vec3(0.8);
  sat = 0.8;
#elif AGX_LOOK == 2
  // Punchy
  slope = vec3(1.0);
  power = vec3(1.35, 1.35, 1.35);
  sat = 1.4;
#endif

  // ASC CDL
  val = pow(val * slope + offset, power);
  return luma + sat * (val - luma);
}

// <-

vec4 do_agx(vec4 fragColor) {
  vec3 col = fragColor.xyz;

  // AgX
  // ->
  col = agx(col);
  col = agxLook(col);
  col = agxEotf(col);
  // <-

  fragColor = vec4(col, 1.0);
  return fragColor;
}

#define RESOLUTION 0.25

const float phi2 = 1.32471795724474602596;
const vec2 a = vec2(1.0 / phi2, 1.0 / (phi2 * phi2));

vec2 R2(float n) { return fract(a * n + 0.5); }
vec2 jitter() {
  return ((R2(float(int(time) % 1000)) * 2. - 1.) / (wh * 2. * RESOLUTION));
}

vec3 gamma_correct(vec3 linear) {
  bvec3 cutoff = lessThan(linear, vec3(0.0031308));
  vec3 higher = 1.055 * pow(linear, vec3(1.0 / 2.4)) - 0.055;
  vec3 lower = linear * 12.92;
  return mix(higher, lower, cutoff);
}

// eotf_pq parameters
const float Lp = 10000.0;
const float m1 = 2610.0 / 16384.0;
const float m2 = 1.7 * 2523.0 / 32.0;
const float c1 = 107.0 / 128.0;
const float c2 = 2413.0 / 128.0;
const float c3 = 2392.0 / 128.0;

vec3 eotf_pq(vec3 x) {
  x = sign(x) * pow(abs(x), vec3(1.0 / m2));
  x = sign(x) * pow((abs(x) - c1) / (c2 - c3 * abs(x)), vec3(1.0 / m1)) * Lp;
  return x;
}

vec3 eotf_pq_inverse(vec3 x) {
  x /= Lp;
  x = sign(x) * pow(abs(x), vec3(m1));
  x = sign(x) * pow((c1 + c2 * abs(x)) / (1.0 + c3 * abs(x)), vec3(m2));
  return x;
}

// XYZ <-> ICh parameters
const float W = 140.0;
const float b = 1.15;
const float g = 0.66;

vec3 XYZ_to_ICh(vec3 XYZ) {
  XYZ *= W;
  XYZ.xy = vec2(b, g) * XYZ.xy - (vec2(b, g) - 1.0) * XYZ.zx;

  const mat3 XYZ_to_LMS =
      transpose(mat3(0.41479, 0.579999, 0.014648, -0.20151, 1.12065, 0.0531008,
                     -0.0166008, 0.2648, 0.66848));

  vec3 LMS = XYZ_to_LMS * XYZ;
  LMS = eotf_pq_inverse(LMS);

  const mat3 LMS_to_Iab = transpose(mat3(0.0, 1.0, 0.0, 3.524, -4.06671,
                                         0.542708, 0.199076, 1.0968, -1.29588));

  vec3 Iab = LMS_to_Iab * LMS;

  float I = eotf_pq(vec3(Iab.x)).x / W;
  float C = length(Iab.yz);
  float h = atan(Iab.z, Iab.y);
  return vec3(I, C, h);
}

vec3 ICh_to_XYZ(vec3 ICh) {
  vec3 Iab;
  Iab.x = eotf_pq_inverse(vec3(ICh.x * W)).x;
  Iab.y = ICh.y * cos(ICh.z);
  Iab.z = ICh.y * sin(ICh.z);

  const mat3 Iab_to_LMS =
      transpose(mat3(1.0, 0.2772, 0.1161, 1.0, 0.0, 0.0, 1.0, 0.0426, -0.7538));

  vec3 LMS = Iab_to_LMS * Iab;
  LMS = eotf_pq(LMS);

  const mat3 LMS_to_XYZ =
      transpose(mat3(1.92423, -1.00479, 0.03765, 0.35032, 0.72648, -0.06538,
                     -0.09098, -0.31273, 1.52277));

  vec3 XYZ = LMS_to_XYZ * LMS;
  XYZ.x = (XYZ.x + (b - 1.0) * XYZ.z) / b;
  XYZ.y = (XYZ.y + (g - 1.0) * XYZ.x) / g;
  return XYZ / W;
}

const mat3 XYZ_to_sRGB =
    transpose(mat3(3.2404542, -1.5371385, -0.4985314, -0.9692660, 1.8760108,
                   0.0415560, 0.0556434, -0.2040259, 1.0572252));

const mat3 sRGB_to_XYZ =
    transpose(mat3(0.4124564, 0.3575761, 0.1804375, 0.2126729, 0.7151522,
                   0.0721750, 0.0193339, 0.1191920, 0.9503041));

bool in_sRGB_gamut(vec3 ICh) {
  vec3 sRGB = XYZ_to_sRGB * ICh_to_XYZ(ICh);
  return all(greaterThanEqual(sRGB, vec3(0.0))) &&
         all(lessThanEqual(sRGB, vec3(1.0)));
}

vec3 tonemap(vec3 sRGB) {
  vec3 ICh = XYZ_to_ICh(sRGB_to_XYZ * sRGB);

  const float s0 = 0.71;
  const float s1 = 1.04;
  const float p = 1.40;
  const float t0 = 0.01;
  float n = s1 * pow(ICh.x / (ICh.x + s0), p);
  ICh.x = clamp(n * n / (n + t0), 0.0, 1.0);

  if (!in_sRGB_gamut(ICh)) {
    float C = ICh.y;
    ICh.y -= 0.5 * C;

    for (float i = 0.25; i >= 1.0 / 256.0; i *= 0.5) {
      ICh.y += (in_sRGB_gamut(ICh) ? i : -i) * C;
    }
  }

  return XYZ_to_sRGB * ICh_to_XYZ(ICh);
}

vec3 sharpn222(vec2 iResolution) {
  float amp = -0.8;

  vec3 c1 = texture2D(TAA, texCoord).xyz * 5.;
  vec3 c2 =
      texture2D(TAA, ((texCoord * iResolution + vec2(1., 0.)) / iResolution))
          .xyz *
      amp;
  vec3 c3 =
      texture2D(TAA, ((texCoord * iResolution + vec2(-1., 0.)) / iResolution))
          .xyz *
      amp;
  vec3 c4 =
      texture2D(TAA, ((texCoord * iResolution + vec2(0., 1.)) / iResolution))
          .xyz *
      amp;
  vec3 c5 =
      texture2D(TAA, ((texCoord * iResolution + vec2(0., -1.)) / iResolution))
          .xyz *
      amp;

  vec3 c = c1 + c2 + c3 + c4 + c5;

  return max(c, 0.);
}

float random3d(vec3 p) {
  return fract(sin(p.x * 214. + p.y * 241. + p.z * 123.) * 100. +
               cos(p.x * 42. + p.y * 41.2 + p.z * 32.) * 10.);
}
vec2 DistortPosition(in vec2 position) {
  float CenterDistance = length(position);
  float DistortionFactor = mix(1.0f, CenterDistance, 0.9f);
  return position / DistortionFactor;
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

vec3 mul3(in mat3 m, in vec3 v) {
  return vec3(dot(v, m[0]), dot(v, m[1]), dot(v, m[2]));
}

vec3 mul3(in vec3 v, in mat3 m) { return mul3(m, v); }

vec3 srgb2oklab(vec3 c) {

  mat3 m1 =
      mat3(0.4122214708, 0.5363325363, 0.0514459929, 0.2119034982, 0.6806995451,
           0.1073969566, 0.0883024619, 0.2817188376, 0.6299787005);

  vec3 lms = mul3(m1, c);

  lms = pow(lms, vec3(1. / 3.));

  mat3 m2 = mat3(+0.2104542553, +0.7936177850, -0.0040720468, +1.9779984951,
                 -2.4285922050, +0.4505937099, +0.0259040371, +0.7827717662,
                 -0.8086757660);

  return mul3(m2, lms);
}

vec3 oklab2srgb(vec3 c) {
  mat3 m1 = mat3(1.0000000000, +0.3963377774, +0.2158037573, 1.0000000000,
                 -0.1055613458, -0.0638541728, 1.0000000000, -0.0894841775,
                 -1.2914855480);

  vec3 lms = mul3(m1, c);

  lms = lms * lms * lms;

  mat3 m2 = mat3(+4.0767416621, -3.3077115913, +0.2309699292, -1.2684380046,
                 +2.6097574011, -0.3413193965, -0.0041960863, -0.7034186147,
                 +1.7076147010);
  return mul3(m2, lms);
}

vec3 lab2lch(in vec3 c) {
  return vec3(c.x, sqrt((c.y * c.y) + (c.z * c.z)), atan(c.z, c.y));
}

vec3 lch2lab(in vec3 c) { return vec3(c.x, c.y * cos(c.z), c.y * sin(c.z)); }

vec3 srgb_to_oklch(in vec3 c) { return lab2lch(srgb2oklab(c)); }
vec3 oklch_to_srgb(in vec3 c) { return oklab2srgb(lch2lab(c)); }

vec3 lodbloom(vec2 iResolution, float lod) {

  vec3 accumBloom = vec3(0.);
  const float atrous_kernel_weights[25] = {
      1.0 / 273.0, 4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0,  1.0 / 273.0,
      4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0,
      7.0 / 273.0, 26.0 / 273.0, 41.0 / 273.0, 26.0 / 273.0, 7.0 / 273.0,
      4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0,
      1.0 / 273.0, 4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0,  1.0 / 273.0};
  for (int i = 0; i < 25; i++) {
    vec2 coords2 = vec2(float(i % 5) - 2., float(i / 5) - 2.) * 2.;
    vec2 newCords = (texCoord * iResolution + coords2) / iResolution;
    // accumBloom += textureLod(TAA, newCords, ceil(log2(max(wh.x,
    // wh.y))*lod)).xyz*atrous_kernel_weights[i];
    accumBloom +=
        clamp(texture2D(TAA, newCords).rgb, 0., 1.) * atrous_kernel_weights[i];
  }
  return accumBloom;
}

void main() {
  //*texture2D(albedo, texCoord).xyz;
  float RW = texture2D(weightS, texCoord).z;
  vec3 col = texture2D(outgoingrS, texCoord).xyz * clamp(RW, 0., 200.);
  // weigth;
  // uniform sampler2D outgoingr;
  // RW = texture2D(weigth, texCoord).z;
  //  col = texture2D(outgoingr, texCoord).xyz*clamp(RW, 0., 200.);

  //  vec3 col = texture2D(Acc, texCoord).xyz;
  // col = upscaleIndirect(wh, texCoord).xyz;
  // col = texture2D(TAA, texCoord).xyz;

  // float con;
  // float sharpness = 0.4;
  // FsrRcasCon(con,sharpness);
  // col = FsrRcasF(texCoord*wh*2., con);

  // vec3 color = sharpn(wh*2.0);
  // col = texture2D(den1, texCoord).xyz;
  vec2 fragCoord = texCoord * wh;
  uint r = uint(uint(fragCoord.x) * uint(1973) +
                uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) |
           uint(1);

  vec4 p22 = vec4((texCoord)*2.0 - 1.0, 0.0, 1.0);
  vec3 dir = (invproj * p22).xyz / (invproj * p22).w;
  dir = normalize(mat3(invview) * dir);
  vec3 wi = dir;

  float depth = texture2D(normal, texCoord).w;
  float depthCenter = texture2D(normal, vec2(0.5)).w;
  // float f = 1.0 - exp(-pow((depth - depthCenter),2.)*0.01);
  //  f = max(1.0/(1.0+exp(-10.*(depthCenter-depth))),0.001);

  vec3 color2 = col;
  // color = texture2D(den1, texCoord).xyz;
  // if(texCoord.x > 0.5){
  // color2 = blur2(texCoord, clamp(f*0., 0., 20.), wh);

  // float f = 1.0 - exp(-pow((depth - depthCenter)*0.2,2.));

  float f = 1.0 - exp(-pow(abs(depth - depthCenter), 4.) * 0.01);
  float C = 2. * ((depth - depthCenter) / depth);
  float f1 = 0.01;
  f = C;

  float As = .4;
  float focalLength = 0.028;
  float cOC = As * (abs(depth - depthCenter) / max(depth, 0.01)) *
              (focalLength / max(depthCenter - focalLength, 0.001));

  float shh = 0.024;
  float percent = cOC / max(shh, 0.001);

  f = percent;
  /*
  CoC = abs(aperture * (focallength * (objectdistance - planeinfocus)) /
  (objectdistance * (planeinfocus - focallength)))
  */

  // vec3 blur3(vec2 p, float dist, float centerDepth, float focalLength, vec2
  // iResolution ) {

  vec4 wp = texture2D(watpos, texCoord);
  float fov = 2. * atan(1. / projection[1][1]);
  // 2 * atan(1 / projMatrix[1][1])
  float fc = .1;
  // Color = kaw(Color);
  bool mixen = false;
  if (fov < 3.14159 * 0.3) {
    fc = 4.;
    mixen = true;
  }
  color2 = blur3(texCoord, depthCenter, fc, wh, r, mixen);

  // vec3 avgBloom = textureLod(fogLO, texCoord, ceil(log2(max(wh.x,
  // wh.y))*0.65)).xyz; color2 = mix(color2,avgBloom,0.09); float mm
  // = 1.-exp(-0.0001*depth); color2 = mix(color2, vec3(10.), mm); vec3
  // upscaleIndirect(vec2 iResolution, vec2 texcc){

  // color2 =  upscaleIndirect(wh, texCoord);/
  // color2 = texture2D(fogLO, texCoord).xyz;

  // vec3 depthOfField(vec2 texCoord, float focusPoint, float focusScale, vec2
  // uPixelSize) color2 = depthOfField(texCoord, depthCenter, 8., 1./wh);

  // vec3 lodbloom(vec2 iResolution, float lod){
  vec3 blm = lodbloom(wh, 2.);
  // color2 = mix(color2, pow(blm, vec3(1.2))*2., 0.2);
  color2 += clamp(pow(blm, vec3(1.5)) * 0.6, 0., 10.);
  // color2 = ;
  // color2 = sharpn222(wh);
  //\color2 = texture2D(den1, texCoord).xyz;

  // vec3 PP = texture2D(position, texCoord).xyz;
  // vec3 cam = viewPos;
  // if(texture2D(albedo, texCoord).w > 0.5){
  //     PP = cam + wi * 100.;
  // }

  // vec3 div = (PP-cam)/50.;

  // vec3 volCol = vec3(0.), volAbs = vec3(1.);
  // vec3 stepAbs = vec3(1.)*exp(-0.05*length(div));
  //             //vec3 d = normalize(direction);

  // vec3 stepCol = (vec3(1.) - stepAbs) ;
  // vec3 stepCol2 = (vec3(1.) - stepAbs) ;
  // vec3 accum = vec3(0.);
  // for(int i = 0; i < 50; i++){
  //     cam += div*max(random3d(cam),0.5);
  //     vec3 sampPos = (cam+90.0 - floor(viewPos));
  //     //accum += imageLoad(rcpos, ivec3(sampPos)).xyz;
  //     float dens = (imageLoad(rcpos, ivec3(sampPos)).w);

  //     accum += volAbs*(imageLoad(rcpos, ivec3(sampPos)).xyz)*0.3*exp(
  //     -4.73*max(1.-dens/40.,0.001)) + length(cam - viewPos)*0.0000001; volAbs
  //     *= exp(-length(cam - viewPos)*0.1*max(dens,0.1)*length(cam -
  //     viewPos)*.001);
  //     //rcpos
  // }
  // color2 = color2*volAbs.x + accum;

  col = color2;
  // col = texture2D(colortexFog, texCoord).xyz/max(texture2D(colortexFog,
  // texCoord).w,0.1);
  vec2 iResolution = wh;
  vec3 N = texture2D(normal, texCoord).xyz;
  // col = imageLoad(rcpos, ivec3((texture2D(position, texCoord).xyz + 90. -
  // floor(viewPos)))).xyz;

  // col = texture2D(normal, texCoord).xyz;

  // float expos = texture2D(fogLO, texCoord).w;//fogLO
  float avgLuma =
      pow(textureLod(fogLO, texCoord, ceil(log2(max(wh.x, wh.y)))).a, 2.);
  float EV100 = log2(avgLuma * 100. / 65.);
  float exposure = clamp((0.5) * exp2(EV100), 0.00018, 1.);
  col = max(col, 0.);
  // col = col*col*1.8;
  col /= exposure;
  // col *= 10.;
  col *= brightness;
  //  col = pow(col, vec3(1.2, 2.0, 1.9));
  //   col += texture2D(colorfog, texCoord).xyz*0.8;
  //    col *= col/max(lum(col),0.01);
  // col = tonemap_uchimura2(col);

  uint seedCam = uint(time);
  vec2 smallOffset =
      ((vec2(rndf(seedCam), rndf(seedCam))) * 2. - 1.) / iResolution;

  if (texture2D(albedo, texCoord - smallOffset).w < 0.5) {
    col = srgb_to_oklch(col);
    col.z *= 1.2;
    col.y *= 1.2;
    col.x *= 0.7;
    col = oklch_to_srgb(col);
    //   col = pow(col, vec3(1.2, 1.5, 1.8));
    col = tonemap(col);

  } else {
    col = srgb_to_oklch(col);
    col.x *= 0.6;
    col = oklch_to_srgb(col);
    col = do_agx(vec4(col, 1.)).xyz;
  }
  //  col = tonemap(col);

  // col = do_agx(vec4(col, 1.)).xyz;

  // col = rgbtohsv(col);
  // col.g *= 1.5;
  // col.r *= 1.2;
  // col = hsvtorgb(vec3(clamp(col.x,0.,360.),clamp(col.y, 0., 1.),clamp(col.z,
  // 0., 1.))); col = vec3(avgLuma);

  // col = pow(col, vec3(1./2.2));
  // col = sRGB(col);
  // float sn = texture2D(sunTex, texCoord).b;

  // col = vec3(sn)*0.05;

  // col = shadowS(texture2D(position, texCoord).rgb + texture2D(normal,
  // texCoord).xyz*0.1);
  outputColor = vec4(gamma_correct(col), 1.);
}