#version 450 compatibility
#extension GL_ARB_shading_language_packing : enable

layout(location = 0) out vec4 den2;
layout(location = 1) out vec4 var2;
layout(location = 2) out vec4 colorfog2;
layout(location = 3) out vec4 secalb;

in vec2 texCoord;

/*
_denoiseShader.Use();
            _denoiseShader.SetInt("color", 0);
            _denoiseShader.SetInt("position", 1);
            _denoiseShader.SetInt("normal", 2);
            _denoiseShader.SetInt("albedo", 3);
            _denoiseShader.SetInt("secondpos", 4);
            _denoiseShader.SetInt("weigth", 5);
            _denoiseShader.SetInt("outgoingr", 6);
            _denoiseShader.SetInt("weightS", 7);
            _denoiseShader.SetInt("outgoingrS", 8);
            _denoiseShader.SetInt("prevN", 9);
            _denoiseShader.SetInt("ACC", 10);
            _denoiseShader.SetInt("den1", 11);

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
uniform sampler2D ACC;
uniform sampler2D den1;
uniform sampler2D var1;
uniform sampler2D reflN;
uniform sampler2D reflectionAlb;
uniform sampler2D colorfog;
uniform sampler2D holdinfo;
uniform sampler2D watpos;
uniform sampler2D watnorm;

/*
_TemporalRestirShader.SetInt("prevW", 5);
            _TemporalRestirShader.SetInt("prevL", 6);
            _TemporalRestirShader.SetInt("prevP", 7);
             */

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

float lum(vec3 c) {
  return sqrt(0.299 * c.x * c.x + 0.587 * c.y * c.y + 0.114 * c.z * c.z);
}

float g3x3(vec2 coords, vec2 iResolution) {
  const float atrous_kernel_weights[25] = {
      1.0 / 273.0, 4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0,  1.0 / 273.0,
      4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0,
      7.0 / 273.0, 26.0 / 273.0, 41.0 / 273.0, 26.0 / 273.0, 7.0 / 273.0,
      4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0,
      1.0 / 273.0, 4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0,  1.0 / 273.0};
  // return texture2D(colortex15, coords*1.0).x;
  vec2 p = coords * iResolution;
  float ppp2 = 0.;
  float ppp = 0.;
  for (int i = 0; i < 25; i++) {
    vec2 coords2 = vec2(float(i % 5) - 2., float(i / 5) - 2.) * 1.;
    // ppp += (texelFetch(colortex6, ivec2((p + coords2)*0.5 ), 0).x) ;
    ppp += texture2D(var1, ((p + coords2) / iResolution)).x *
           atrous_kernel_weights[i];
  }
  return ppp;
}

void main() {
  vec2 fragCoord = texCoord * wh;
  uint r = uint(uint(fragCoord.x) * uint(1973) +
                uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) |
           uint(1);
  if (texture2D(albedo, texCoord).w > 0.5) {
    return;
  }
  vec3 InitSampleLO = texture2D(color, texCoord).xyz;
  vec4 pst = texture2D(secondpos, texCoord).xyzw;

  vec3 cameraOffset = viewPos - lastViewPos;
  vec3 View = texture2D(position, texCoord).xyz;
  vec4 Projected = vec4(View.xyz, 1.); // + vec4(cameraOffset, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;

  vec2 iResolution = wh;

  /*const float atrous_kernel_weights[25] = {
                  1.0 / 256.0, 4.0 / 256.0,  6.0 / 256.0,  4.0 / 256.0,  1.0 /
  256.0, 4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0, 6.0
  / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0, 4.0 /
  256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0, 4.0
  / 256.0,  6.0 / 256.0,  4.0 / 256.0,  1.0 / 256.0};

  const float atrous_kernel_weights[49] = {
          0.0,0.,1.,2.,1.,0.,0.,
          0.,3.,13.,22.,13.,3.,0.,
          1.,13.,59.,97.,59.,13.,1.,
          2.,22.,97.,159.,97.,22.,2.,
          1.,13.,59.,97.,59.,13.,1.,
          0.,3.,13.,22.,13.,3.,0.,
          0.0,0.,1.,2.,1.,0.,0.
  };	*/
  /*
const float atrous_kernel_weights[25] = {
1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0,
4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0,
4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0 };
*/
  float atrous_kernel_weights[9] = {

      16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0,
      24.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0};

  float mult = pow(2., texture2D(den1, texCoord).w);
  vec3 col = vec3(0.);
  float w = 0.;
  vec2 currCoors = texCoord * iResolution;
  float hist = texture2D(ACC, texCoord).w;
  hist = max(1.0 - hist / 22., 0.);
  hist = 0.;
  float moment1 = 0.;
  float moment2 = 0.;
  float momentCount = 0.;
  vec3 currrp = texture2D(den1, texCoord).xyz;
  vec4 ND = texture2D(normal, texCoord);

  float var = 0.;
  float w2 = 0.;
  float curvar = texture2D(var1, texCoord).x;
  // float roughness = texture2D(color, texCoord).w;
  float roughness = texture2D(holdinfo, texCoord).w;
  vec3 curN = texture2D(reflN, texCoord).xyz;
  float depth1 = texture2D(normal, texCoord).w;

  for (int i = 0; i < 9; i++) {
    vec2 coords = vec2(float(i % 3) - 1., float(i / 3) - 1.) * mult;

    vec2 fincords = (currCoors + coords) / iResolution;

    vec4 ND2 = texture2D(normal, fincords);
    float wd = exp(-(abs(ND.w - ND2.w) / 122.945));
    float wp = max(pow(max(dot(ND.xyz, ND2.xyz), 0.), 6.), 0.0);
    float nextrp = texture2D(var1, fincords).x;
    float wrp = exp(-(abs(((curvar)) - (nextrp)) / (1.2 + 0.001)));
    // wrp = max(wrp, 1.0-clamp(hist*0.01, 0., 1.));

    float weigth = 1.0;

    float depth2 = texture2D(normal, fincords).w;

    float depthWeight = pow(exp(-abs((depth1) - (depth2)) * 0.1), 12.);

    weigth = wp * depthWeight;

    var += pow(atrous_kernel_weights[i], 2.) * (weigth * weigth) * nextrp;
    w2 += pow((atrous_kernel_weights[i]) * weigth, 2.);
  }
  vec4 currVAR =
      vec4((var / max(pow(w2, 1.), 0.001)), texture2D(var1, texCoord).yzw);
  var2 = currVAR;

  float DP1 = texture2D(reflectionAlb, texCoord).w;

  vec3 dn1 = texture2D(colorfog, texCoord).xyz;
  vec3 reflct = vec3(0.);
  float reflctdiv = (0.);

  vec3 shad1 = vec3(0.);
  float shad1we = 0.;

  vec3 shd1 = texture2D(reflectionAlb, texCoord).xyz;
  float rflLength = texture2D(reflN, texCoord).w;

  float normLength = texture2D(color, texCoord).w;

  for (int i = 0; i < 9; i++) {
    vec2 coords = vec2(float(i % 3) - 1., float(i / 3) - 1.) * mult;

    vec2 fincords = (currCoors + coords) / iResolution;
    if (texture2D(albedo, fincords).w > 0.5) {
      continue;
    }
    vec4 ND2 = texture2D(normal, fincords);
    // float wd = exp(-(abs(ND.w - ND2.w) / 122.945));
    float wd = pow(exp(-abs((ND.w) - (ND2.w)) * 0.01), 1.);
    float wp = max(pow(max(dot(ND.xyz, ND2.xyz), 0.), 26.), 0.0);
    vec3 nextrp = texture2D(den1, fincords).xyz;
    vec3 nextN = texture2D(reflN, fincords).xyz;

    // float g3x3(vec2 coords, vec2 iResolution) {

    float dll = max(sqrt(g3x3(texCoord, iResolution)), 0.);
    dll *= pow(clamp(normLength / 320., 0.0001, 1.), 2.);
    float wrp =
        exp(-(abs((lum(currrp)) - lum(nextrp)) / (1.71 * dll + 0.000001)));

    // wrp = max(wrp, 1.0-clamp(hist*0.5, 0., 1.));
    // wrp = max(wrp, hist);
    float weigth = 1.0;
    float depth2 = texture2D(normal, fincords).w;

    float depthWeight = pow(exp(-abs((depth1) - (depth2)) * 0.1), 12.);

    weigth = wp * wrp * depthWeight;

    // vec3 shd2 = texture2D(reflectionAlb, fincords).xyz;

    // float wrp2 = exp(-(abs((lum(shd1)) - lum(shd2)) / (1.72 + 0.00001)));

    // float wp2 = max(pow(max(dot(ND.xyz, ND2.xyz), 0.), 126.), 0.0);
    // wrp2 = max(wrp2, clamp(rflLength*0.01, 0., 0.75));
    //  wp2 = max(wp2, clamp(rflLength*0.01, 0., 0.25));

    //  float wt2 = wp2 * wrp2*depthWeight;
    // weigth = max(weigth, hist);
    // shad1 += (atrous_kernel_weights[i]) * wt2 * shd2;
    // shad1we += (atrous_kernel_weights[i]) * wt2;

    col += (atrous_kernel_weights[i]) * weigth * nextrp;
    w += ((atrous_kernel_weights[i]) * weigth);
  }
  vec4 wpz = texture2D(watpos, texCoord);
  vec4 wn = texture2D(watnorm, texCoord);
  if (wpz.w > 0.5) {
    ND = wn;
    roughness = 0.001;
  }
  for (int i = 0; i < 9; i++) {
    vec2 coords =
        vec2(float(i % 3) - 1., float(i / 3) - 1.) * mult * pow(roughness, 0.5);

    vec2 fincords = (currCoors + coords) / iResolution;
    if (texture2D(albedo, fincords).w > 0.5) {
      continue;
    }

    vec4 ND2 = texture2D(normal, fincords);
    if (wpz.w > 0.5) {
      ND2 = texture2D(watnorm, fincords);
    }
    float wd = exp(-(abs(ND.w - ND2.w) / 122.945));
    float wp = max(pow(max(dot(ND.xyz, ND2.xyz), 0.), 26.), 0.0);
    vec3 nextrp = texture2D(den1, fincords).xyz;
    vec3 nextN = texture2D(reflN, fincords).xyz;

    vec3 dn2 = texture2D(colorfog, fincords).xyz;

    float dll = max(sqrt(currVAR.x), 0.);

    float wrp2 = exp(-(abs((lum(dn1)) - lum(dn2)) / (12.072 * dll + 0.00001)));

    // wrp = max(wrp, 1.0-clamp(hist*0.5, 0., 1.));
    wrp2 = max(wrp2, hist);

    float DP2 = texture2D(reflectionAlb, fincords).w;
    float dotWeigth = max(exp(-abs(DP1 - DP2) / 0.0001), sqrt(roughness));
    // weigth = max(weigth, hist);
    float weigth2 = (wp * dotWeigth);
    if (roughness < 0.5 && texture2D(secondpos, texCoord).w < 0.5) {

      weigth2 *= max(pow(max(dot(normalize(curN), normalize(nextN)), 0.0), 6.),
                     sqrt(sqrt(roughness)));
      // nextrp *= texture2D(reflectionAlb, fincords).xyz;
    }

    // weigth2 *= dotWeigth;

    reflct += (atrous_kernel_weights[i]) * weigth2 * dn2;
    reflctdiv += (atrous_kernel_weights[i]) * weigth2;
  }

  vec3 ret = col / max(w, 0.001);
  // //ret = currrp;

  float mean = moment1 / momentCount;
  float variance = moment2 / momentCount - pow(mean, 2.0);

  if (lum(currrp) > mean + 4.5 * max(sqrt(variance), 0.001)) {
    // ret = mean / lum(currrp) * ret;
  }
  // ret = texture2D(reflN, texCoord).xyz;
  // ret = currVAR.xxx;
  colorfog2 =
      vec4(reflct / max(reflctdiv, 0.0001), texture2D(colorfog, texCoord).w);
  den2 = vec4(ret, max(texture2D(den1, texCoord).w - 1., 1.));
  // secalb = vec4(shad1/max(shad1we,0.001), texture2D(reflectionAlb,
  // texCoord).w);
}