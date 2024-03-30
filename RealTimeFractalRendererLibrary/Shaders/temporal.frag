#version 430

layout(location = 0) out vec4 weigths;
layout(location = 1) out vec4 outrad;
layout(location = 2) out vec4 outpos;
layout(location = 3) out vec4 weigthsFog;
layout(location = 4) out vec4 outradFog;
layout(location = 5) out vec4 temporalPositionFog;

// temporalPositionFog
in vec2 texCoord;

uniform sampler2D color;
uniform sampler2D position;
uniform sampler2D normal;
//uniform sampler2D albedo;
uniform sampler2D secondpos;
uniform sampler2D prevW;
uniform sampler2D prevL;
uniform sampler2D prevP;
uniform sampler2D prevN;

//TODO: use or remove these:
//uniform sampler2D reflAlb;
//uniform sampler2D prevPosition;
//uniform sampler2D prevSecondPosition;
//uniform sampler2D tempFog;
//uniform sampler2D LOFog;
//uniform sampler2D fog;
//uniform sampler2D fogsecpos;
//uniform sampler2D tempfogpos;

/*
_TemporalRestirShader.SetInt("tempFog", 12);
            _TemporalRestirShader.SetInt("LOFog", 13);
*/
/*
_TemporalRestirShader.SetInt("prevW", 5);
            _TemporalRestirShader.SetInt("prevL", 6);
            _TemporalRestirShader.SetInt("prevP", 7);
             */

//uniform mat4 view;
//uniform mat4 projection;
//uniform mat4 invview;
//uniform mat4 invproj;
uniform mat4 prevview;
uniform mat4 prevproj;
uniform vec2 wh;
uniform float time;
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

float lum(vec3 c) {
  return sqrt(0.299 * c.x * c.x + 0.587 * c.y * c.y + 0.114 * c.z * c.z);
}

void main() {
  vec2 fragCoord = texCoord * wh;
  uint r = uint(uint(fragCoord.x) * uint(1973) +
                uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) |
           uint(1);

  vec3 InitSampleLO = texture2D(color, texCoord).xyz;
  vec4 pst = texture2D(secondpos, texCoord).xyzw;

  //vec3 cameraOffset = viewPos - lastViewPos;
  vec3 View = texture2D(position, texCoord).xyz;
  vec4 Projected = vec4(View.xyz, 1.); // + vec4(cameraOffset, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;

  float roughness = texture2D(color, texCoord).w;
  bool isthesky = false;

  bool outsc = false;
  vec4 currN = texture2D(normal, texCoord);
  vec4 prevN = texture2D(prevN, ProjectedCoordinates);
  bool disc = false;
  if (dot(normalize(currN.xyz), normalize(prevN.xyz)) < 0.) {
    disc = true;
  }
  if (abs(currN.w - prevN.w) > 2.) {
    disc = true;
  }
  //float isSpec = texture2D(reflAlb, texCoord).w;
  if (ProjectedCoordinates.x > 1. || ProjectedCoordinates.x < 0. ||
      ProjectedCoordinates.y > 1. || ProjectedCoordinates.y < 0. || disc) {
    ProjectedCoordinates = texCoord;
    outsc = true;
  }

 
  vec3 weigthR = texture2D(prevW, ProjectedCoordinates).xyz;
  float RM = weigthR.x;
  float Rw = weigthR.y;
  float RW = weigthR.z;
  vec3 LO = texture2D(prevL, ProjectedCoordinates).xyz;

  if (outsc) {
    RM = 0.;
    RW = 0.;
    Rw = 0.;
    LO = InitSampleLO;
  }

  //  vec3 test = LO*0.9 + InitSampleLO*0.1;
  vec4 _pst = texture2D(prevP, ProjectedCoordinates);
  vec3 pporigin = texture2D(position, texCoord).xyz;
  float cosT = max(dot(normalize(_pst.xyz - pporigin), currN.xyz), 0.);

  float w = lum(InitSampleLO);
  vec3 prevCol = LO;
  /*if(isthesky){
      RM = 0.;
      Rw = 0.;
      RW = 0.;
     // LO = InitSampleLO;
  }*/

  Rw /= max(RM, 0.001);
  RM = min(RM, 20.);
  Rw *= max(RM, 0.001);

  Rw = Rw + w;
  RM = RM + 1.0;
  /// bool changed = false;
  if (rndf(r) < w / max(Rw, 0.001)) {
    LO = InitSampleLO;
    // isK = globalIllum.w;
    _pst = pst;
    //_nst = nst;
    // changed = true;
  }
  // R.Update(S, w, rndf(r));

  if (RM * lum(LO) < 0.001) {
    RW = 0.;
  } else {
    RW = Rw / max(RM * lum(LO), 0.001);
  }

  weigths = vec4(RM, Rw, RW, 1.0);
  outrad = vec4(clamp(LO, 0., 1000.), 1.0);
  outpos = _pst;

  // weigths;
  // outrad;
  // outpos;
}