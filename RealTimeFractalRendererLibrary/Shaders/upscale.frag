#version 450 compatibility
// colortexFogPrev

layout(location = 0) out vec4 upp;

// prevSecondPosition
in vec2 texCoord;

uniform sampler2D TAA;
uniform sampler2D upbefore;
uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D prevN;
uniform sampler2D albedo;

/*
   _upscale.SetInt("normal", 3);
            _upscale.SetInt("prevN", 4);
*/
/*
_TAAShader.SetInt("spatfog", 19);
            _TAAShader.SetInt("spatfogLO", 20);

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

float halton(int i, float b) {
  float f = 1.;
  float r = 0.;

  while (float(i) > 0.) {
    f = f / max(b, 0.0000001);
    r = r + f * float(i % int(b));
    i = i / int(b);
  }
  return r;
}

// from wrigher
vec2 halton_2_3(int idx) {
  float u = halton(idx + 1, 2.0) - 0.5;
  float v = halton(idx + 1, 3.0) - 0.5;
  return vec2(u, v);
}
const float phi2 = 1.32471795724474602596090885447809734; // root of X^3-X-1=0.
const float phi2sq = phi2 * phi2;

#define dot2(a) dot(a, a)
#define sinc(x) (sin(x) / x)

float lacnzos(float x, float s) {
  float xpi = acos(-1.) * x;
  float xpis = s * xpi;
  return x == 0. ? 1. : sinc(xpi) * sinc(xpis);
}
float RENDERSCALE = 0.5;

#define EXPOSURE 15.0

#define BLOOM_FACTOR 0.6

#define ROTATION

#define DOWNSAMPLE_BLUR_RADIUS 5

#define UPSAMPLE_BLUR
#define OPTIMIZED_UPSAMPLE_BLUR

#define MAX_LOD 6

#define STEPS 512
#define MAX_DIST 100.
#define EPS 1e-4

#define PI (acos(-1.))
#define TAU (PI * 2.)
float gaussian(vec2 i, float sigma) {
  return exp(-dot(i, i) / (2.0 * sigma * sigma));
}

vec3 lodbloom(vec2 iResolution, float lod) {

  vec3 accumBloom = vec3(0.);
  const float atrous_kernel_weights[25] = {
      1.0 / 273.0, 4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0,  1.0 / 273.0,
      4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0,
      7.0 / 273.0, 26.0 / 273.0, 41.0 / 273.0, 26.0 / 273.0, 7.0 / 273.0,
      4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0,
      1.0 / 273.0, 4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0,  1.0 / 273.0};
  for (int i = 0; i < 25; i++) {
    vec2 coords2 = vec2(float(i % 5) - 2., float(i / 5) - 2.) * lod * 5.;
    vec2 newCords = (texCoord * iResolution + coords2) / iResolution;
    accumBloom +=
        clamp(textureLod(TAA, newCords, ceil(log2(max(wh.x, wh.y)) * lod)).xyz,
              0., 1.) *
        atrous_kernel_weights[i];
  }
  return accumBloom;
}

void main() {
  vec2 iResolution = wh;

  // vec3 col = texture2D(TAA, texCoord).xyz;

  // upbefore
  // vec2 currJitter = jitter();

  vec3 View = texture2D(position, texCoord).xyz;
  vec4 Projected = vec4(View.xyz, 1.); // + vec4(cameraOffset, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;

  // vec3 lodbloom(vec2 iResolution, float lod){

  vec3 bloom = lodbloom(iResolution, 0.1);
  bloom += lodbloom(iResolution, 0.2);
  bloom += lodbloom(iResolution, 0.3);
  bloom += lodbloom(iResolution, 0.4);
  bloom += lodbloom(iResolution, 0.5);
  bloom += lodbloom(iResolution, 0.6);

  // upp = vec4(clamp(texture(TAA, texCoord).xyz,0.,1000.), 1.);
  upp = vec4(bloom / 6., 0.);
}