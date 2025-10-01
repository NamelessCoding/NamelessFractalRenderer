
layout(binding = 4, Rgba16f) uniform image2D sun;

// layout (location = 0) out vec4 sun;

// out vec4 outputColor;
in vec2 texCoord;

uniform sampler2D sunprev;

uniform vec3 ldir;
uniform vec3 lpos;

uniform mat4 lightproj;
uniform mat4 lightview;

uniform float scale;

#define BAYER_LIMIT 16
#define BAYER_LIMIT_H 4

const int bayerFilter[BAYER_LIMIT] =
    int[](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);

vec2 DistortPosition(in vec2 position) {
  float CenterDistance = length(position);
  float DistortionFactor = mix(1.0f, CenterDistance, 0.9f);
  return position / DistortionFactor;
}

float remap(float x, float a, float b, float c, float d) {
  return (((x - a) / (b - a)) * (d - c)) + c;
}
vec3 remap(vec3 v, vec3 l0, vec3 h0, vec3 ln, vec3 hn) {
  return ln + ((v - l0) * (hn - ln)) / (h0 - l0);
}
float random3d(vec3 p) {
  return fract(sin(p.x * 214. + p.y * 241. + p.z * 123.) * 100. +
               cos(p.x * 42. + p.y * 41.2 + p.z * 32.) * 10.);
}

vec2 boxIntersection(in vec3 ro, in vec3 rd, vec3 boxSize) {
  vec3 m = 1.0 / rd;
  vec3 n = m * ro;
  vec3 k = abs(m) * boxSize;
  vec3 t1 = -n - k;
  vec3 t2 = -n + k;
  float tN = max(max(t1.x, t1.y), t1.z);
  float tF = min(min(t2.x, t2.y), t2.z);
  if (tN > tF || tF < 0.0)
    return vec2(-1.0);
  return vec2(tN, tF);
}

// https://iquilezles.org/articles/intersectors
vec2 sphereIntersection(in vec3 ro, in vec3 rd, float ra) {
  float b = dot(ro, rd);
  float c = dot(ro, ro) - ra * ra;
  float h = b * b - c;
  if (h < 0.0)
    return vec2(-1.0); // no intersection
  h = sqrt(h);
  return vec2(-b - h, -b + h);
}

void main() {
  vec4 p22 = vec4((clamp(texCoord, 0., 1.) * 2.0 - 1.0), 0.0, 1.0);

  // vec3 dirnear = (invproj2 * p22).xyz / (invproj2 * p22).w;
  // dirnear = normalize(mat3(invview2) * dirnear);

  ivec2 iFragCoord = ivec2(texCoord * scale);

  // int index = int(time) % BAYER_LIMIT;
  int prevIndex = int(texture2D(sunprev, texCoord).w);

  int iCoord = (iFragCoord.x + BAYER_LIMIT_H * iFragCoord.y) % BAYER_LIMIT;
  float lengthSun = 0.;
  vec3 pos = vec3(0.);
  ivec2 finCoords = iFragCoord;
  vec3 dir = -normalize(ldir);
  vec3 n = dir;
  vec3 W = (abs(n.x) > 0.99) ? vec3(0., 1., 0.) : vec3(1., 0., 0.);
  vec3 N = n;
  vec3 T = normalize(cross(N, W));
  vec3 B = cross(T, N);
  // return normalize(x*T + y*B + z*N);

  vec2 uv = texCoord * 2. - 1.;
  vec3 p = vec3(lpos + vec3(B * uv.x + -T * uv.y) * 200.);
  vec3 cam = p;
  if(renderSkyAndSun){
    if (traceSun(p, dir, 380., time)) {
      vec4 sspace = lightproj * lightview * vec4(p, 1.);

      sspace.xyz /= sspace.w;
      vec3 coords = sspace.xyz * 0.5f + 0.5f;

      finCoords = ivec2(coords.xy * scale);
      pos = p;
      lengthSun = coords.z;
    // lengthSun = length(p-cam);
    } else {
      vec4 sspace = lightproj * lightview * vec4(p, 1.);

      sspace.xyz /= sspace.w;
      vec3 coords = sspace.xyz * 0.5f + 0.5f;

      finCoords = ivec2(coords.xy * scale);
      lengthSun = coords.z;
    }
  }
  

  // vec3 cloudsPos(vec3 p, vec3 d, vec3 lig, inout float dist){
  float dist = 100.;
  float dist2 = 100.;

  imageStore(sun, ivec2(finCoords),
             vec4(vec3(lengthSun, lengthSun, dist2), float(prevIndex + 1)));
  // sun = vec4(vec3(pos), prevIndex);
}