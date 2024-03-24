
layout(binding = 4, Rgba16f) uniform image2D sun;

// layout (location = 0) out vec4 sun;

// out vec4 outputColor;
in vec2 texCoord;

// uniform sampler2D texture0;

uniform sampler2D sunprev;
uniform sampler3D worl;

uniform vec2 wh;
uniform float time;
uniform vec3 ldir;
uniform vec3 lpos;
uniform vec3 viewPos;

uniform mat4 lightproj;
uniform mat4 lightview;

uniform vec3 lpos2;

uniform mat4 lightproj2;
uniform mat4 lightview2;

uniform mat4 invview;
uniform mat4 invproj;

uniform mat4 invview2;
uniform mat4 invproj2;
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

float fbm(vec3 p, vec3 cam) {

  vec4 a = vec4(0.);

  a.x = texture(worl, p * 0.003).g;
  a.y = texture(worl, p * 0.012 + vec3(2.)).b;
  a.z = texture(worl, p * 0.018 + vec3(4.)).a;
  a.w = texture(worl, p * 0.003 + vec3(4.)).x;

  float b = clamp(2. * a.x + 1. * a.y + .5 * a.z, 0., 100.99);

  b *= a.w;
  b = max(b - 1., 0.);

  float Srb = clamp(
      remap(clamp((cam.z - 200.) / 15., 0., 0.09), 0., 0.09, 0., 1.), 0., 1.);

  return clamp(((Srb)*b), 0., 1.);
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

vec3 cloudsPos(vec3 p, vec3 d, inout float dist) {

  vec2 tts =
      boxIntersection(p - vec3(0., 0., 320.), d, vec3(100000., 100000., 1.));
  vec3 pos = vec3(50000.);

  if (tts.x > 0.) {
    // p += d*tts.x;
  }
  p += d * 3.;
  vec3 ccc = p;

  float transmission = 1.0;
  vec3 Ex = vec3(1.0);

  vec3 cam = p;

  // vec3 energyLoss = exp(-rayleighcoefficients*mm);

  vec3 accum = vec3(0.);
  float minus = 0.95;
  float mult = 1.0;

  bool firsth = false;

  for (int i = 0; i < 15; i++) {
    // accum += ph(length(cam), reyleighH)*length(div);
    float density =
        max(fbm(cam * mult, cam) - minus - abs(cam.z - 220.) * 0.00014, 0.);
    // density = smoothstep(0.,1.,density);
    density = clamp(density, 0., 1.);
    density = 1.0 - pow(1.0 - density, 4.);
    if (density > 0.00001 && length(cam - ccc) > 3.) {
      // if(!firsth){
      firsth = true;
      pos = cam;
      break;
      //}

      // if(length(cam)>7500.){break;}
    }
    cam += d * (1. - 0.5 * random3d(cam));
  }
  dist = length(cam - ccc);
  if (!firsth) {
    cam = ccc + d * 400.;
  }
  return cam;
}

void main() {
  vec4 p22 = vec4((clamp(texCoord, 0., 1.) * 2.0 - 1.0), 0.0, 1.0);

  vec3 dirnear = (invproj2 * p22).xyz / (invproj2 * p22).w;
  dirnear = normalize(mat3(invview2) * dirnear);

  ivec2 iFragCoord = ivec2(texCoord * scale);

  int index = int(time) % BAYER_LIMIT;
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
  vec3 p = vec3(lpos + vec3(B * uv.x + -T * uv.y) * 100.);
  vec3 cam = p;

  if (traceSun(p, dir, 380.)) {
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

  // vec3 cloudsPos(vec3 p, vec3 d, vec3 lig, inout float dist){
  float dist = 100.;
  float dist2 = 100.;
  /* if (iCoord == bayerFilter[index] && renderSkyAndSun ){

       vec3 nrmdir = -normalize(vec3(ldir.xyz));
       n = nrmdir;
       W = (abs(n.x)>0.99)?vec3(0.,1.,0.):vec3(1.,0.,0.);
       N = n;
       T = normalize(cross(N,W));
       B = cross(T,N);


       vec3 pp = vec3(lpos2 + vec3(B*uv.x + -T*uv.y)*200.).xzy;
      // pp = lpos2.xzy;
      // nrmdir = dirnear;
       vec3 finPos = cloudsPos(pp, nrmdir.xzy, dist);

       vec4 sspace = lightproj2 * lightview2 * vec4(finPos.xzy, 1.);

       sspace.xyz /= sspace.w;
       vec3 coords = sspace.xyz * 0.5f + 0.5f;

       if(coords.x >= 0. && coords.x <= 1. && coords.y >= 0. && coords.y <= 1.){
           finCoords = ivec2(coords.xy*scale);
           dist2 = coords.z;
       }
      // dist2 = dist;
   }else if (index > 0 || prevIndex >= 16){

       //col += texelFetch(skyTex, ivec2(wh.xy*texCoord),0).rgb;
       //pos += texture2D(sunprev, texCoord).rgb;
      // pos += imageLoad(sun, ivec2(iFragCoord)).xyz;
       dist2 = imageLoad(sun, ivec2(finCoords)).b;
       //finCoords = iFragCoord;
       prevIndex = 20;
   }
*/

  imageStore(sun, ivec2(finCoords),
             vec4(vec3(lengthSun, lengthSun, dist2), float(prevIndex + 1)));
  // sun = vec4(vec3(pos), prevIndex);
}