#version 450 compatibility
#extension GL_ARB_shading_language_packing : enable

layout(location = 0) out vec4 skyfin;

in vec2 texCoord;

uniform sampler2D skyprev;
uniform sampler3D worl;
uniform sampler2D suns;

uniform vec2 wh;
uniform float time;
uniform vec3 ldir;
uniform vec3 viewPos;

uniform vec3 lpos2;

uniform mat4 lightproj2;
uniform mat4 lightview2;

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

float ph(float h, float H) { return exp(-(abs(h) / H)); }
float PM(float cost, float g) {
  float a = 3. / (8. * 3.14159);
  float b = (1.0 - g * g) * (1.0 + cost * cost);
  float c = (2.0 + g * g) * pow(1.0 + g * g - 2. * g * cost, 3. / 2.);
  return a * (b / c);
}
float RayleighPhase(float cost) {
  return (3. / (16. * 3.14159)) * (1.0 + cost * cost);
}
vec3 boreyleigh(float costheta, vec3 wave) {
  float n = 1.00029;
  float N = 1.504;
  float a = 1.0 + costheta * costheta;
  float v = 3.14159 * 3.14159 * pow(n * n - 1., 2.);
  return (v / (3. * N * wave * wave * wave * wave * 0.000000000002)) * a;
}
vec3 bommie(float costheta, vec3 wave) {
  float T = 5.;
  float C = (0.6544 * T - 0.6510);
  vec3 Bm =
      0.434 * C * 3.14159 * ((4. * 3.14159 * 3.14159) / (wave * wave)) * 0.67;
  return 0.434 * C * ((4. * 3.14159 * 3.14159) / (wave * wave)) * 0.5 * Bm;
}
// NOT MY CODE///////////
bool intersect22(vec3 p, vec3 C, float size, vec3 d, inout vec2 t) {

  vec3 o_minus_c = p - C.xyz;

  float p2 = dot(d, o_minus_c);
  float q = dot(o_minus_c, o_minus_c) - (size * size);

  float discriminant = (p2 * p2) - q;

  float dRoot = sqrt(discriminant);
  t.x = -p2 - dRoot;
  t.y = -p2 + dRoot;

  return true;
}
vec3 Be(vec3 b0, float h, float H) { return b0 * exp(-h / H); }
vec3 Brr(vec3 wave) {
  float n = 1.029;
  float N = 10.;
  return 8. * pow(3.14159, 3.) *
         (pow(n * n - 1.0, 2.) / (3. * N * wave * wave * wave * wave));
}

vec3 sky(vec3 p, vec3 d, vec3 lig) {
  p = vec3(0., 0., 6400.);
  vec3 wavelengths = vec3(700., 530., 420.);
  lig.z = clamp(lig.z, 0., 1.);

  vec2 t = vec2(0.);
  float reyleighH = 4000.;
  float MieH = 1200.;

  vec3 accumulateLight = vec3(0.);
  vec3 accumulateLightMie = vec3(0.);
  if (intersect22(p, vec3(0., 0., 0.), 6420.0, d, t)) {
    // col = vec3(t.x);
    vec3 m = p;

    vec3 cam = p;
    vec3 fin = p + d * t.y;
    vec3 div = vec3(fin - cam) / 20.;
    float mm = length(cam - fin);

    vec3 accum = vec3(0.);
    vec3 accum11 = vec3(0.);
    vec3 accum1111 = vec3(0.);

    ///////////////////////////////////
    float PMMM = PM(max(dot(d, lig), 0.), 0.76);
    float PRRR = RayleighPhase(max(dot(d, lig), 0.0));
    vec3 coefficients = vec3(33.1, 13.5, 5.8);
    float Bs = 110.;
    vec3 Be0 = Be(coefficients, 0.0, 2000.);
    // vec3 BRrgb = vec3(6.5, 1.73, 2.30);
    vec3 BRrgb = Brr(wavelengths * 0.00109);
    vec3 BMrgb = vec3(0.01);
    ///////////////////////////////////

    float prevM = 0.0;
    float prevR = 0.0;

    for (int i = 0; i < 20; i++) {
      accum += ph(max(cam.z - 1660., 0.), reyleighH) * length(div);
      // accum11 += Bwave(wavelengths*0.01, cam.z, MieH)*length(div);;
      accum1111 += ph(max(cam.z - 1460., 0.), MieH) * length(div);

      // accum += ph(length(cam)-6500., reyleighH)*length(div);
      // accum += max(length(cam)-6360., 0.)*length(div);
      // accum += exp(-max(cam.z-m.z,0.)/20.)*length(div);
      // accum += exp(-max(cam.z-m.z,0.)/10.)*length(div);
      // accum11 += ph(length(cam)-6500., MieH)*length(div);

      vec3 accum2 = vec3(0.);
      vec3 accum222 = vec3(0.);
      float accum3 = 0.;
      // energy = energy*(1.0-rayleighcoefficients);
      if (intersect22(cam, vec3(0., 0., 0.), 6420.0, lig, t)) {
        vec3 cam2 = cam;
        vec3 fin2 = cam2 + lig * t.y;
        vec3 div2 = vec3(fin2 - cam2) / 20.;
        for (int k = 0; k < 20; k++) {
          // accum2 += Bwave(wavelengths*0.01, cam2.z, MieH)*length(div2);;
          accum222 += ph(max(cam2.z - 3160., 0.), reyleighH) * length(div2);
          // accum2 += ph(length(cam2)-6500., reyleighH)*length(div2);
          accum2 += max(length(cam2) - 6260., 0.) * length(div2);
          //  accum2 += exp(-(max((cam2.z-cam.z), 0.)/20.))*length(div2);
          // accum2 += exp(-max(cam2.z-cam.z,0.)/1.)*length(div2);
          // accum2 += ph(max(cam2.z-6360.,0.), MieH)*length(div2);

          // accum3 += ph(length(cam2)-6360., MieH)*length(div2);
          cam2 += div2;
        }

        // accumulateLightMie += (currM + prevM)/(2.0*length(div));
        // vec3 Br = boreyleigh(max(dot(d,lig),0.), wavelengths*0.15);
        // vec3 Bm = bommie(max(dot(d,lig),0.), wavelengths*0.15);
        vec3 t1 = BRrgb * accum222 + BMrgb * accum2;
        vec3 t2 = BRrgb * accum + BMrgb * accum1111;

        accumulateLight +=
            (exp(-pow(t1 * 0.0054, vec3(1.))) * exp(-t2 * 0.028)) *
            length(div * 1.) * .0005 * (PRRR * BRrgb * 230. + PMMM * 10.) *
            mix(vec3(0.9, 0.4, 0.2), vec3(0.9),
                clamp(dot(lig, vec3(0., 0., 1.)) * 4., 0., 1.));

        // T += exp(-(accum2*boreyleigh(max(dot(d,lig),0.), rayleighS) +
        // accum3*bommie(max(dot(d,lig),0.), mieS) ));
        // T += exp(-(wavelengths*accum2+bm*accum3));

        // vec3 Bm = bommie(max(dot(d,lig),0.), wavelengths*0.005);

        // prevM = currM;/
        // prevR = currR;
      }

      if (cam.z < 6360.) {
        break;
      }
      cam += div;
    }
    // vec3 F(vec3 wave, float s, float cost){

    // T += exp(-accum*rayleighcoefficients);
  }
  /// accumulateLight *= PR(max(dot(d,lig),0.))*3.;
  // accumulateLight *= Y(max(dot(d,lig),0.))*4.4;
  // return exp(-(accumulateLight)*2.);
  // return vec3(0.);
  return max(accumulateLight * 1.4 *
                 pow(clamp(dot(lig, vec3(0., 0., 1.)), 0., 1.), 2.),
             0.);
  // return
  // vec3(accumulateLight*9999999999999999.06+accumulateLightMie*999999999999999999.);
}

vec3 bommieconstant(vec3 wave) {
  float T = 5.;
  float C = (0.6544 * T - 0.6510);
  vec3 Bm =
      0.434 * C * 3.14159 * ((4. * 3.14159 * 3.14159) / (wave * wave)) * 0.67;
  return Bm;
}
float PR(float cost) { return (3. / (16. * 3.14159)) * (1.0 + cost * cost); }

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

float worley3d(vec3 p) {
  vec3 f = floor(p);

  float ll = 999.;
  for (int i = 0; i < 27; i++) {
    vec3 coords =
        vec3(float(i % 3) - 1., mod(float(i / 3) - 1., 3.), float(i / 9) - 1.);
    vec3 col = f + coords;
    vec3 curr =
        vec3(random3d(col), random3d(col + 2.), random3d(col + 4.)) - 0.5;
    float len = length((col + curr) - p);
    ll = min(ll, len);
  }
  return ll;
}

float hash(vec3 p3) {
  p3 = fract(p3 * .1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}
float noise222(in vec3 x) {
  vec3 i = floor(x);
  vec3 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);

  return mix(
      mix(mix(hash(i + vec3(0, 0, 0)), hash(i + vec3(1, 0, 0)), f.x),
          mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
      mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x),
          mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y),
      f.z);
}

float fbmss(vec3 p) {
  float scale = 0.1;
  float threshold = 0.3;
  float sum = 0.;
  p += vec3(5., 0., 0.);
  for (int i = 1; i <= 8; i++) {
    sum +=
        texture3D(worl, (p)*scale * pow(2., float(i))).w / pow(1.7, float(i));
  }
  return max(sum - threshold, 0.);
}

float fbm(vec3 p, vec3 cam) {
  // p.yz = rot(p.yz, iTime*0.3);
  // float a = texture(iChannel0, p).x*0.5 +texture(iChannel0, p*2.).y*0.25+
  // texture(iChannel0, p*4.).z*0.125+texture(iChannel0, p*8.).x*0.0625;
  // float a = texture(iChannel0, p).x*0.9;

  // float b = fbmss(p*122.);
  vec4 a = vec4(0.);

  a.x = texture3D(worl, p * 0.0011).g;
  a.y = texture3D(worl, p * 0.011 + vec3(2.)).b;
  a.z = texture3D(worl, p * 0.017 + vec3(4.)).a;
  a.w = texture3D(worl, p * 0.0011 + vec3(4.)).x;

  vec2 mr = texture3D(worl, p * 0.007).gx;
  /*
  float perlinWorley = a.w;

      // worley fbms with different frequencies
      vec3 worley = vec3(a.xyz);
      float wfbm = worley.x * .625 +
                           worley.y * .125 +
                           worley.z * .25;

      // cloud shape modeled after the GPU Pro 7 chapter
      float cloud = remap(perlinWorley, wfbm - 1., 1., 0., 1.);
      cloud = remap(cloud, .85, 1., 0., 1.); // fake cloud coverage

      return cloud;
  */
  // return max(a.x-.77,0.);
  // return a.x;
  float b = clamp(2. * a.x + 1. * a.y + .5 * a.z + mr.x * 1., 0., 100.99);
  /// b b, 2.1);
  // b = a.w;
  // return max(b-0.95,0.);
  // return max(b-0.3,0.);
  // b = remap(, b, 1., 0., 1.);
  // b *= clamp(fbm2(p.xy*2.)-abs(cam.z-6500.)*0.00014, 0.001, 1.);
  // b-= a.w;
  // b = max(b-1.2-max(1.-a.w*a.x,0.), 0.);
  b *= (a.w + mr.y) * 0.5;
  b = max(b - 1.4, 0.);
  // float b = texture3D(worl, p*22.).x;
  // b += max(fbmcc(p*.05)-0.4,0.);
  float Srb = clamp(
      remap(clamp((cam.z - 200.) / 15., 0., 0.09), 0., 0.09, 0., 1.), 0., 1.);
  // a *= clamp(abs(length(p)-6500.)*0.00013, 0.0, 1.);
  // a -= clamp((p.z)*0.4,0., 1.);
  // a = max(a,0.);
  // vec3 pos = vec3(0.,0.,6500.)-p;
  // pos.xz = rot(pos.xz, iTime*200.);
  // float cap = box(pos, vec3(100.,500.,100.));
  // cap = capsule(pos, vec3(0.,-2000., 6500.), vec3(0.,2000.,6500.), 100.);
  // cap = abs(cap)+0.01;
  // float density = exp(-cap*0.00002);
  return clamp(((Srb)*b), 0., 1.);
}

void swap(inout float t1, inout float t2) {
  float m = t1;
  t1 = t2;
  t2 = m;
}
bool intersectB(vec3 RayPosition, vec3 rayDir, inout float tmin,
                inout float tmax)
// bool intersect(const Ray &r)
{
  vec3 orig = RayPosition;
  vec3 dir = rayDir;
  tmin = (-15000. - orig.x) / dir.x;
  tmax = (15000. - orig.x) / dir.x;

  if (tmin > tmax)
    swap(tmin, tmax);

  float tymin = (-15000. - orig.y) / dir.y;
  float tymax = (15000. - orig.y) / dir.y;

  if (tymin > tymax)
    swap(tymin, tymax);

  if ((tmin > tymax) || (tymin > tmax))
    return false;

  if (tymin > tmin)
    tmin = tymin;

  if (tymax < tmax)
    tmax = tymax;

  float tzmin = (6500. - orig.z) / dir.z;
  float tzmax = (6400. - orig.z) / dir.z;

  if (tzmin > tzmax)
    swap(tzmin, tzmax);

  if ((tmin > tzmax) || (tzmin > tmax))
    return false;

  if (tzmin > tmin)
    tmin = tzmin;

  if (tzmax < tmax)
    tmax = tzmax;

  return true;
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

vec3 shadowS(vec3 pos) {
  vec4 sspace = lightproj2 * lightview2 * vec4(pos.xzy, 1.);
  sspace.xyz /= sspace.w;

  vec3 coords = sspace.xyz * 0.5 + 0.5;

  // float currDepth = texelFetch(sunTex, ivec2(2048.*coords), 0).r;
  float currDepth = texture2D(suns, coords.xy).b;
  // vec3 currPos = texture2D(sunTex, coords.xy).xyz;
  vec3 col = vec3(0.);
  if (coords.z < currDepth + 0.001 && coords.x >= 0. && coords.x <= 1. &&
      coords.y >= 0. && coords.y <= 1.) {
    col = vec3(1.);
  }

  if (length(pos.xz - viewPos.xz) > 90.) {
    //  col = vec3(1.);
  }

  return col;
}

vec4 clouds2(vec3 p, vec3 d, vec3 lig, inout float dist, inout vec3 sunShaf) {
  vec3 waves = vec3(0.00000519673, 0.0000121427, 0.0000296453);
  vec3 befo = p;
  vec2 tts =
      boxIntersection(p - vec3(0., 0., 200.), d, vec3(100000., 100000., 0.1));
  if (tts.x > 0.) {
    p += d * tts.y;
  }

  // vec3 drn = (p - befo)/20.;
  // for(int i = 0; i < 20; i++){
  //   befo += drn*max(random3d(befo),0.6);
  //   sunShaf += shadowS(befo)*10.;
  // }

  float transmission = 1.0;
  vec3 Ex = vec3(1.0);

  float phase = PM(max(dot(d, lig), 0.), 0.76);

  vec3 wavelengths = vec3(680., 550., 440.);

  vec2 t = vec2(0.);
  vec3 energy = vec3(1.); // 0000296453
  vec3 rayleighcoefficients = vec3(0.00000519673, 0.0000121427, 0.0000296453);
  vec3 T = vec3(0.);
  float reyleighH = 8500.;
  float MieH = 1200.;

  vec3 accumulateLight = vec3(0.);
  vec3 accumulateLightMie = vec3(0.);
  vec3 accumother = vec3(0.);
  // if(intersect(p, vec3(0.,0.,0.), 8500.0, d, t)){
  // col = vec3(t.x);
  vec3 ccc = p;
  vec3 cam = p;
  vec3 fin = p + d * t.y;
  vec3 div = vec3(fin - cam) / 40.;
  // vec3 precomputed = ((vec3(5.8,13.5,33.1)))*exp(-6.);
  // vec3 precomputed2 = vec3(0.210)*exp(-5.);
  float mm = length(cam - fin);
  // vec3 energyLoss = exp(-rayleighcoefficients*mm);
  float Is = 3.;
  vec3 Ip = vec3(0.);
  vec3 accum = vec3(0.);
  float accum11 = 0.;
  float minus = 0.95;
  float mult = 1.0;

  float zz = max(dot(vec3(0., 0., 1.), lig), 0.);
  /////////////
  // vec3 br = boreyleighconstant(wavelengths.zyx*0.0005);
  vec3 bm = bommieconstant(wavelengths * 0.024);

  ////////////
  // float pm = PM(max(dot(vec3(0.,0.,1.),lig),0.), 0.76)*5.;
  // float pr = PR(max(dot(vec3(0.,0.,1.),lig),0.))*2.;
  // vec3 sky2 = skyp3( normalize(vec3(0.,1.0,0.8)), lig);
  // vec3 sky(vec3 p, vec3 d, vec3 lig){
  vec3 sky2 = sky(vec3(0.), lig, lig);
  float pr2 = PR(max(dot(d, lig), 0.));
  float keepdensity = 0.;
  bool firsth = false;
  vec3 firstHit = p;
  for (int i = 0; i < 30; i++) {
    // accum += ph(length(cam), reyleighH)*length(div);
    float density =
        max(fbm(cam * mult, cam) - minus - abs(cam.z - 220.) * 0.00014, 0.);
    // density = smoothstep(0.,1.,density);
    density = clamp(density, 0., 1.);
    density = 1.0 - pow(1.0 - density, 4.);
    keepdensity += density;
    if (density > 0.00001) {
      if (!firsth) {
        dist = length(cam - ccc);
        firsth = true;
        firstHit = cam;
      }
      accum += density * 20.6;
      // accum11 += ph(length(cam)-6500., MieH)*length(div);

      vec3 accum2 = vec3(0.);
      // float accum3 = 0.;
      // energy = energy*(1.0-rayleighcoefficients);
      vec3 cam2 = cam;

      for (int k = 0; k < 10; k++) {
        float density2 = max(
            fbm(cam2 * mult, cam2) - minus - abs(cam2.z - 220.) * 0.00014, 0.);
        // density2 = smoothstep(0.,1.,density2);
        density2 = clamp(density2, 0., 1.);
        density2 = 1.0 - pow(1.0 - density2, 4.);

        // accum2 += ph(cam2.z, 1300.)*30.;
        accum2 += density2 * 40.1;
        // accum3 += ph(length(cam2)-6500., MieH)*length(div2);
        cam2 += lig * (1. - 0.2 * random3d(cam2));
      }

      Ex = Ex * exp(-accum2 * 0.01) *
           (1.0 - exp(-accum2 * max(zz, 0.05) * 310.8));
      transmission *= 1.0 - density;

      // sky(vec3 p, vec3 d, vec3 lig){
      /*accumulateLight += density * max(Ex, 0.0) * (1.0-exp(-density*0.07)) *
                              (pm2 * exp(-pow(.9,octave) * 0.1 *accum2) * 24.4 *
         sky2 + pr2  * exp(-.1  *pow(.9,octave) * accum2)*26.
                              + exp(-.1 * accum * (1.0 - zz * 0.9)) * 20. *sky2
                               ) ;*/

      float octave = 0.;

      for (int i = 0; i < 8; i++) {
        // vec3 pm2 =
        // vec3((PM(max(dot(d,lig),0.),0.76)*40.+PM(max(dot(d,lig),0.),pow(0.5,
        // octave)))*12.*max(1.0-zz,0.8));
        vec3 pm2 = vec3((PM(max(dot(d, lig), 0.), 0.76))) * 140. +
                   vec3((PM(max(dot(d, lig), 0.), pow(0.5, octave)))) * 13.5 +
                   vec3((PM(max(dot(d, lig), 0.), -0.5))) * 40.;
        // pm2 *= mix(vec3(0.9,0.6,0.2), vec3(0.9), max(dot(lig, vec3(0.,
        // 0., 1.)), 0.));

        accumulateLight +=
            (1. - exp(-density * 0.6)) * density * transmission *
            (pm2 + pr2 * 12.) * pow(0.5, octave) * 5. * sky2 *
            (exp(-accum2 * 0.1 * pow(.5, octave)) * exp(-accum * 0.1));

        octave += 0.2;
      }

    } // else{
      // vec3 shadowS(vec3 pos) {

    //  sunShaf += shadowS(cam)*10.;

    //}
    //    octave += 1.0;

    cam += d * (1. - 0.5 * random3d(cam));
    // if(length(cam)>7500.){break;}
  }
  // accumulateLight *= sky2;

  /*
  vec3 drn = (firstHit - befo)/30.;
  for(int i = 0; i < 30; i++){
    befo += drn*max(random3d(befo),0.7);
    sunShaf += shadowS(befo)*10.*PM(max(dot(d,lig),0.),0.76);
  }
  */
  return vec4(accumulateLight * 1.3, transmission);
}

#define BAYER_LIMIT 16
#define BAYER_LIMIT_H 4

const int bayerFilter[BAYER_LIMIT] =
    int[](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);

void main() {

  vec3 ldir2 = vec3(ldir.xy, ldir.z * -1.);

  vec2 iResolution = wh;
  uint seedCam = uint(time);
  vec2 smallOffset =
      ((vec2(rndf(seedCam), rndf(seedCam))) * 2. - 1.) / iResolution;

  vec2 thetaphi =
      (((texCoord)*2.0) - vec2(1.0)) * vec2(3.1415926535897932384626433832795,
                                            1.5707963267948966192313216916398);
  vec3 rayDirection = vec3(cos(thetaphi.y) * cos(thetaphi.x), sin(thetaphi.y),
                           cos(thetaphi.y) * sin(thetaphi.x));
  ivec2 iFragCoord = ivec2(texCoord * wh);

  int index = int(time) % BAYER_LIMIT;
  int prevIndex = int(texture2D(skyprev, texCoord).w);

  int iCoord = (iFragCoord.x + BAYER_LIMIT_H * iFragCoord.y) % BAYER_LIMIT;
  vec3 col = vec3(0.);
  if (iCoord == bayerFilter[index]) {
    vec3 d = rayDirection;
    vec3 lig = ldir2.xzy;
    vec3 sky2 =
        sky(vec3(0.), d, lig); // + exp(-acos(max(dot(d, lig),0.))*100.)*1.;

    if (dot(d, vec3(0., 0., 1.)) < 0.) {
      // sky2 = vec3(0.);
    }

    // sky2 *= max(dot(ldir, vec3(0.,1.,0.)),0.34);
    vec3 final = sky2 * 1.;

    float dist = 0.;
    vec3 posit = vec3(0.);
    vec3 sunshafts = vec3(0.);
    vec4 mmm = vec4(0.);
    vec3 sunsh = vec3(0.);
    mmm = clouds2(vec3(0., 0., 180.), d, normalize(vec3(lig.xy, abs(lig.z))),
                  dist, sunsh);
    mmm.rgb *= max(dot(ldir2, vec3(0., 1., 0.)), 0.1);
    mmm.rgb = pow(mmm.rgb, vec3(1.24));
    // }else{
    //   mmm = clouds2(vec3(0.,0.,5200.),d,vec3(lig.),dist,0.01);
    //   mmm.rgb *= 0.01;
    // }

    // distanceToClouds = dist;
    vec3 skys = final;
    final = final * mmm.w + pow(mmm.xyz, vec3(1.));

    float f = exp(-dist * .028);
    final.xyz = final.xyz * f + skys.xyz * (1.0 - f);
    // sn = fog();
    // final += sunshafts;
    float mmmms = max(pow(dot(ldir2, vec3(0., 1., 0.)) * 0.5 + 0.5, 4.), 0.01);
    col = final * mmmms;

    // col += vec3(sunsh/70.)*0.2;

  } else if (index > 0 || prevIndex >= 16) {

    // col += texelFetch(skyTex, ivec2(wh.xy*texCoord),0).rgb;
    col += texture2D(skyprev, texCoord).rgb;
    prevIndex = 20;
  }
  skyfin = vec4(col, float(prevIndex + 1));
}