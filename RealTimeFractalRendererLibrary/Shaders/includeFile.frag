#version 430

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

float box(vec3 p, vec3 c) {
  vec3 a = abs(p) - c;
  return max(a.x, max(a.y, a.z));
}

vec3 pal(float t, vec3 a) { return 0.5 + 0.5 * cos(2. * 3.14159 * t + a); }

float escape = 0.;

#define SceneRadius 100.0
#define DetailLevel 4.
#define StepFactor 1.

#define Pi 3.14159265359

float seed;

mat3 rotationMatrix(vec3 rotEuler) {
  float c = cos(rotEuler.x), s = sin(rotEuler.x);
  mat3 rx = mat3(1, 0, 0, 0, c, -s, 0, s, c);
  c = cos(rotEuler.y), s = sin(rotEuler.y);
  mat3 ry = mat3(c, 0, -s, 0, 1, 0, s, 0, c);
  c = cos(rotEuler.z), s = sin(rotEuler.z);
  mat3 rz = mat3(c, -s, 0, s, c, 0, 0, 0, 1);

  return rz * rx * ry;
}
void fold(inout vec3 z, vec3 o, vec3 n) {
  z -= 2. * n * min(dot(z - o, n), 0.);
}

float sdfIFS(vec3 z, inout float ls) {

  float scale = 2.;
  int Iterations = 20;
  mat3 rot = rotationMatrix(vec3(.5) * Pi);
  ls = 0.;
  vec3 n1 = normalize(vec3(1., 1., -1.));
  vec3 n2 = normalize(vec3(1., -1., -1.));
  // vec3 n3 = normalize(vec3(1., -1., 1.));
  escape = 0.;
  vec3 ot = vec3(1.);
  for (int i = 0; i < Iterations; i++) {
    fold(z, vec3(-.0), n1);
    fold(z, vec3(-.25), n2);
    // fold(z, vec3(-.25), n3);
    n1 *= rot;
    z = z * scale - sign(z) * (scale - 1.0);
    ot = min(abs(z), ot);
    escape += exp(-0.2 * dot(z, z));
  }
  // surf = Surface(true, 1.0, .1, vec3(0.), vec3(.8));
  // if(ot.r >= .75) surf = Surface(false, 1., .0, ot.ggb*30.*vec3(12., 2., .5),
  // vec3(0.1));
  if (ot.r >= .35)
    ls = ot.g * 3.;

  return length(z) * pow(scale, float(-Iterations));
}

float sdf(in vec3 pos, inout float ls) {
  float sSc = length(pos) - SceneRadius;
  float s = sdfIFS(pos, ls);
  return abs(sSc) > abs(s) || s > 0. ? s : sSc;
}

float sdf2(vec3 pos, inout float ls) {
  // Surface surf;
  return sdf(pos, ls);
}

float de23(vec3 p0, inout float ls) {
  vec4 p = vec4(p0, 1.);
  ls = 0.;
  p.xyz = abs(p.xyz);
  if (p.x < p.z)
    p.xz = p.zx;
  if (p.z < p.y)
    p.zy = p.yz;
  if (p.y < p.x)
    p.yx = p.xy;
  for (int i = 0; i < 8; i++) {
    if (p.x < p.z)
      p.xz = p.zx;
    if (p.z < p.y)
      p.zy = p.yz;
    if (p.y > p.x)
      p.yx = p.xy;
    p.xyz = abs(p.xyz);
    p *= (1.8 / clamp(dot(p.xyz, p.xyz), .0, 1.));
    p.xyz -= vec3(3.6, 1.9, 0.5);
    if (dot(p.xyz, p.xyz) < 0.1) {
      ls += 0.1;
    }
  }
  float m = 1.5;
  p.xyz -= clamp(p.xyz, -m, m);
  return length(p.xyz) / p.w;
}
float de33(vec3 p, inout float ls) {
  p = p.xzy;
  vec3 cSize = vec3(1., 1., 1.3);
  float scale = 1.;
  ls = 0.;
  for (int i = 0; i < 12; i++) {
    p = 2.0 * clamp(p, -cSize, cSize) - p;
    float r2 = dot(p, p + sin(p.z * .3));
    float k = max((2.) / (r2), .027);
    p *= k;
    scale *= k;
    if (dot(p, p) < 0.1)
      ls += 0.1;
  }
  float l = length(p.xy);
  float rxy = l - 4.0;
  float n = l * p.z;
  rxy = max(rxy, -(n) / 4.);
  return (rxy) / abs(scale);
}
vec3 fold2(vec3 p0) {
  vec3 p = p0;
  // if(abs(p.x) > 1.)p.x = 1.0-p.x;
  // if(abs(p.y) > 1.)p.y = 1.0-p.y;
  // if(abs(p.z) > 1.)p.z = 1.0-p.z;
  if (length(p) > 2.)
    return p;
  p = mod(p, 2.) - 1.;

  return p;
}

float DE33(vec3 p0, inout float ls) {
  vec4 p = vec4(p0, 1.);
  escape = 0.;
  ls = 0.;
  for (int i = 0; i < 12; i++) {
    // p.xyz = clamp(p.xyz, vec3(-2.3), vec3(2.3))-p.xyz;
    // p.xyz += sin(float(i+1));
    if (p.x > p.z)
      p.xz = p.zx;
    if (p.z > p.y)
      p.zy = p.yz;
    if (p.y > p.x)
      p.yx = p.xy;
    p = abs(p);
    // p.xyz = fold(p.xyz);
    p.xyz = fold2(p.xyz);

    // p.xyz = fract(p.xyz*0.5 - 1.)*2.-1.0;
    p.xyz = mod(p.xyz - 1., 2.) - 1.;
    p *= (1.0 / clamp(dot(p.xyz, p.xyz), 0.1, 1.));

    // p.xyz-=vec3(0.1,0.4,0.2);
    // p*=1.2;
    escape += exp(-0.2 * dot(p.xyz, p.xyz));
    if (dot(p.xyz, p.xyz) < 0.1) {
      ls += 0.1;
    }
  }
  p /= p.w;
  return length(p.xz) * 0.25;
}

vec3 fold3(vec3 p0) {
  vec3 p = p0;
  // if(abs(p.x) > 1.)p.x = 1.0-p.x;
  // if(abs(p.y) > 1.)p.y = 1.0-p.y;
  // if(abs(p.z) > 1.)p.z = 1.0-p.z;
  if (length(p) > 2.)
    return p;
  p = mod(p, 2.) - 1.;

  return p;
}

// float escape;
float DE32(vec3 p0, inout float ls) {
  ls = 0.;
  vec4 p = vec4(p0, 1.);
  escape = 0.;
  for (int i = 0; i < 12; i++) {
    // p.xyz = clamp(p.xyz, vec3(-2.3), vec3(2.3))-p.xyz;
    // p.xyz += sin(float(i+1));
    if (p.x > p.z)
      p.xz = p.zx;
    if (p.z > p.y)
      p.zy = p.yz;
    p = abs(p);
    p.xyz = fold3(p.xyz);

    // p.xyz = fract(p.xyz*0.5 - 1.)*2.-1.0;
    p.xyz = mod(p.xyz - 1., 2.) - 1.;
    p *= (1.05 / dot(p.xyz, p.xyz));
    // p*=1.2;
    escape += exp(-0.2 * dot(p.xyz, p.xyz));
  }
  p /= p.w;
  return length(p.xz) * 0.25;
}

void boxFold(inout vec4 p, vec3 s) { p.xyz = clamp(p.xyz, -s, s) * 2. - p.xyz; }

void sphereFold(inout vec4 p, float mr, float fr) {
  p *= fr / clamp(dot(p.xyz, p.xyz), mr, fr);
}

/*
 float Scale = 1.29842932;
 vec3 Offset = vec3(-3.2189972,-0.9234828,-4.5382586);
 vec3 BoxScale = vec3(4.4101877,10.,0.1742627);
 vec3 BoxFold = vec3(2.,0.33821572,0.40479362);
 vec3 Rotation = vec3(90.,0.,90.);
 int Iterations = 24;
 float MinRadius = 22.807018;
 float FixedRadius = 30.3693575;


vec4 orbitTrap = vec4(9999.);
float DEf(vec3 pos, inout float ls) {
        vec3 c = cos(Rotation*TWO_PI/360.);
        vec3 s = sin(Rotation*TWO_PI/360.);
        mat3 rx = mat3(1, 0, 0, 0, c.x, s.x, 0, -s.x, c.x);
        mat3 ry = mat3(c.y, 0, s.y, 0, 1, 0, -s.y, 0, c.y);
        mat3 rz = mat3(c.z, s.z, 0, -s.z, c.z, 0, 0, 0, 1);
        mat3 rot = rx*ry*rz;
orbitTrap = vec4(9999.);
        vec4 p = vec4(pos, 1);
ls = 0.;
        float trap = SceneRadius*2.;

        for (int i = 0; i < Iterations; i++) {
                orbitTrap = min(orbitTrap, vec4(abs(p.xyz), length(p.xyz)/p.w));
                sphereFold(p, MinRadius, FixedRadius);
                p.xyz += Offset;
                boxFold(p, BoxFold);
                p.xyz *= rot;
                //p.xyz = p.zxy * vec3(1, -1, -1);
                p *= Scale;
        if(dot(p.xyz,p.xyz)<3.9){
            ls += 0.1;
        }
                trap = min(boxDE(p, BoxScale), trap);
        }
        return trap;
}*/
/*
 float Scale; slider[1,1.5,5]
 vec3 Offset; slider[(-10,-10,-10),(1,1,1),(10,10,10)]
 vec3 BoxScale; slider[(0,0,0),(1,1,1),(10,10,10)]
 vec3 BoxFold; slider[(0,0,0),(.1,.1,.1),(2,2,2)]
 vec3 Rotation; slider[(0,0,0),(0,0,0),(360,360,360)]
 int Iterations; slider[1,16,64]
 float MinRadius; slider[0,1,50]
 float FixedRadius; slider[0,10,50]
*/
float boxDE(vec4 p, vec3 s) {
  vec3 a = abs(p.xyz) - s;
  return (min(max(max(a.x, a.y), a.z), 0.) + length(max(a, 0.))) / p.w;
}

float TWO_PI = 2. * 3.14159;
float Scale = 1.67517732;
vec3 Offset = vec3(4.1917026, 1.2732476, -5.9370528);
vec3 BoxScale = vec3(0.4949054, 0.2474527, 10.);
vec3 BoxFold = vec3(0.6300578, 0.81213874, 1.04046244);
vec3 Rotation = vec3(90., 270., 81.509436);
int Iterations = 16;
float MinRadius = 5.2785925;
float FixedRadius = 28.1250005;

float DE3333(vec3 pos, inout float ls) {
  ls = 0.;
  vec3 c = cos(Rotation.xxy * TWO_PI / 360.);
  vec3 s = sin(Rotation.xxy * TWO_PI / 360.);
  mat3 rx = mat3(1, 0, 0, 0, c.x, s.x, 0, -s.x, c.x);
  mat3 ry = mat3(c.y, 0, s.y, 0, 1, 0, -s.y, 0, c.y);
  mat3 rz = mat3(c.z, s.z, 0, -s.z, c.z, 0, 0, 0, 1);
  mat3 rot = rx * ry * rz;

  vec4 p = vec4(pos, 1);

  float trap = SceneRadius * 2.;

  for (int i = 0; i < Iterations; i++) {
    // orbitTrap = min(orbitTrap, vec4(abs(p.xyz), length(p.xyz)/p.w));
    sphereFold(p, MinRadius, FixedRadius);
    p.xyz += Offset;
    boxFold(p, BoxFold);
    p.xyz *= rot;
    p *= Scale;
    trap = min(boxDE(p, BoxScale), trap);
    if (dot(p.xyz, p.xyz) < 0.2) {
      ls += 0.1;
    }
  }
  return trap;
}

float sdBox(vec3 p, vec3 b) {
  vec3 di = abs(p) - b;
  float mc = max(di.x, max(di.y, di.z));
  return min(mc, length(max(di, 0.0)));
}
float l = 0.;
float rough = 1.;
vec3 cccc = vec3(1.);
vec3 metalAlb = vec3(1.);
float fractal(vec3 p) {
  vec3 w = p;
  vec3 q = p;

  q.xz = mod(q.xz + 1.0, 2.0) - 1.0;

  float d = sdBox(q, vec3(1.0));
  float s = 1.0;
  for (int m = 0; m < 6; m++) {
    float h = float(m) / 6.0;

    p = q - 0.5 * sin(abs(p.y) + float(m) * 3.0 + vec3(0.0, 3.0, 1.0));

    vec3 a = mod(p * s, 2.0) - 1.0;
    s *= 3.0;
    vec3 r = abs(1.0 - 3.0 * abs(a));

    float da = max(r.x, r.y);
    float db = max(r.y, r.z);
    float dc = max(r.z, r.x);
    float c = (min(da, min(db, dc)) - 1.0) / s;

    d = max(c, d);
  }

  vec2 res = vec2(d, 1.0);

  {
    d = length(w - vec3(0.22, 0.35, 0.4)) - 0.09;
    if (d < res.x)
      res = vec2(d, 2.0);
  }

  {
    d = w.y + 0.22;
    if (d < res.x)
      res = vec2(d, 3.0);
  }

  if (res.y < 1.5) {
    cccc = vec3(0.38) * vec3(1.2, 0.8, 0.6);

  } else if (res.y < 2.5) {
    // surfColor = vec3(0.37);
    cccc = vec3(5.);
    rough = 0.001;
  } else // if( tm.y<2.5 )
  {
    cccc = vec3(0.38) * vec3(1.2, 0.8, 0.6);
  }

  return res.x;
}
float escape2 = 0.;
float fractal_de15(vec3 p) {
  p = abs(p) - 1.2;
  if (p.x < p.z)
    p.xz = p.zx;
  if (p.y < p.z)
    p.yz = p.zy;
  if (p.x < p.y)
    p.xy = p.yx;
  escape2 = 0.;
  float s = 1.;
  for (int i = 0; i < 6; i++) {
    p = abs(p);
    float r = 2. / clamp(dot(p, p), .1, 1.);
    s *= r;
    p *= r;
    p -= vec3(.6, .6, 3.5);
    escape2 += exp(-0.2 * dot(p, p));
  }
  float a = 1.5;
  p -= clamp(p, -a, a);
  return length(p) / s;
}

float fractal_de46(vec3 p) {
  float s = 2.;
  float e = 0.;
  escape = 0.;
  for (int j = 0; ++j < 7;) {
    p.xz = abs(p.xz) - 2.3, p.z > p.x ? p = p.zyx : p,
    p.z = 1.5 - abs(p.z - 1.3 + sin(p.z) * .2), p.y > p.x ? p = p.yxz : p,
    p.x = 3. - abs(p.x - 5. + sin(p.x * 3.) * .2), p.y > p.x ? p = p.yxz : p,
    p.y = .9 - abs(p.y - .4),
    e = 12. * clamp(.3 / min(dot(p, p), 1.), .0, 1.) +
        2. * clamp(.1 / min(dot(p, p), 1.), .0, 1.),
    p = e * p - vec3(7, 1, 1), s *= e;
    escape += exp(-0.2 * dot(p, p));
  }
  return length(p) / s;
}
float desss(vec3 p) {
  vec4 o = vec4(p, 1);
  vec4 q = o;
  for (float i = 0.; i < 9.; i++) {
    o.xyz = clamp(o.xyz, -1., 1.) * 2. - o.xyz;
    o = o * clamp(max(.25 / dot(o.xyz, o.xyz), .25), 0., 1.) * vec4(11.2) + q;
  }
  return (length(o.xyz) - 1.) / o.w - 5e-4;
}

//////////////////////FROM JB//////////////////////////////
// tree shape
mat2 rotate2Dd(float r) { return mat2(cos(r), sin(r), -sin(r), cos(r)); }
float PI = 3.14159;
float deTree(vec3 p) {
  float d, a;
  d = a = 1.0f;
  for (int j = 0; j++ < 17;) {
    p.xz = abs(p.xz) * rotate2Dd(PI / 3.0f);
    d = min(d, max(length(p.zx) - 0.3f, p.y - 0.4f) / a);
    p.yx *= rotate2Dd(0.7f);
    p.y -= 3.0f;
    p *= 1.6f;
    a *= 1.6f;
  }
  return d;
}

float deTree2(vec3 p) {
  float d, a;
  d = a = 1.0f;
  for (int j = 0; j++ < 19;) {
    p.xz = abs(p.xz) * rotate2Dd(PI / 3.0f);
    d = min(d, max(length(p.zx) - 0.3f, p.y - 0.4f) / a);
    p.yx *= rotate2Dd(0.7f);
    p.y -= 3.0f;
    p *= 1.6f;
    a *= 1.6f;
  }
  return d;
}

///////////////////////////////////////////////////////////

float escape333 = 0.;
float jb(vec3 p) {
  float s = 3., e;
  s *= e = 3. / min(dot(p, p), 50.);
  p = abs(p) * e;
  escape333 = 0.;
  for (int i = 0; i++ < 12;) {
    p = vec3(2, 4, 2) - abs(p - vec3(4, 4, 2)),
    s *= e = 8. / min(dot(p, p), 9.), p = abs(p) * e;
    escape333 += exp(-0.2 * dot(p, p));
  }
  return min(length(p.xz) - .1, p.y) / s;
}
vec3 mb(vec3 p) {
  p.xyz = p.xzy;
  vec3 z = p;
  vec3 dz = vec3(0.0);
  float power = 8.0;
  float r, theta, phi;
  float dr = 1.0;

  float t0 = 1.0;
  for (int i = 0; i < 9; ++i) {
    r = length(z);
    if (r > 2.0)
      continue;
    theta = atan(z.y / z.x);
#ifdef phase_shift_on
    phi = asin(z.z / r) + iTime * 0.1;
#else
    phi = asin(z.z / r);
#endif

    dr = pow(r, power - 1.0) * dr * power + 1.0;

    r = pow(r, power);
    theta = theta * power;
    phi = phi * power;

    z = r * vec3(cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi)) + p;

    t0 = min(t0, r);
  }
  return vec3(0.5 * log(r) * r / dr, t0, 0.0);
}
/// JUMPHERE

void sphereFoldz(inout vec3 z, inout float dz) {
  float r2 = dot(z, z);
  if (r2 < 0.5) {
    float temp = 2.0;
    z *= temp;
    dz *= temp;
  } else if (r2 < 1.0) {
    float temp = 1.0 / r2;
    z *= temp;
    dz *= temp;
  }
}

void boxFoldz(inout vec3 z, inout float dz) {
  z = clamp(z, -1.0, 1.0) * 2.0 - z;
}

float mandelbox(vec3 z) {
  float scale = 2.0;
  vec3 offset = z;
  float dr = 1.0;
  for (int n = 0; n < 10; n++) {
    boxFoldz(z, dr);
    sphereFoldz(z, dr);
    z = scale * z + offset;
    dr = dr * abs(scale) + 1.0;
  }
  float r = length(z);
  return r / abs(dr);
}

float h(float x) { return 2. * 3.14159 * mod(10000. * sin(10000. * x), 1.); }

float N2(vec2 xy, float t, float phi) {
  float accum = 0.;
  for (int i = 0; i < 30; i++) {
    float k = exp(4. * ((float(i)) / 30.));
    float a = 1. / (k * sqrt(1. + xy.x * xy.x));
    float b = cos(k * ((xy.x) - sqrt(5.81 / k) * t) + t * phi * h(float(i)));
    accum += min(a * abs(b), 0.99);
  }

  return accum;
}

float N3(vec2 xy, float t, float phi) {
  float accum = 0.;
  for (int i = 0; i < 30; i++) {
    float k = exp(4. * ((float(i)) / 30.));
    float a = 1. / (k * sqrt(1. + xy.y * xy.y));
    float b = cos(k * ((xy.y) - sqrt(5.81 / k) * t) + t * phi * h(float(i)));
    accum += min(a * abs(b), 0.99);
  }

  return accum;
}
vec2 rotXX(vec2 a, float t) {
  float l = length(a);
  a /= l;

  float ang = (a.y < 0.) ? 2. * 3.14159 - acos(a.x) : acos(a.x);
  ang += t * 3.14159 / 180.;

  return vec2(cos(ang), sin(ang)) * l;
}

float comb(vec2 uv, float t) {
  float accum = 0.;
  for (int i = 0; i < 1; i++) {

    // accum += N2(rot(uv*0.17-10.,
    // 360.), t, fbmcc(uv*0.0001+float(i),t,0.1))
    // +N3(rot(uv+10.,360.), t,(fbmcc(uv*0.0003+float(i), t, 0.1)));
    accum += N2(rotXX(vec2(1.) - uv, (float(i) / 3.) * 360.), t + uv.y, 0.1) *
             N3(rotXX(vec2(1.) - uv, -(float(i) / 3.) * 360.), t + uv.x, 0.5);
    // accum *= N2(uv, t, fbmcc(uv*0.001, t, 0.1));
    // accum *= N3(uv, t, fbmcc(uv*0.003, t, 0.1));
  }

  return accum / 30.;
}

float fbmcc(in vec2 x, float ts, in float H) {
  float G = exp2(-H);
  float f = 1.0;
  float a = 1.;
  float t = 0.0;
  for (int i = 0; i < 5; i++) {
    t += a * comb(f * x, ts);
    f *= 2.0;
    a *= G;
  }
  return t;
}
float fbmss(vec2 p) {
  float t = 1.;
  return (pow(fbmcc((mod(p * 0.5 - 50., 100.) - 50.) + 100., t, 0.6), .7)) *
         112.1;
}

mat2 zx(float a) { return mat2(cos(a), sin(a), -sin(a), cos(a)); }

float deWATER(vec3 p) {
  float e, i = 0., j, f, a, w;
  // p.yz *= zx( .7 );
  f = .3;
  i < 45. ? p : p -= .001;
  e = p.y + 5.;
  for (a = j = .9; j++ < 30.; a *= .8) {
    vec2 m = vec2(1.) * zx(j);
    // float x = dot( p.xz, m ) * f + t + t; // time varying behavior
    float x = dot(p.xz, m) * f + 0.;
    w = exp(sin(x) - 1.);
    p.xz -= m * w * cos(x) * a;
    e -= w * a;
    f *= 1.2;
  }
  return e;
}

float isWat = 0.;
float mapPl(vec3 p) {
  // p.y -= 1.;
  return deWATER(p / 0.6) * 0.6;

  p = p.xzy;
  p.x *= 0.6;
  p.y *= 1.;
  float plane =
      p.z + (2. + ((sin(p.x + cos(p.y)) + cos(p.y + sin(p.x))) * 0.03 +
                   (sin(p.x * 5. + cos(p.y)) + cos(p.z * 5. + sin(p.x * 2.))) *
                       0.005));
  // rough = 0.001; cccc = vec3(0.6, 0.7, 0.9);
  // plane += fbmss(p.xy);
  return plane;

  // return deWATER(p/0.6)*0.6;
}

float map(vec3 p) {
  float ls = 0.;
  p = p.xzy;
  isWat = 0.;
  float c = 32.1;
  vec3 l33 = vec3(3., 1., 3.);
  //   p = p-c*clamp(round(p/c),-l33,l33);
  // p = floor(p*5.)/5.;
  l = 0.;
  rough = 1.;
  cccc = vec3(0.6);

  float aa3 = jb((vec3(-15.0, -2.7, 3.0) - p.xzy) / 10.) * 10.;
  float aa = fractal_de46((vec3(-5.0, 25.0, 13.0) - p) / 10.) * 10.;
  // float sphere = mb(p/15.).x*15.;
  float sphere = length(p) - 3.5;
  // float sphere2 = length(vec3(0., 25., 0.)-p)-3.5;
  // float sphere3 = length(vec3(0., 10., 10.)-p)-3.5;

  // float tree = deT(p);

  /////////////////////
  /*
         vec3 pos = p.xzy;
          const float dLeaves = deTree2( (pos + vec3( 0.0f, 7.0f, 0.0f ))/6.
     )*6.; float sceneDist = dLeaves; if ( sceneDist == dLeaves  ) {
              //hitPointSurfaceType = DIFFUSE;
              //const float noiseValue = 0.25f * perlinfbm( vec3( p.xyz ), 0.9f,
     6 );
              //hitPointColor = vec3( 0.06f, 0.13f, 0.02f ) * ( 1.0f -
     noiseValue ); cccc = vec3(0.2, 0.9, 0.2); rough = 0.1;
          }

          const float dTrunk = deTree( (pos + vec3( 0.0f, 7.0f, 0.0f ))/6. )*6.;
          sceneDist = min( sceneDist, dTrunk );
          if ( sceneDist == dTrunk  ) {
              cccc = vec3(0.9);
              rough = 1.;

              //hitPointSurfaceType = WOOD;
          }*/

  // return sceneDist;
  ///////////////////////

  // float sphere = mandelbox(p/2.)*2.;

  // float final = min(min(min(min(min(min(b2, a),b),c1),c2),c3),aa);
  //  float final = min(min(aa,aa),min(aa3, min(sphere, min(sphere2,
  //  min(sphere3, sceneDist)))));
  float final = min(aa, min(aa3, min(sphere, sphere)));
  // float final = min(aa3, sphere);
  // if(final == b2)l = 3.;
  // if(final == c1)c = vec3(0.9,0.1,0.1);
  // if(final == c2)c = vec3(0.1,0.9,0.1);
  if (final == aa) {
    rough = 1.0;
    cccc = pal(escape * 2., vec3(0.8, 0.4, 0.4));
  }
  if (final == aa3) {
    rough = 1.0;
    cccc = pal(escape333, vec3(0.4, 0.4, 0.8));
  }
  if (final == sphere) {
    rough = 0.01;
    l = 0.;
    cccc = vec3(0.9);
  }
  // if(final == sphere2 ){rough = 1.;l = 3.; cccc = vec3(0.9);}
  /// if(final == sphere3 ){rough = 1.;l = 3.; cccc = vec3(0.2, 0.2, 0.8);}

  // if(final == plane){}

  return final;

  // return fractal_de46(p/5.)*5.;
}

/*
float map(vec3 p){
rough = 1.;
cccc = vec3(1.);
l = 0.;

return fractal(p/10.)*10.;
}*/
/*
float map(vec3 p){
        l = 0.;
    rough = 0.1;
        cccc = vec3(1.);
        //return box(mod(p, 12.)-6., vec3(1.));
    float a = jb(p/6.)*6.;

    float b = box(p, vec3(.7, 15., .7));
    float c = length(vec3(5., 0., 0.)-p)-5.;
    float c2 = box(vec3(-10., 0., 0.)-p, vec3(4.));

    float final =min(a,min(b,min(c, c2)));
    if(final == b){
        l = 2.;
        cccc = vec3(0.9, 0.6, 0.2);
    }
    if(final == a){
      //  l=escape*0.006;
      rough = 1.;
        cccc = pal(escape, vec3(0.9,0.6,0.2));

    }
    if(final == c || final == c2){
        cccc = vec3(1.0);
        rough = 0.1;
        //rough = ;
    }
return final;

}
*/

float smin_op(float a, float b, float k) {
  float h = max(0., k - abs(b - a)) / k;
  return min(a, b) - h * h * h * k / 6.;
}

void sphere_fold(inout vec3 z, inout float dz) {
  float fixed_radius2 = 1.9;
  float min_radius2 = 0.1;
  float r2 = dot(z, z);
  if (r2 < min_radius2) {
    float temp = (fixed_radius2 / min_radius2);
    z *= temp;
    dz *= temp;
  } else if (r2 < fixed_radius2) {
    float temp = (fixed_radius2 / r2);
    z *= temp;
    dz *= temp;
  }
}
void box_fold(inout vec3 z, inout float dz) {
  float folding_limit = 1.0;
  z = clamp(z, -folding_limit, folding_limit) * 2.0 - z;
}
float de222(vec3 z) {
  vec3 offset = z;
  float scale = -2.8;
  float dr = 1.0;
  escape = 0.;
  for (int n = 0; n < 15; ++n) {
    box_fold(z, dr);
    sphere_fold(z, dr);
    z = scale * z + offset;
    dr = dr * abs(scale) + 1.0;
    escape += exp(-0.2 * dot(z.xyz, z.xyz));
  }
  float r = length(z);
  return r / abs(dr);
}
void box_fold(inout vec3 z) {
  float folding_limit = 1.0;
  z = clamp(z, -folding_limit, folding_limit) * 2.0 - z;
}
float DEer(vec3 p0) {
  // p0 = mod(p0, 2.)-1.;
  vec4 p = vec4(p0, 1.);
  escape = 0.;
  // p.xyz=1.0-abs(abs(p.xyz+sin(p.xyz)*1.)-1.);
  // p = abs(p);
  if (p.x < p.z)
    p.xz = p.zx;
  if (p.z > p.y)
    p.zy = p.yz;
  if (p.y > p.x)
    p.yx = p.xy;

  for (int i = 0; i < 12; i++) {
    // if(p.x > p.z)p.xz = p.zx;
    // if(p.z > p.y)p.zy = p.yz;
    if (p.y > p.x)
      p.yx = p.xy;
    p.xyz = abs(p.xyz);

    // box_fold(p.xyz);
    sphere_fold(p.xyz, p.w);
    // sphere_fold(p.xyz,p.w);

    // p.xyz = abs(p.xyz);
    uint seed = uint(p.x + p.y + p.z);
    p *= (3.3 / clamp(dot(p.xyz, p.xyz), 0.8, 1.6));
    p.xyz = abs(p.xyz) - vec3(2.5, 2.2, 1.3);
    // p*=1.2;
    p.xz -= sin(float(i) * 55.) * .5;
    escape += exp(-0.2 * dot(p.xyz, p.xyz));
    // vec3 norm = normalize(p.xyz);
    // float theta = acos(norm.z/length(norm.xyz));
    // float phi = atan(norm.y/norm.x);
    // escape = min(max(theta,phi),escape);
  }
  float m = 3.5;
  p.xyz -= clamp(p.xyz, -m, m);
  return length(p.xyz) / p.w;
}
float DEer2(vec3 p0) {
  // p0 = mod(p0, 2.)-1.;
  vec4 p = vec4(p0, 1.);
  escape2 = 0.;
  // p.xyz=1.0-abs(abs(p.xyz+sin(p.xyz)*1.)-1.);
  // p = abs(p);
  if (p.x < p.z)
    p.xz = p.zx;
  if (p.z > p.y)
    p.zy = p.yz;
  if (p.y > p.x)
    p.yx = p.xy;

  for (int i = 0; i < 12; i++) {
    // if(p.x > p.z)p.xz = p.zx;
    // if(p.z > p.y)p.zy = p.yz;
    if (p.y > p.x)
      p.yx = p.xy;
    p.xyz = abs(p.xyz);

    // box_fold(p.xyz);
    sphere_fold(p.xyz, p.w);
    // sphere_fold(p.xyz,p.w);

    // p.xyz = abs(p.xyz);
    uint seed = uint(p.x + p.y + p.z);
    p *= (3.9 / clamp(dot(p.xyz, p.xyz), 0.4, 2.));
    p.xyz = abs(p.xyz) - vec3(5.5, 2.2, 1.3);
    // p*=1.2;
    p.xz -= sin(float(i) * 55.) * .5;
    escape2 += exp(-0.2 * dot(p.xyz, p.xyz));
    // vec3 norm = normalize(p.xyz);
    // float theta = acos(norm.z/length(norm.xyz));
    // float phi = atan(norm.y/norm.x);
    // escape = min(max(theta,phi),escape);
  }
  float m = 3.5;
  p.xyz -= clamp(p.xyz, -m, m);
  return (length(p.xyz) / p.w) * 0.75;
}
const float pi = 3.14159;
vec2 rot(vec2 a, float c) {

  c = c * pi / 180.;
  // pythagoras theorem
  float l = length(a);

  a /= l;

  float ang = (a.y < 0.) ? 2. * pi - acos(a.x) : acos(a.x);
  ang += c;

  return vec2(cos(ang), sin(ang)) * l;
}

float prob = 1.;
/*
float map(vec3 p){
    cccc = vec3(0.9);
    rough = 0.01;
    l = 0.;
    prob = 1.;
    p = p.xzy;
        float aa2 = DEer((vec3(0.,5.,-10.0)-p)/10.)*10.;
        vec3 pos = vec3(-14.,-15.,-20.0)-p;
        pos.xy = rot(pos.xy, 180.);
                float aa22 = DEer2((pos)/10.)*10.;


    float plane = p.z + (5.+((sin(p.x + cos(p.y))+cos(p.y+sin(p.x)))*0.03+
    //float plane = length(vec3(15.,0.,0.)-p)-4.;

    (sin(p.x*5.+cos(p.y)) + cos(p.z*5.+sin(p.x*2.)))*0.005  ) );
    //float final = min(min(min(min(min(aa,plane),aaa),aa5),aa4),aa44);
   // float final = min(min(min(min(aa,aaaa),a22),aa44),plane);
    float final = min(aa2,aa22);
    //if(final == b2)l = 3.;
    //if(final == c1)c = vec3(0.9,0.1,0.1);
    //if(final == c2)c = vec3(0.1,0.9,0.1);
    vec3 color = vec3(0.5,0.5,0.8);
   // if(final == aa){a2 = 0.1;prob=0.9;c=vec3(0.5);}
       if(final == aa2){rough = 0.1;prob = 0.1;cccc=pal(escape,
vec3(0.9,0.9,0.2)); l=escape*0.00;} if(final == aa22){rough = 1.;prob =
0.8;cccc=pal(escape2, vec3(0.2,0.9,0.9)); l=escape2*0.000;}


    return final;
}
*/

float DEer222(vec3 p0) {
  // p0 = mod(p0, 2.)-1.;
  vec4 p = vec4(p0, 1.);
  escape = 0.;
  // p.xyz=1.0-abs(abs(p.xyz+sin(p.xyz)*1.)-1.);

  if (p.x < p.z)
    p.xz = p.zx;
  if (p.z > p.y)
    p.zy = p.yz;
  if (p.y > p.x)
    p.yx = p.xy;

  for (int i = 0; i < 12; i++) {
    // if(p.x > p.z)p.xz = p.zx;
    // if(p.z > p.y)p.zy = p.yz;
    if (p.y > p.x)
      p.yx = p.xy;
    // p.xyz = abs(p.xyz);

    // box_fold(p.xyz);
    sphere_fold(p.xyz, p.w);
    // p.xyz = abs(p.xyz);
    uint seed = uint(p.x + p.y + p.z);
    p *= (1.9 / clamp(dot(p.xyz, p.xyz), 0., 1.0));
    p.xyz = abs(p.xyz) - vec3(3.5, .5, 3.3);
    // p*=1.2;
    p.yxz -= sin(float(i) * 1.) * 0.9;
    escape += exp(-0.2 * dot(p.xyz, p.xyz));
    // vec3 norm = normalize(p.xyz);
    // float theta = acos(norm.z/length(norm.xyz));
    // float phi = atan(norm.y/norm.x);
    // escape = min(max(theta,phi),escape);
  }
  float m = 1.5;
  p.xyz -= clamp(p.xyz, -m, m);
  return length(p.xyz) / p.w;
}
/*
float map(vec3 p){
    cccc = vec3(0.9);
    rough = 0.01;
    l = 0.;
    prob = 1.;
    p = p.xzy;
        float aa2 = DEer222((vec3(-0.,28.,-0.0)-p)/14.)*14.;
        float fr = box(vec3(0., 0., -35.)-p, vec3(120., 120., 0.1));
        float fr2 = box(vec3(0., 0., 35.)-p, vec3(120., 120., 0.1));

    //float final = min(min(min(min(min(aa,plane),aaa),aa5),aa4),aa44);
   // float final = min(min(min(min(aa,aaaa),a22),aa44),plane);
    float final = min(aa2, min(fr, fr2));
    //if(final == b2)l = 3.;
    //if(final == c1)c = vec3(0.9,0.1,0.1);
    //if(final == c2)c = vec3(0.1,0.9,0.1);
    vec3 color = vec3(0.5,0.5,0.8);
   // if(final == aa){a2 = 0.1;prob=0.9;c=vec3(0.5);}
              if(final == aa2){rough = 0.1;cccc=pal(0., vec3(0.2,0.2,0.9));}
              if(final == fr || final == fr2){l=2.;cccc=vec3(cos(p.x*0.1),
sin(p.y*0.1), cos(p.z*0.1))*0.5 + 0.5;}


    return final;
}*/

/*
float map(vec3 p){
        l = 0.;
    rough = 1.;
        cccc = vec3(1.);
        //return box(mod(p, 12.)-6., vec3(1.));
    //float mand = mb(p/5.).x*5.;
    vec3 s = vec3(0., 0., 5.);
    vec3 l2 = vec3(0., 0., 10.);
    vec3 q = p - s*clamp(round(p/s),-l2,l2);
    float mand = box(q, vec3(1., 10., 1.));

float bx = box(vec3(0., -3., 0.)-p, vec3(20., 1., 20.));
float sx = length(vec3(10., 3., 3.)-p)-3.;
sx = box(vec3(20., 0., 0.)-p, vec3(0.1, 10., 10.));
float final = min(mand, min(bx,sx));
if(final == sx){ l = 32.;cccc=vec3(1.); }
return final;

}*/
/*
float map(vec3 p){
    l = 0.;
    rough = 1.;
    cccc = vec3(1.);
p = p.xzy;
p.y *= -1.;

float a = box(vec3(0.,4.,-5.)-p, vec3(10.,10.,0.2));
    float b = box(vec3(0.,4.,5.)-p, vec3(10.,10.,0.2));
    //float b = -box(vec3(0.,0.,0.)-p, vec3(10.,10.,10.2));

    float c1 = box(vec3(7.,4.,0.)-p, vec3(0.2,10.,10.));
    float c2 = box(vec3(-7.,4.,0.)-p, vec3(0.2,10.,10.));
    float c3 = box(vec3(0.,9.,0.)-p, vec3(10.,0.2,10.));
    float b2 = box(vec3(0.,4.,4.)-p, vec3(2.,2.,0.00000000000004));
    float aa = box(vec3(-3.5,4.,-1.)-p, vec3(1.3, 1.3, 4.));
    float aa2 = box(vec3(3.5,0.,-1.)-p, vec3(1.3, 1.3, 4.));
 //   float aa3 = box(vec3(3.5,4.,-1.)-p, vec3(1.3, 1.3, 4.));
//float aa3 = DE((vec3(2.,4.,2.)-p)/3.)*3.;
    float mand = mb((vec3(0., 4., -1.)-p)/2.).x*2.;

    float final = min(min(min(min(min(min(min(min(b2,
a),b),c1),c2),c3),mand),mand),mand);
    //float final = min(min(min(aa,b),aa2),aa3);
    if(final == b2)l = 3.;
    if(final == c1)cccc = vec3(0.9,0.1,0.1);
    if(final == c2)cccc = vec3(0.1,0.9,0.1);
    //if(final == aa){rough = 1.;}
    if(final == mand){rough = 0.1;}
   // if(final == aa3){rough = 0.01;}
return final;

}
*/

// 2D rotation function
mat2 rot33(float a) { return mat2(cos(a), sin(a), -sin(a), cos(a)); }

// "Amazing Surface" fractal
vec4 formula(vec4 p) {
  p.xz = abs(p.xz + 1.) - abs(p.xz - 1.) - p.xz;
  p.y -= .25;
  p.xy *= rot33(radians(35.));
  p = p * 2. / clamp(dot(p.xyz, p.xyz), .2, 1.);
  return p;
}

// Distance function
float de(vec3 pos) {
#ifdef WAVES
  pos.y += sin(pos.z - t * 6.) * .15; // waves!
#endif
  float hid = 0.;
  vec3 tpos = pos;
  tpos.z = abs(3. - mod(tpos.z, 6.));
  vec4 p = vec4(tpos, 1.);
  for (int i = 0; i < 4; i++) {
    p = formula(p);
  }
  float fr = (length(max(vec2(0.), p.yz - 1.5)) - 1.) / p.w;
  float ro = max(abs(pos.x + 1.) - .3, pos.y - .35);
  ro = max(ro, -max(abs(pos.x + 1.) - .1, pos.y - .5));
  pos.z = abs(.25 - mod(pos.z, .5));
  ro = max(ro, -max(abs(pos.z) - .2, pos.y - .3));
  ro = max(ro, -max(abs(pos.z) - .01, -pos.y + .32));
  float d = min(fr, ro);
  return d;
}

mat2 ro3t(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

float map33(vec3 p) {
  float scene = 100.;
  float t = floor(1. / 5.);
  float falloff = 1.0;
  for (float index = 0.; index < 8.; ++index) {
    p.xz *= ro3t(t / falloff);
    p = abs(p) - 0.5 * falloff;
    scene = min(scene, max(p.x, max(p.y, p.z)));
    falloff /= 1.8;
  }
  return -scene;
}

const int Iterations333 = 14;
const float detail = .00002;
const float vvvvv = 2.;

vec3 lightdir = normalize(vec3(0., -0.3, -1.));

float ot = 0.;
float det = 0.;

float hitfloor;
float hitrock;

float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

float tt;

float de3333333(vec3 pos) {
  hitfloor = 0.;
  hitrock = 0.;
  vec3 p = pos;
  p.xz = abs(.5 - mod(pos.xz, 1.)) + .01;
  float DEfactor = 1.;
  ot = 1000.;
  for (int i = 0; i < Iterations333; i++) {
    p = abs(p) - vec3(0., 2., 0.);
    float r2 = dot(p, p);
    float sc = vvvvv / clamp(r2, 0.4, 1.);
    p *= sc;
    DEfactor *= sc;
    p = p - vec3(0.5, 1., 0.5);
  }
  float rr = length(pos + vec3(0., -3.03, 1.85 - tt)) - .017;
  float fl = pos.y - 3.013;
  float d = min(fl, length(p) / DEfactor - .0005);
  d = min(d, -pos.y + 3.9);
  d = min(d, rr);
  if (abs(d - fl) < .0001)
    hitfloor = 1.;
  if (abs(d - rr) < .0001)
    hitrock = 1.;
  return d;
}
float cc, ss;

vec3 path(float ti) {
  return vec3(0., 2.5, 0.) + vec3(cos(ti) * .9, cos(ti * .5), sin(ti) * .8);
}

vec4 formula22(vec4 p) {
  // p.y-=t*.25;
  p.y = abs(3. - mod(p.y, 6.));
  for (int i = 0; i < 6; i++) {
    p.xyz = abs(p.xyz) - vec3(.0, 1., .0);
    p = p * 1.6 / clamp(dot(p.xyz, p.xyz), .2, 1.) - vec4(0.4, 1.5, 0.4, 0.);
    p.xz *= mat2(cc, ss, -ss, cc);
  }
  return p;
}

float texture1(vec3 p) {
  p = abs(1. - mod(p, 2.));
  vec3 c = vec3(3.);
  float es = 1000., l = 0.;
  for (int i = 0; i < 8; i++) {
    p = abs(p + c) - abs(p - c) - p;
    p /= clamp(dot(p, p), .25, 1.);
    p = p * -1.5 + c;
    es = min(min(abs(p.x), abs(p.y)), es);
  }
  return es * es;
}

float texture2(vec3 p) {
  // p.xz=abs(.75-mod(p.xz,1.5));
  p = formula22(vec4(p, 0.)).xyz;
  return .13 +
         clamp(pow(max(0., 1. - max(abs(p.x), abs(p.z))), 2.) * 2., .1, .7);
}

vec2 decccc(vec3 pos) {
  float aa = smoothstep(0., 1., clamp(cos(pos.y * .4) * 1.5, 0., 1.)) * 3.14159;
  cc = cos(aa);
  ss = sin(aa);
  float hid = 0.;
  vec3 tpos = pos;
  // tpos.xz=abs(1.5-mod(tpos.xz,3.))-1.5;
  vec4 p = vec4(tpos, 1.);
  float y = max(0., .3 - abs(pos.y - 3.3)) / .3;
  p = formula(p);
  float fl = pos.y - 3.7 - length(sin(pos.xz * 60.)) * .01;
  float fr = max(abs(p.z / p.w) - .01, length(p.zx) / p.w - .002);
  float bl = max(abs(p.x / p.w) - .01, length(p.zy) / p.w - .0005);
  fr = smin(bl, fr, .02);
  fr *= .9;
  // float fr=length(p.xyz)/p.w;
  fl -= (length(p.xz) * .005 + length(sin(pos * 3. + 5.)) * .15);
  fl *= .9;
  float d = smin(fl, fr, .7);
  if (abs(d - fl) < .2) {
    hid = 1.;
  }
  return vec2(d, hid);
}
/*
float map(vec3 p){
l = 0.;
rough = 0.1;
cccc = vec3(0.9);
float sphere = length(p)-40.;
float a = decccc(p/1.).x*1.;
float final = min(a, a);

return final;
}*/

float fractal_de7(vec3 p) {
  p = p.xzy;
  vec3 cSize = vec3(1., 1., 1.3);
  float scale = 1.;
  escape = 0.;
  for (int i = 0; i < 12; i++) {
    p = 2.0 * clamp(p, -cSize, cSize) - p;
    float r2 = dot(p, p + sin(p.z * .3));
    float k = max((2.) / (r2), .027);
    p *= k;
    scale *= k;
    escape += exp(-0.2 * dot(p, p));
  }
  float l = length(p.xy);
  float rxy = l - 4.0;
  float n = l * p.z;
  rxy = max(rxy, -(n) / 4.);
  return (rxy) / abs(scale);
}

////////////////////////////////////
/*
float map(vec3 p){
    p = p.xzy;
    p.y *= -1.;
    cccc = vec3(0.9,0.8,0.6);
    rough = 1.0;
    l = 0.;
    //float a = box(vec3(0.,4.,-5.)-p, vec3(10.,10.,0.2));
    //float b = box(vec3(0.,4.,5.)-p, vec3(10.,10.,0.2));
    //float c1 = box(vec3(7.,4.,0.)-p, vec3(0.2,10.,10.));
    //float c2 = box(vec3(-7.,4.,0.)-p, vec3(0.2,10.,10.));
    //float c3 = box(vec3(0.,9.,0.)-p, vec3(10.,0.2,10.));
    float b2 = box(vec3(0.,-9.,0.)-p, vec3(8.,0.00000000000004,8.));
    float aa = fractal_de7(vec3(20.,15.,-10.)-p);

    //float final = min(min(min(min(min(min(b2, a),b),c1),c2),c3),aa);
    float final = min(aa,b2);
    if(final == b2)l = 7.;
    //if(final == c1)c = vec3(0.9,0.1,0.1);
    //if(final == c2)c = vec3(0.1,0.9,0.1);
    if(final == aa){rough = 1.0;cccc=pal(escape,vec3(0.5,0.8,0.4));}

    return final;
}*/

#define fold45(p) (p.y > p.x) ? p.yx : p
float dexx(vec3 p) {
  float scale = 2.1, off0 = .8, off1 = .3, off2 = .83;
  vec3 off = vec3(2., .2, .1);
  float s = 1.0;
  escape = 0.;
  for (int i = 0; ++i < 20;) {
    p.xy = abs(p.xy);
    p.xy = fold45(p.xy);
    p.y -= off0;
    p.y = -abs(p.y);
    p.y += off0;
    p.x += off1;
    p.xz = fold45(p.xz);
    p.x -= off2;
    p.xz = fold45(p.xz);
    p.x += off1;
    p -= off;
    p *= scale;
    p += off;
    s *= scale;
    escape += exp(-0.2 * dot(p, p));
  }
  return length(p) / s;
}

float dezzz(vec3 p) {
  float e = 4., s;
  for (int j = 0; ++j < 7; p = abs(p) - .9, e /= s = min(dot(p, p), .9), p /= s)
    ;
  return length(p.xz) / e;
}

float dezzz2(vec3 p) {
  float s = 2., r2;
  p = abs(p);
  for (int i = 0; i < 12; i++) {
    p = 1. - abs(p - 1.);
    r2 = (i % 3 == 1) ? 1.1 : 1.2 / dot(p, p);
    p *= r2;
    s *= r2;
  }
  return length(cross(p, normalize(vec3(1)))) / s - 0.005;
}
float dexxx(vec3 p) {
  p = sin(2.8 * p + 5. * sin(p * .3));
  float s = 2., e;
  for (int i = 0; i++ < 6;)
    p = abs(p - 1.7) - 1.5, s *= e = 2.3 / clamp(dot(p, p), .3, 1.2),
    p = abs(p) * e;
  return length(p.zy) / s;
}

float exc = 0.;

mat2 rotate2D(float r) { return mat2(cos(r), sin(r), -sin(r), cos(r)); }
float dexcc(vec3 p0) {
  vec4 p = vec4(p0, 3.);
  exc = 0.;
  p *= 2. / min(dot(p.xyz, p.xyz), 30.);
  for (int i = 0; i < 14; i++) {
    p.xyz = vec3(2., 4., 2.) - (abs(p.xyz) - vec3(2., 4., 2.));
    p.xyz = mod(p.xyz - 4., 8.) - 4.;
    p *= 9. / min(dot(p.xyz, p.xyz), 12.);
    exc += exp(-0.2 * dot(p.xyz, p.xyz));
  }
  p.xyz -= clamp(p.xyz, -1.2, 1.2);
  return length(p.xyz) / p.w;
}
float dezzz2xxx22(vec3 p0) {
  vec4 p = vec4(p0, 1.);
  escape = 0.;
  p = abs(p);
  if (p.x < p.z)
    p.xz = p.zx;
  if (p.z < p.y)
    p.zy = p.yz;
  if (p.y < p.x)
    p.yx = p.xy;
  for (int i = 0; i < 8; i++) {
    if (p.x < p.z)
      p.xz = p.zx;
    if (p.z < p.y)
      p.zy = p.yz;
    if (p.y < p.x)
      p.yx = p.xy;
    p.xyz = abs(p.xyz);
    p *= (1.8 / clamp(dot(p.xyz, p.xyz), -1.0, 1.));
    p.xyz -= vec3(0.3, 1.9, 0.4);
    escape += exp(-0.2 * dot(p.xyz, p.xyz));
  }
  float m = 1.5;
  p.xyz -= clamp(p.xyz, -m, m);
  return length(p.xyz) / p.w;
}

float dezzz2xxx(vec3 p0) {
  vec4 p = vec4(p0, 1.);
  escape = 0.;
  p = abs(p);
  if (p.x < p.z)
    p.xz = p.zx;
  if (p.z < p.y)
    p.zy = p.yz;
  if (p.y < p.x)
    p.yx = p.xy;
  for (int i = 0; i < 12; i++) {
    if (p.x < p.z)
      p.xz = p.zx;
    if (p.z < p.y)
      p.zy = p.yz;
    if (p.y < p.x)
      p.yx = p.xy;
    p = abs(p);
    p *= (1.9 / clamp(dot(p.xyz, p.xyz), 0.1, 1.));
    p.xyz -= vec3(0.2, 1.9, 0.6);
    escape += exp(-0.2 * dot(p.xyz, p.xyz));
  }
  float m = 1.2;
  p.xyz -= clamp(p.xyz, -m, m);
  return (length(p.xyz) / p.w);
}
/*
float map(vec3 p){
  p = p.xzy;
  p.z *= -1.;
  cccc = vec3(0.9,0.8,0.6);
  rough = 1.0;
  l = 0.;
  //float a = box(vec3(0.,4.,-5.)-p, vec3(10.,10.,0.2));
  //float b = box(vec3(0.,4.,5.)-p, vec3(10.,10.,0.2));
  //float c1 = box(vec3(7.,4.,0.)-p, vec3(0.2,10.,10.));
  //float c2 = box(vec3(-7.,4.,0.)-p, vec3(0.2,10.,10.));
  //float c3 = box(vec3(0.,9.,0.)-p, vec3(10.,0.2,10.));
  float b2 = box(vec3(0.,0.,-2.)-p, vec3(20.,20.,.1));
  vec3 c = vec3(10., 10., 1.);
  float b3 = box(vec3(0.,0.,15.)-p, vec3(20.,20.,.1));

  float aa = dexx(p/10.)*10.;
  float bbb = jb((vec3(0., 16., 0.)-p.xzy)/10.)*10.;
  float aa2 = dezzz2((vec3(0., 0., 12.)-p)/12.)*12.;
    //  aa2 = max(aa2, box(vec3(0., 0., 12.)-p, vec3(16.)));

  //float final = min(min(min(min(min(min(b2, a),b),c1),c2),c3),aa);
  float final = min(min(aa, bbb),min(b2, b3));
  if(final == b2){l = 20.;cccc = vec3(0.2,0.8,0.9);}
      if(final == b3){l = 20.;cccc = vec3(0.9, 0.5, 0.2);}

  //if(final == c1)c = vec3(0.9,0.1,0.1);
  //if(final == c2)c = vec3(0.1,0.9,0.1);
   if(final == aa){rough = 0.1;cccc=vec3(0.9);}
  if(final == bbb){rough = 1.;cccc=pal(escape333*0.5, vec3(0.9,0.6,0.2));}
  if(final == aa2){rough = 1.; cccc=vec3(0.8);}
  return final;
}*/

// Plane with normal n (n is normalized) at some distance from the origin
float fPlane(vec3 p, vec3 n, float distanceFromOrigin) {
  return dot(p, n) + distanceFromOrigin;
}

float vmax(vec3 v) { return max(max(v.x, v.y), v.z); }

// Cheap Box: distance to corners is overestimated
float fBoxCheap(vec3 p, vec3 b) { // cheap box
  return vmax(abs(p) - b);
}

// third trellis type structure
#define fold45(p) (p.y > p.x) ? p.yx : p
float deFractal3(vec3 p) {
  float s = 1.0f;
  float scale = 2.1f, off0 = 0.8f, off1 = 0.3f, off2 = 0.83f;
  vec3 off = vec3(2.0f, 0.2f, 0.1f);
  for (int i = 0; ++i < 20;) {
    p.xy = abs(p.xy);
    p.xy = fold45(p.xy);
    p.y -= off0;
    p.y = -abs(p.y);
    p.y += off0;
    p.x += off1;
    p.xz = fold45(p.xz);
    p.x -= off2;
    p.xz = fold45(p.xz);
    p.x += off1;
    p -= off;
    p *= scale;
    p += off;
    s *= scale;
  }
  return length(p) / s;
}

float deFractal4(vec3 pos) {
  escape = 0.;
  vec3 tpos = pos;
  tpos.xz = abs(0.5f - mod(tpos.xz, 1.0f));
  vec4 p = vec4(tpos, 1.0f);
  float y = max(0.0f, 0.35f - abs(pos.y - 3.35f)) / 0.35f;
  for (int i = 0; i < 7; i++) {
    p.xyz = abs(p.xyz) - vec3(-0.02f, 1.98f, -0.02f);
    p = p * (2.0f + 0.0f * y) / clamp(dot(p.xyz, p.xyz), 0.4f, 1.0f) -
        vec4(0.5f, 1.0f, 0.4f, 0.0f);
    p.xz *= mat2(-0.416f, -0.91f, 0.91f, -0.416f);

    escape += exp(-0.2 * dot(p.xyz, p.xyz));
  }
  return (length(max(abs(p.xyz) - vec3(0.1f, 5.0f, 0.1f), vec3(0.0f))) -
          0.05f) /
         p.w;
}

float deStairs(vec3 P) {
  vec3 Q;
  float a, d = min((P.y - abs(fract(P.z) - 0.5f)) * 0.7f, 1.5f - abs(P.x));
  for (a = 2.0f; a < 6e2f; a += a)
    Q = P * a, Q.xz *= rotate2D(a),
    d += abs(dot(sin(Q), Q - Q + 1.0f)) / a / 7.0f;
  return d;
}

// Repeat around the origin by a fixed angle.
// For easier use, num of repetitions is use to specify the angle.
float pModPolar(inout vec2 p, float repetitions) {
  float angle = 2. * 3.14159 / repetitions;
  float a = atan(p.y, p.x) + angle / 2.;
  float r = length(p);
  float c = floor(a / angle);
  a = mod(a, angle) - angle / 2.;
  p = vec2(cos(a), sin(a)) * r;
  // For an odd number of repetitions, fix cell index of the cell in -x
  // direction (cell index would be e.g. -5 and 5 in the two halves of the
  // cell):
  if (abs(c) >= (repetitions / 2.))
    c = abs(c);
  return c;
}

// ==============================================================================================
// ==============================================================================================

vec3 GetColorForTemperature(const float temperature) {
  mat3 m = (temperature <= 6500.0f)
               ? mat3(vec3(0.0f, -2902.1955373783176f, -8257.7997278925690f),
                      vec3(0.0f, 1669.5803561666639f, 2575.2827530017594f),
                      vec3(1.0f, 1.3302673723350029f, 1.8993753891711275f))
               : mat3(vec3(1745.0425298314172f, 1216.6168361476490f,
                           -8257.7997278925690f),
                      vec3(-2666.3474220535695f, -2173.1012343082230f,
                           2575.2827530017594f),
                      vec3(0.55995389139931482f, 0.70381203140554553f,
                           1.8993753891711275f));
  return mix(
      clamp(vec3(m[0] / (vec3(clamp(temperature, 1000.0f, 40000.0f)) + m[1]) +
                 m[2]),
            vec3(0.0f), vec3(1.0f)),
      vec3(1.0f), smoothstep(1000.0f, 0.0f, temperature));
}

#define NOHIT 0
#define EMISSIVE 1
#define DIFFUSE 2
#define METALLIC 3
#define MIRROR 4

int hitPointSurfaceType = NOHIT;
vec3 hitPointColor = vec3(0.0f);

// ==============================================================================================
// ==============================================================================================

// overal distance estimate function - the "scene"
// hitPointSurfaceType gives the type of material
// hitPointColor gives the albedo of the material
float raymarchEpsilon = 0.1;

float de333(vec3 p) {
  // init nohit, far from surface, no diffuse color
  hitPointSurfaceType = NOHIT;
  float sceneDist = 1000.0f;
  hitPointColor = vec3(0.0f);

  vec3 pCache = p;
  vec3 floorCielingColor = vec3(0.9f);
  float dFloor = fPlane(p, vec3(0.0f, 1.0f, 0.0f), 4.0f);
  sceneDist = min(dFloor, sceneDist);
  if (sceneDist == dFloor && dFloor <= raymarchEpsilon) {
    hitPointColor = floorCielingColor;
    hitPointSurfaceType = DIFFUSE;
  }

  // hexagonal symmetry
  pModPolar(p.xz, 6.0f);

  float dWall = fPlane(p, vec3(-1.0f, 0.0f, 0.0f), 35.0f);
  sceneDist = min(dWall, sceneDist);
  if (sceneDist == dWall && dWall <= raymarchEpsilon) {
    hitPointColor = floorCielingColor * 0.7f;
    hitPointSurfaceType = DIFFUSE;
  }

  float dUpperWall = fPlane(p, vec3(-1.0f, -1.0f, 0.0f), 40.0f);
  sceneDist = min(dUpperWall, sceneDist);
  if (sceneDist == dUpperWall && dUpperWall <= raymarchEpsilon) {
    hitPointColor = vec3(1.0f);
    hitPointSurfaceType = DIFFUSE;
  }

  float dLightBars =
      fBoxCheap(p - vec3(0.0f, 10.0f, 0.0f), vec3(10.0f, 0.1f, 1.0f));
  sceneDist = min(dLightBars, sceneDist);
  if (sceneDist == dLightBars && dLightBars <= raymarchEpsilon) {
    hitPointColor = GetColorForTemperature(4800.0f);
    hitPointSurfaceType = EMISSIVE;
  }

  float dLightBarHousing =
      fBoxCheap(p - vec3(0.0f, 10.15f, 0.0f), vec3(11.0f, 0.2f, 1.1f));
  sceneDist = min(dLightBarHousing, sceneDist);
  if (sceneDist == dLightBarHousing && dLightBarHousing <= raymarchEpsilon) {
    hitPointColor = vec3(0.618f);
    hitPointSurfaceType = DIFFUSE;
  }

  float dFractal3 = deFractal3(pCache * 0.5f) / 0.5f;
  sceneDist = min(dFractal3, sceneDist);
  if (sceneDist == dFractal3 && dFractal3 <= raymarchEpsilon) {
    hitPointColor = vec3(0.9f, 0.1f, 0.05f);
    hitPointSurfaceType = EMISSIVE;
  }

  float dFractal4 = deFractal4((pCache * 0.2f)) / 0.2f;
  sceneDist = min(dFractal4, sceneDist);
  if (sceneDist == dFractal4 && dFractal4 <= raymarchEpsilon) {
    hitPointColor = vec3(0.15);
    hitPointSurfaceType = DIFFUSE;
  }

  float dStairs = deStairs((pCache * 0.3f)) / 0.3f;
  sceneDist = min(dStairs, sceneDist);
  if (sceneDist == dStairs && dStairs <= raymarchEpsilon) {
    hitPointColor = vec3(0.45f);
    hitPointSurfaceType = DIFFUSE;
  }

  return sceneDist;
}
/*
float map(vec3 p){
    //p = p.xzy;
    p.y += 45.;
    //p.z *= -1.;
   cccc = vec3(0.9,0.8,0.6);
    rough = 0.2;
    l = 0.;
    //float a = box(vec3(0.,4.,-5.)-p, vec3(10.,10.,0.2));
    //float b = box(vec3(0.,4.,5.)-p, vec3(10.,10.,0.2));
    //float c1 = box(vec3(7.,4.,0.)-p, vec3(0.2,10.,10.));
    //float c2 = box(vec3(-7.,4.,0.)-p, vec3(0.2,10.,10.));
    //float c3 = box(vec3(0.,9.,0.)-p, vec3(10.,0.2,10.));
    float a = de333(p/10.)*10.;
    cccc = hitPointColor;
    if(hitPointSurfaceType == EMISSIVE){
        l = 14.;
    }
    return a;

}
*/
/*
float map(vec3 p){
    p = p.xzy;
    p.z *= -1.;
    cccc = vec3(0.9,0.8,0.6);
    rough = 1.0;
    l = 0.;

    float bbb = jb((vec3(0., 16., 0.)-p.xzy)/5.)*5.;
    float aa2 = dexx(p/13.)*13.;
    //float final = min(min(min(min(min(min(b2, a),b),c1),c2),c3),aa);
    float final = min(aa2, bbb);

    //if(final == c1)c = vec3(0.9,0.1,0.1);
    //if(final == c2)c = vec3(0.1,0.9,0.1);
    if(final == bbb){rough = 1.0;cccc=pal(escape333, vec3(0.9,0.6,0.8));}
    if(final == aa2){rough = 0.1; cccc = vec3(0.9,0.6,0.2);}
    return final;
}
*/
/*
float map(vec3 p){
    p = p.xzy;
    p.z *= -1.;
    cccc = vec3(0.9,0.8,0.6);
    rough = 1.0;
    l = 0.;
    //float a = box(vec3(0.,4.,-5.)-p, vec3(10.,10.,0.2));
    //float b = box(vec3(0.,4.,5.)-p, vec3(10.,10.,0.2));
    //float c1 = box(vec3(7.,4.,0.)-p, vec3(0.2,10.,10.));
    //float c2 = box(vec3(-7.,4.,0.)-p, vec3(0.2,10.,10.));
    //float c3 = box(vec3(0.,9.,0.)-p, vec3(10.,0.2,10.));
    float b2 = box(vec3(0.,0.,0.)-p, vec3(2.,2.,12.));
    vec3 c = vec3(10., 10., 1.);
        float b3 = box(mod(p,c)-c*0.5, vec3(0.1,0.1,2.));

    float aa = dexx(p/10.)*10.;
    float bbb = jb((vec3(0., 16., 0.)-p.xzy)/10.)*10.;
    float aa2 = dezzz2((vec3(0., 0., 12.)-p)/12.)*12.;
      //  aa2 = max(aa2, box(vec3(0., 0., 12.)-p, vec3(16.)));

    //float final = min(min(min(min(min(min(b2, a),b),c1),c2),c3),aa);
    float final = min(min(aa, bbb),min(aa, aa));
    if(final == b2){l = 7.;cccc = vec3(0.2,0.8,0.9);}
        if(final == b3){l = 14.;cccc = vec3(0.9, 0.5, 0.2);}

    //if(final == c1)c = vec3(0.9,0.1,0.1);
    //if(final == c2)c = vec3(0.1,0.9,0.1);
     if(final == aa){rough = 1.;cccc=vec3(0.9,0.67,0.2);}
    if(final == bbb){rough = 0.1;cccc=vec3(0.9,0.67,0.2);}
    if(final == aa2){rough = 1.; cccc=vec3(0.8);}
    return final;
}*/
/*
float map(vec3 p){
    p = p.xzy;
    p.z *= -1.;
    cccc = vec3(0.9,0.8,0.6);
    rough = 1.0;
    l = 0.;

    float bbb = jb((vec3(0., 16., 0.)-p.xzy)/5.)*5.;
    float aa2 = dexx(p/13.)*13.;
    //float final = min(min(min(min(min(min(b2, a),b),c1),c2),c3),aa);
    float final = min(aa2, bbb);

    //if(final == c1)c = vec3(0.9,0.1,0.1);
    //if(final == c2)c = vec3(0.1,0.9,0.1);
    if(final == bbb){rough = 0.1;cccc=pal(escape333, vec3(0.9,0.6,0.8));}
    if(final == aa2){rough = 0.1; cccc = vec3(0.9,0.6,0.2);}
    return final;
}*/

float fractal_de51(vec3 p) {
  for (int j = 0; ++j < 8;)
    p.z -= .3, p.xz = abs(p.xz), p.xz = (p.z > p.x) ? p.zx : p.xz,
               p.xy = (p.y > p.x) ? p.yx : p.xy, p.z = 1. - abs(p.z - 1.),
               p = p * 3. - vec3(10, 4, 2);

  return length(p) / 6e3 - .001;
}

/*
float map(vec3 p){

    cccc = vec3(0.9,0.8,0.6);
    rough = 1.;
    l = 0.;
      //  return length(p)-3.;
    p = p.xzy;

    //float a = box(vec3(0.,4.,-5.)-p, vec3(10.,10.,0.2));
    //float b = box(vec3(0.,4.,5.)-p, vec3(10.,10.,0.2));
    //float c1 = box(vec3(7.,4.,0.)-p, vec3(0.2,10.,10.));
    //float c2 = box(vec3(-7.,4.,0.)-p, vec3(0.2,10.,10.));
    //float c3 = box(vec3(0.,9.,0.)-p, vec3(10.,0.2,10.));
    float b2 = box(vec3(-5.,-59.5,15.)-p, vec3(22.,0.00000000000004,22.));
    float aa2 = fractal_de51((vec3(-20.0,28.0,13.0)-p)/10.)*10.;
    float aa3 = fractal_de15((vec3(-15.0,-2.7,3.0)-p)/10.)*10.;
    float aa = fractal_de46((vec3(-5.0,25.0,13.0)-p)/10.)*10.;

    //float final = min(min(min(min(min(min(b2, a),b),c1),c2),c3),aa);
    float final = min(min(aa,aa2),aa3);
    if(final == b2)l = 3.;
    //if(final == c1)c = vec3(0.9,0.1,0.1);
    //if(final == c2)c = vec3(0.1,0.9,0.1);
    if(final == aa||final == aa2){rough
= 1.;cccc=pal(escape*2.,vec3(0.8,0.4,0.4));} if(final == aa3){rough =
0.1;cccc=vec3(0.9);}

    return final;
}
*/
vec3 fold(vec3 p0) {
  vec3 p = p0;
  // if(abs(p.x) > 1.)p.x = 1.0-p.x;
  // if(abs(p.y) > 1.)p.y = 1.0-p.y;
  // if(abs(p.z) > 1.)p.z = 1.0-p.z;
  if (length(p) > 2.)
    return p;
  p = mod(p, 2.) - 1.;

  return p;
}

// float escape;
float DE3(vec3 p0) {
  vec4 p = vec4(p0, 1.);
  escape = 0.;

  for (int i = 0; i < 12; i++) {
    // p.xyz = clamp(p.xyz, vec3(-2.3), vec3(2.3))-p.xyz;
    // p.xyz += sin(float(i+1));
    if (p.x > p.z)
      p.xz = p.zx;
    if (p.z > p.y)
      p.zy = p.yz;
    if (p.y > p.x)
      p.yx = p.xy;
    p = abs(p);
    // p.xyz = fold(p.xyz);
    p.xyz = fold(p.xyz);

    // p.xyz = fract(p.xyz*0.5 - 1.)*2.-1.0;
    p.xyz = mod(p.xyz - 1., 2.) - 1.;
    p *= (1.0 / clamp(dot(p.xyz, p.xyz), 0.1, 1.));

    // p.xyz-=vec3(0.1,0.4,0.2);
    // p*=1.2;
    escape += exp(-0.2 * dot(p.xyz, p.xyz));
  }
  p /= p.w;
  return length(p.xz) * 0.25;
}

/*
float map(vec3 p){
    p = p.xzy;
    cccc = vec3(0.9);
    rough = 1.;
    l = 0.;
    //float a = box(vec3(0.,4.,-5.)-p, vec3(10.,10.,0.2));
    //float b = box(vec3(0.,4.,5.)-p, vec3(10.,10.,0.2));
    //float c1 = box(vec3(7.,4.,0.)-p, vec3(0.2,10.,10.));
    //float c2 = box(vec3(-7.,4.,0.)-p, vec3(0.2,10.,10.));
    //float c3 = box(vec3(0.,9.,0.)-p, vec3(10.,0.2,10.));
    float b2 = box(vec3(0.,-4.5,0.)-p, vec3(8.,0.00000000000004,8.));
    float aa = DE3((vec3(42.,55.,-21.)-p)/5.)*5.;

    //float final = min(min(min(min(min(min(b2, a),b),c1),c2),c3),aa);
    float final = min(aa,b2);
    if(final == b2)l = 13.;
    //if(final == c1)c = vec3(0.9,0.1,0.1);
    //if(final == c2)c = vec3(0.1,0.9,0.1);
    if(final == aa){rough = 0.1;cccc=vec3(0.9);}

    return final;
}*/

#define repid(p, r) (floor((p + r * .5) / r))
#define rep(p, r) (mod(p - r * .5, r) - r * .5)

const float pi2 = pi * 2.;

mat2 rot(float a) {
  float c = cos(a), s = sin(a);
  return mat2(c, s, -s, c);
}

vec2 pmod(vec2 p, float r) {
  float a = pi / r - atan(p.x, p.y);
  float n = pi2 / r;
  a = floor(a / n) * n;
  return p * rot(a);
}

float sdHex(vec3 p, vec2 h, float r) {
  p.zy = p.yz;
  const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
  p = abs(p);
  p.xy -= 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy;
  vec2 d = vec2(length(p.xy - vec2(clamp(p.x, -k.z * h.x, k.z * h.x), h.x)) *
                    sign(p.y - h.x),
                p.z - h.y);
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// http://mercury.sexy/hg_sdf/
float ost(float a, float b, float r, float n) {
  float s = r / n;
  float u = b - r;
  return min(min(a, b), 0.5 * (u + a + abs((mod(u - a + s, 2. * s)) - s)));
}

float sdFloor(vec3 p) {
  vec2 hx = vec2(1.73205081, 1) * 1.04;
  vec3 q = p;
  q.xz = mod(p.xz + hx * 0.5, hx) - hx * 0.5;
  float d = sdHex(q, vec2(0.5, 1.0), 0.04);
  q.xz = mod(p.xz, hx) - hx * 0.5;
  d = min(d, sdHex(q, vec2(0.5, 1.0), 0.04));
  return d;
}

float sdWall(vec3 p) {
  vec3 q = p;

  // wall
  float d = -(abs(p.x) - 8.);

  // square
  q = p;
  q.z = rep(q.z, 240.);
  d = max(d, -sdHex(q, vec2(40., 100.), 0.0));

  // pillars
  vec3 pp = p;
  pp.z = rep(pp.z, 240.);

  q.x = abs(q.x) - 7.;
  q.z = rep(q.z, 20.);
  q = abs(q) - 1.0;
  // clipping pillars in square
  d = min(d, max(max(q.x, q.z), -(abs(pp.z) - 40. + 5.0)));

  return d;
}

float sdSquareObjects(vec3 p, float r, out float id) {
  vec3 q = p;

  q.z = rep(q.z, 240.);
  vec3 pp = q;
  q.xz *= rot(pi / 6.);
  q.xz = pmod(q.xz, 6.);
  q.z -= 15.;

  id = repid(q.z, 15.);
  q.z = rep(q.z, 15.);
  float d = length(q.xz) - r;
  d = max(d, -length(pp.xz) + 15. - r);
  return d;
}

float tts;

float sdSquareFrame(vec3 p) {
  float id;
  float d = sdSquareObjects(p, 1.5 + 1., id);
  d = max(d, -(abs(p.y - 13.) -
               (1.0 - exp(sin(tts * 5. - id * 1.25) * 5.) / exp(5.)) * 5.));
  return d;
}

float sdSquareEmission(vec3 p) {
  float id;
  float d = sdSquareObjects(p, 1.5, id);
  d = max(d, (abs(p.y - 13.) -
              (1.0 - exp(sin(tts * 5. - id * 1.25) * 5.) / exp(5.)) * 5.));
  return d;
}

float sdCeil(vec3 p) { return -(p.y - 30.0); }

vec4 volumeMap0(vec3 p) {
  vec3 q = p;
  q.z = rep(q.z, 30.);
  q.y -= 2.;
  float d = length(q.zy) - .5;
  return vec4(vec3(1., 0.001, .1) * .25, d);
}

vec4 volumeMap1(vec3 p) {
  vec3 q = p;
  q.z = rep(q.z, 30.);
  q.z += 15.;
  q.y -= 20.;
  float d = length(q.zy) - .5;
  return vec4(vec3(.01, 0.01, 1.0) * .4, d);
}

vec4 volumeMap2(vec3 p) {
  vec3 q = p;
  q.z = rep(q.z, 20.);
  q.y -= 12.;
  q.x = abs(q.x) - 6.;
  float d = max(length(q.xy) - .25, abs(q.z) - 3.0);
  return vec4(vec3(.01, 0.01, 1.0) * .3, d);
}

vec4 volumeMap3(vec3 p) {
  float d = sdSquareEmission(p);
  return vec4(vec3(1., 0.01, 0.001) * .2, d);
}
/*
float map(vec3 p) {

    l = 0.;
    rough = 0.1;
    cccc = vec3(0.1);

    float d = sdFloor(p);

    d = ost(d, sdWall(p), 1.0, 5.0);

    d = min(d, sdSquareFrame(p));
    float mm = sdCeil(p);
    d = min(d, mm);


    vec4 a = volumeMap0(p);
    vec4 b = volumeMap1(p);
    vec4 c = volumeMap2(p);
    vec4 ds = volumeMap3(p);

    float ms = min(a.w, min(b.w, min(c.w, ds.w)));
    d = min(d, ms);

    if(d == ms){
        l = 3.;
        if(d == a.w){
            cccc = a.xyz;
        }else if(d == b.w){
            cccc = b.xyz;
        }else if(d == c.w){
            cccc = c.xyz;
        }else if(d == ds.w){
            cccc = ds.xyz;
        }
    }

    return d;
}*/

// https://iquilezles.org/articles/distfunctions
float sdBoxsss(vec3 p, vec3 b) {
  vec3 di = abs(p) - b;
  float mc = max(di.x, max(di.y, di.z));
  return min(mc, length(max(di, 0.0)));
}

float mapz(vec3 p) {
  vec3 w = p;
  vec3 q = p;

  q.xz = mod(q.xz + 1.0, 2.0) - 1.0;

  float d = sdBoxsss(q, vec3(1.0));
  float s = 1.0;
  for (int m = 0; m < 7; m++) {
    float h = float(m) / 6.0;

    p = q.yzx - 0.5 * sin(1.5 * p.x + 6.0 + p.y * 3.0 + float(m) * 5.0 +
                          vec3(1.0, 0.0, 0.0));

    vec3 a = mod(p * s, 2.0) - 1.0;
    s *= 3.0;
    vec3 r = abs(1.0 - 3.0 * abs(a));

    float da = max(r.x, r.y);
    float db = max(r.y, r.z);
    float dc = max(r.z, r.x);
    float c = (min(da, min(db, dc)) - 1.0) / s;
    d = max(c, d);
  }

  return d * 0.5;
}

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * .1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float hash13(vec3 p3) {
  p3 = fract(p3 * .1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float Scalexxxx = 4.;
float MinRad2 = 0.25;

float sr = 4.0;
vec3 fo = vec3(0.7, .9528, .9);
vec3 gh = vec3(.8, .7, 0.5638);
vec3 gw = vec3(.3, 0.5, .2);
vec4 X = vec4(.1, 0.5, 0.1, .3);
vec4 Y = vec4(.1, 0.8, .1, .1);
vec4 Z = vec4(.2, 0.2, .2, .45902);
vec4 R = vec4(0.19, .1, .1, .2);
vec4 orbitTrap = vec4(40000.0);
//--------------------------------------------------------------------------
float DBFold(vec3 p, float fo, float g, float w) {
  if (p.z > p.y)
    p.yz = p.zy;
  float vx = p.x - 2. * fo;
  float vy = p.y - 4. * fo;
  float v = max(abs(vx + fo) - fo, vy);
  float v1 = max(vx - g, p.y - w);
  v = min(v, v1);
  v1 = max(v1, -abs(p.x));
  return min(v, p.x);
}
// the coordinates are pushed/pulled in parallel

// the coordinates are pushed/pulled in parallel
vec3 DBFoldParallel(vec3 p, vec3 fo, vec3 g, vec3 w) {
  vec3 p1 = p;
  p.x = DBFold(p1, fo.x, g.x, w.x);
  p.y = DBFold(p1.yzx, fo.y, g.y, w.y);
  p.z = DBFold(p1.zxy, fo.z, g.z, w.z);
  return p;
}
// serial version
vec3 DBFoldSerial(vec3 p, vec3 fo, vec3 g, vec3 w) {
  p.x = DBFold(p, fo.x, g.x, w.x);
  p.y = DBFold(p.yzx, fo.y, g.y, w.y);
  p.z = DBFold(p.zxy, fo.z, g.z, w.z);
  return p;
}
float Map(vec3 p) {
  vec4 JC = vec4(p, 1.);
  float r2 = dot(p, p);
  float dd = 1.;
  for (int i = 0; i < 6; i++) {

    p = p - clamp(p.xyz, -1.0, 1.0) * 2.0; // mandelbox's box fold

    // Apply pull transformation
    vec3 signs = sign(p); // Save 	the original signs
    p = abs(p);
    p = DBFoldParallel(p, fo, gh, gw);

    p *= signs; // resore signs: this way the mandelbrot set won't extend in
                // negative directions

    // Sphere fold
    r2 = dot(p, p);
    float t = clamp(1. / r2, 1., 1. / MinRad2);
    p *= t;
    dd *= t;

    // Scale and shift
    p = p * Scalexxxx + JC.xyz;
    dd = dd * Scalexxxx + JC.w;
    p = vec3(1.0, 1.0, .92) * p;

    r2 = dot(p, p);
    orbitTrap = min(orbitTrap, abs(vec4(p.x, p.y, p.z, r2)));
  }
  dd = abs(dd);
#if 1
  return (sqrt(r2) - sr) / dd; // bounding volume is a sphere
#else
  p = abs(p);
  return (max(p.x, max(p.y, p.z)) - sr) / dd; // bounding volume is a cube
#endif
}

#define SceneRadiusxxx 2.
#define StepFactor 1.

struct Surface {
  bool metallic;
  float ior, roughness;
  vec3 emission, diffuse;
};

mat3 rotationMatrixbbbb(vec3 rotEuler) {
  float c = cos(rotEuler.x), s = sin(rotEuler.x);
  mat3 rx = mat3(1, 0, 0, 0, c, -s, 0, s, c);
  c = cos(rotEuler.y), s = sin(rotEuler.y);
  mat3 ry = mat3(c, 0, -s, 0, 1, 0, s, 0, c);
  c = cos(rotEuler.z), s = sin(rotEuler.z);
  mat3 rz = mat3(c, -s, 0, s, c, 0, 0, 0, 1);

  return rz * rx * ry;
}

void foldmmm(inout vec3 z, vec3 o, vec3 n) {
  z -= 2. * n * min(dot(z - o, n), 0.);
}
#define Pi 3.14159265359

float sdfIFS(vec3 z, out Surface surf) {

  float scale = 2.;
  int Iterations = 16;
  mat3 rot = rotationMatrixbbbb(vec3(.5) * Pi);

  vec3 n1 = normalize(vec3(1., 1., -1.));
  vec3 n2 = normalize(vec3(1., -1., -1.));
  // vec3 n3 = normalize(vec3(1., -1., 1.));

  vec3 ot = vec3(1.);
  for (int i = 0; i < Iterations; i++) {
    foldmmm(z, vec3(-.0), n1);
    foldmmm(z, vec3(-.25), n2);
    // fold(z, vec3(-.25), n3);
    n1 *= rot;
    z = z * scale - sign(z) * (scale - 1.0);
    ot = min(abs(z), ot);
  }
  surf = Surface(true, 1.0, .1, vec3(0.), vec3(.8));
  if (ot.r >= .75)
    surf = Surface(false, 1., .0, ot.ggb * 30. * vec3(12., 2., .5), vec3(0.1));
  return length(z) * pow(scale, float(-Iterations));
}

float sdf(in vec3 pos, out Surface surf) {
  float sSc = length(pos) - SceneRadiusxxx;
  float s = sdfIFS(pos * rotationMatrixbbbb(vec3(.5, .0, .0) * Pi), surf);
  return abs(sSc) > abs(s) || s > 0. ? s : sSc;
}
/*
float sdf(vec3 pos){
    Surface surf;
    return sdf(pos, surf);
}*/
/*

struct Surface {
    bool metallic;
    float ior, roughness;
    vec3 emission, diffuse;
};

*/
/*
float map(vec3 p) {
    //p.y += 25.;
    l = 0.;
    rough = 1.0;
    cccc = vec3(0.9);

    Surface surf;
    float a = sdf(p/25., surf)*25.;
    if(a < 0.1){
        cccc = surf.diffuse;
        rough = surf.roughness;
        if(surf.emission.x > 0.0 || surf.emission.y > 0. || surf.emission.z >
0.){
           // l = 1.;
            cccc = surf.emission;
        }
    }
    return a*0.5;
    //Surface surf;
    //return sdf(pos, surf);
    //float dist =
}*/

float ph(float h, float H) { return exp(-(abs(h) / H)); }
float PMz(float cost, float g) {
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
    float PMMM = PMz(max(dot(d, lig), 0.), 0.76);
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
            mix(vec3(0.9, 0.4, 0.6), vec3(0.9),
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

vec3 F(float costheta, float y, vec3 A, vec3 B, vec3 C, vec3 D, vec3 E) {
  return (1. + A * exp(B / costheta)) *
         (1.0 + C * exp(D * y) + E * cos(y) * cos(y));
}

vec3 skyp2(vec3 d, vec3 lig) { // my code to begin with
  vec3 zenith = vec3(0., 0., 1.);
  float costheta = max(dot(d, zenith), 0.);
  float costhetas = max(dot(lig, zenith), 0.);
  float cosy = max(dot(lig, d), 0.);
  float y = acos(cosy);
  // return vec3(0.1);
  // if(costheta<0.01)return vec3(0.);
  // simple cie sky
  float T = 3.;
  float X = (4. / 9. - T / 120.) * (3.14159 - 2. * acos(costhetas));
  float Yz = (4.0453 * T - 4.9710) * tan(X) - 0.2155 * T + 2.4192;

  // vec3 template = vec3(*T+,*T+,*T+);
  vec3 AYxy =
      vec3(0.1787 * T - 1.4630, -0.0193 * T - 0.2592, -0.0167 * T - 0.2608);
  vec3 BYxy =
      vec3(-0.3554 * T + 0.4275, -0.0665 * T + 0.0008, -0.0950 * T + 0.0092);
  vec3 CYxy =
      vec3(-0.0227 * T + 5.3251, -0.0004 * T + 0.2125, -0.0079 * T + 0.2102);
  vec3 DYxy =
      vec3(0.1206 * T - 2.5771, -0.0641 * T - 0.8989, -0.0441 * T - 1.6537);
  vec3 EYxy =
      vec3(-0.0670 * T + 0.3703, -0.0033 * T + 0.0452, -0.0109 * T + 0.0529);

  float ts = acos(costhetas);
  float ts2 = ts * ts;
  float ts3 = ts * ts * ts;
  vec3 xz0 = vec3(0.00166 * ts3 - 0.00375 * ts2 + 0.00209 * ts,
                  -0.02903 * ts3 + 0.06377 * ts2 - 0.03202 * ts + 0.00394,
                  0.11693 * ts3 - 0.21196 * ts2 + 0.06052 * ts + 0.25886);

  vec3 yz0 = vec3(0.00275 * ts3 - 0.00610 * ts2 + 0.00317 * ts,
                  -0.04214 * ts3 + 0.08970 * ts2 - 0.04153 * ts + 0.00516,
                  0.15346 * ts3 - 0.26756 * ts2 + 0.06670 * ts + 0.26688);

  float xz = xz0.x * T * T + xz0.y * T + xz0.z;
  float yz = yz0.x * T * T + yz0.y * T + yz0.z;

  vec3 Yxyz = vec3(Yz, xz, yz);
  // test
  // vec3 test1 = F(costheta, y, AYxy, BYxy, CYxy, DYxy, EYxy);
  vec3 Ftop = F(costheta, y, AYxy, BYxy, CYxy, DYxy, EYxy);
  vec3 Fbottom = F(1., ts, AYxy, BYxy, CYxy, DYxy, EYxy);

  vec3 finalYxy = Yxyz * (Ftop / Fbottom);

  vec3 XYZ = vec3((finalYxy.y * finalYxy.x) / finalYxy.z, finalYxy.x,
                  ((1. - finalYxy.y - finalYxy.z) * finalYxy.x) / finalYxy.z);

  vec3 rgb = vec3(3.2404542 * XYZ.x - 1.5371385 * XYZ.y - 0.4985314 * XYZ.z,
                  -0.9692660 * XYZ.x + 1.8760108 * XYZ.y + 0.0415560 * XYZ.z,
                  0.0556434 * XYZ.x - 0.2040259 * XYZ.y + 1.0572252 * XYZ.z);

  // return test1*0.1;
  return rgb * 0.034 + exp(-y * 20.) * vec3(0.9, 0.6, 0.2);
}

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

float ggx_D(vec3 m, vec3 n, float a) {
  float top = a * a;
  float bottom =
      3.14159 *
      pow((a * a - 1.) * (max(dot(m, n), 0.) * max(dot(m, n), 0.)) + 1., 2.);
  return top / bottom;
}

float ggx_pdf(vec3 m, vec3 n, float a) {
  float top = a * a * max(dot(m, n), 0.);
  float bottom =
      3.14159 *
      pow((a * a - 1.) * max(dot(m, n), 0.) * max(dot(m, n), 0.) + 1., 2.);
  return top / bottom;
}

float ggx_pdf2(vec3 m, vec3 n, float a) {
  float ang = sin(acos(dot(m, n)));
  float top = a * a * max(dot(m, n), 0.) * ang;
  float bottom =
      3.14159 *
      pow((a * a - 1.) * max(dot(m, n), 0.) * max(dot(m, n), 0.) + 1., 2.);
  return top / bottom;
}

float ggx_G(vec3 h, vec3 n, vec3 wi, vec3 l, float a) {
  float g1 =
      (2. * max(dot(n, h), 0.) * max(dot(n, -wi), 0.)) / max(dot(-wi, h), 0.);
  float g2 =
      (2. * max(dot(n, h), 0.) * max(dot(n, l), 0.)) / max(dot(-wi, h), 0.);
  float G = min(1., min(g1, g2));
  return G;
}

float ggx_G2(vec3 h, vec3 n, vec3 wi, vec3 l, float a) {
  float top = 2. * max(dot(n, -wi), 0.);
  float bottom = max(dot(n, -wi), 0.) +
                 sqrt(a * a + (1. - a * a) * pow(max(dot(n, -wi), 0.), 2.));
  return top / bottom;
}

vec3 ggx_F(vec3 Fo, float cost) { return Fo + (1. - Fo) * pow(1. - cost, 5.); }
// energy conserving
// vec3 ggx_F(vec3 Fo, vec3 v, vec3 n, vec3 l){
// return Fo + (1.-Fo)*pow(1.-cost,5.);
//}

vec3 fresnelSchlick(vec3 F0, vec3 F90, float LdotH) {
  return F0 + (max(F0, F90) - F0) * pow(1. - LdotH, 5.);
}

vec3 ggx_S(vec3 n, inout uint r, float a) {
  float r1 = rndf(r);
  float r2 = rndf(r);

  float theta = atan(a * sqrt(r1 / (1. - r1)));
  // float theta = acos(sqrt((1.-r1)/(r1*(a*a-1.)+1.) ));
  float phi = 2. * 3.14159 * r2;

  float x = cos(phi) * sin(theta);
  float y = sin(phi) * sin(theta);
  float z = cos(theta);

  vec3 W = (abs(n.x) > 0.99) ? vec3(0., 1., 0.) : vec3(1., 0., 0.);
  vec3 N = n;
  vec3 T = normalize(cross(N, W));
  vec3 B = cross(T, N);
  return normalize(x * T + y * B + z * N);
}

vec3 sampleSun(vec3 n, inout uint r, float theta) {
  float diff = 1. - cos(theta);
  float z = cos(theta) + rndf(r) * diff;
  float angle = rndf(r) * 3.14159 * 2.;
  float radius = sqrt(1. - z * z);
  float x = cos(angle) * radius;
  float y = sin(angle) * radius;

  vec3 W = (abs(n.x) > 0.99) ? vec3(0., 1., 0.) : vec3(1., 0., 0.);
  vec3 N = n;
  vec3 T = normalize(cross(N, W));
  vec3 B = cross(T, N);
  return normalize(x * T + y * B + z * N);
}