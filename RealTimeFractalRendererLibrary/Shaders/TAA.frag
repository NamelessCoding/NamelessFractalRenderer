#version 420
// colortexFogPrev

layout(binding = 0, Rgba16f) uniform image3D rcpos;

layout(location = 0) out vec4 TAA;
layout(location = 1) out vec4 prevPosition;
layout(location = 2) out vec4 prevSecondPosition;
layout(location = 3) out vec4 colortexFogPrev;

// prevSecondPosition
in vec2 texCoord;

uniform sampler2D color;
uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D secondpos;
uniform sampler2D weigth;
uniform sampler2D outgoingr;
uniform sampler2D weightS;
uniform sampler2D outgoingrS;
uniform sampler2D Acc;
uniform sampler2D den1;
uniform sampler2D var1;
uniform sampler2D prevTAA;
uniform sampler2D reflAlb;
uniform sampler2D inf;
uniform sampler2D colorfog;
uniform sampler2D reflnorm;
uniform sampler2D colortexFog;
// colortexFogP
uniform sampler2D colortexFogP;
uniform sampler2D spatfog;
uniform sampler2D spatfogLO;
uniform sampler2D holdinfo;
uniform sampler2D sunTex;
uniform sampler2D skyTex;
uniform sampler2D watpos;
uniform sampler2D watnorm;
uniform sampler3D worl;
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
///////////////////////////

float box(vec3 p, vec3 c) {
  vec3 a = abs(p) - c;
  return max(a.x, max(a.y, a.z));
}

vec3 pal(float t, vec3 a) { return 0.5 + 0.5 * cos(2. * 3.14159 * t + a); }

float escape = 0.;
float DE2(vec3 p0) {
  p0 /= 2.;
  escape = 0.;
  vec4 p = vec4(p0, 1.);
  for (int i = 0; i < 8; i++) {
    p.xyz = mod(p.xyz - 1., 2.) - 1.;
    p *= (1.0 / dot(p.xyz, p.xyz));
    escape += exp(-0.2 * dot(p.xyz, p.xyz));
  }
  return length(p.xz / p.w) * 0.25;
}

float l = 0.;
vec3 cccc = vec3(0.);
float map(vec3 p) {
  l = 0.;
  cccc = vec3(1.);
  // return box(mod(p, 12.)-6., vec3(1.));
  float a = DE2(p);
  cccc = pal(escape, vec3(0.4, 0.6, 0.9));
  return a;
}

vec3 norm(vec3 p) {
  return normalize(
      vec3(map(vec3(p.x + 0.01, p.yz)) - map(vec3(p.x - 0.01, p.yz)),
           map(vec3(p.x, p.y + 0.01, p.z)) - map(vec3(p.x, p.y - 0.01, p.z)),
           map(vec3(p.x, p.y, p.z + 0.01)) - map(vec3(p.x, p.y, p.z - 0.01))));
}

bool trace(inout vec3 p, vec3 d) {
  for (int i = 0; i < 60; i++) {
    float dist = map(p);
    if (dist < 0.01) {
      return true;
    }
    p += d * dist;
  }
  return false;
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
    if (texture2D(albedo, newCords).w > 0.5) {
      continue;
      ;
    }
    float L_W = L(length(coords) / 1., 2.0);
    col += texture2D(den1, newCords).xyz * L_W;
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
  for (int i = 0; i < 20; i++) {
    float m = 1.;
    float r = 2. * 3.14159 * k * float(i);
    vec2 coords = vec2(m * cos(r), m * sin(r)) * dist;
    // vec4 c2 = texture2D(iChannel0, (p+coords)/iResolution.xy).xyzw;
    vec2 cir = (p + coords) / iResolution.xy;
    vec3 ccc = texture2D(albedo, cir).xyz;

    vec3 c = upscaleIndirect(iResolution, cir) * ccc;
    if (texture2D(albedo, cir).w > 0.5) {
      c = ccc;
    }

    // c = c*c *1.8;
    // vec3 bok = pow(c,vec3(4.));
    vec3 bok = vec3(1.);
    s += c * bok;
    div += bok;
  }

  s /= div;

  return s;
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

vec3 blurShadows(vec2 p) {
  vec3 c = vec3(0.);
  vec2 iResolution = wh;
  for (int i = 0; i < 9; i++) {
    vec2 offset = vec2(float(i % 3), float(i / 3)) - 1.;
    vec2 newCords = (p * iResolution + offset) / iResolution;
    c += texture2D(reflAlb, newCords).xyz * vec3(0.9, 0.8, 0.7) * 5.5 *
         (texture2D(albedo, newCords).xyz / 3.14159);
  }
  return c / 9.;
}

vec3 FsrEasuCF(vec2 p) {
  // p/=wh*2.;
  vec3 col = texture2D(den1, p).xyz * texture2D(albedo, p).xyz;
  // texture2D(albedo, TC*2.).w > 0.5 ||
  col += texture2D(reflAlb, p).xyz * vec3(0.9, 0.8, 0.6) * 4.5 *
         (texture2D(albedo, p).xyz / 3.14159);

  return col;
}

/**** EASU ****/
void FsrEasuCon(
    out vec4 con0, out vec4 con1, out vec4 con2, out vec4 con3,
    // This the rendered image resolution being upscaled
    vec2 inputViewportInPixels,
    // This is the resolution of the resource containing the input image (useful
    // for dynamic resolution)
    vec2 inputSizeInPixels,
    // This is the display resolution which the input image gets upscaled to
    vec2 outputSizeInPixels) {
  // Output integer position to a pixel position in viewport.
  con0 = vec4(inputViewportInPixels.x / outputSizeInPixels.x,
              inputViewportInPixels.y / outputSizeInPixels.y,
              .5 * inputViewportInPixels.x / outputSizeInPixels.x - .5,
              .5 * inputViewportInPixels.y / outputSizeInPixels.y - .5);
  // Viewport pixel position to normalized image space.
  // This is used to get upper-left of 'F' tap.
  con1 = vec4(1, 1, 1, -1) / inputSizeInPixels.xyxy;
  // Centers of gather4, first offset from upper-left of 'F'.
  //      +---+---+
  //      |   |   |
  //      +--(0)--+
  //      | b | c |
  //  +---F---+---+---+
  //  | e | f | g | h |
  //  +--(1)--+--(2)--+
  //  | i | j | k | l |
  //  +---+---+---+---+
  //      | n | o |
  //      +--(3)--+
  //      |   |   |
  //      +---+---+
  // These are from (0) instead of 'F'.
  con2 = vec4(-1, 2, 1, 2) / inputSizeInPixels.xyxy;
  con3 = vec4(0, 4, 0, 0) / inputSizeInPixels.xyxy;
}

// Filtering for a given tap for the scalar.
void FsrEasuTapF(inout vec3 aC,  // Accumulated color, with negative lobe.
                 inout float aW, // Accumulated weight.
                 vec2 off,       // Pixel offset from resolve position to tap.
                 vec2 dir,       // Gradient direction.
                 vec2 len,       // Length.
                 float lob,      // Negative lobe strength.
                 float clp,      // Clipping point.
                 vec3 c) {
  // Tap color.
  // Rotate offset by direction.
  vec2 v = vec2(dot(off, dir), dot(off, vec2(-dir.y, dir.x)));
  // Anisotropy.
  v *= len;
  // Compute distance^2.
  float d2 = min(dot(v, v), clp);
  // Limit to the window as at corner, 2 taps can easily be outside.
  // Approximation of lancos2 without sin() or rcp(), or sqrt() to get x.
  //  (25/16 * (2/5 * x^2 - 1)^2 - (25/16 - 1)) * (1/4 * x^2 - 1)^2
  //  |_______________________________________|   |_______________|
  //                   base                             window
  // The general form of the 'base' is,
  //  (a*(b*x^2-1)^2-(a-1))
  // Where 'a=1/(2*b-b^2)' and 'b' moves around the negative lobe.
  float wB = .4 * d2 - 1.;
  float wA = lob * d2 - 1.;
  wB *= wB;
  wA *= wA;
  wB = 1.5625 * wB - .5625;
  float w = wB * wA;
  // Do weighted average.
  aC += c * w;
  aW += w;
}

//------------------------------------------------------------------------------------------------------------------------------
// Accumulate direction and length.
void FsrEasuSetF(inout vec2 dir, inout float len, float w, float lA, float lB,
                 float lC, float lD, float lE) {
  // Direction is the '+' diff.
  //    a
  //  b c d
  //    e
  // Then takes magnitude from abs average of both sides of 'c'.
  // Length converts gradient reversal to 0, smoothly to non-reversal at 1,
  // shaped, then adding horz and vert terms.
  float lenX = max(abs(lD - lC), abs(lC - lB));
  float dirX = lD - lB;
  dir.x += dirX * w;
  lenX = clamp(abs(dirX) / lenX, 0., 1.);
  lenX *= lenX;
  len += lenX * w;
  // Repeat for the y axis.
  float lenY = max(abs(lE - lC), abs(lC - lA));
  float dirY = lE - lA;
  dir.y += dirY * w;
  lenY = clamp(abs(dirY) / lenY, 0., 1.);
  lenY *= lenY;
  len += lenY * w;
}

//------------------------------------------------------------------------------------------------------------------------------
void FsrEasuF(
    out vec3 pix,
    vec2 ip, // Integer pixel position in output.
    // Constants generated by FsrEasuCon().
    vec4 con0, // xy = output to input scale, zw = first pixel offset correction
    vec4 con1, vec4 con2, vec4 con3) {
  //------------------------------------------------------------------------------------------------------------------------------
  // Get position of 'f'.
  vec2 pp = ip * con0.xy + con0.zw; // Corresponding input pixel/subpixel
  vec2 fp = floor(pp);              // fp = source nearest pixel
  pp -= fp;                         // pp = source subpixel

  //------------------------------------------------------------------------------------------------------------------------------
  // 12-tap kernel.
  //    b c
  //  e f g h
  //  i j k l
  //    n o
  // Gather 4 ordering.
  //  a b
  //  r g
  vec2 p0 = fp * con1.xy + con1.zw;

  // These are from p0 to avoid pulling two constants on pre-Navi hardware.
  vec2 p1 = p0 + con2.xy;
  vec2 p2 = p0 + con2.zw;
  vec2 p3 = p0 + con3.xy;

  // TextureGather is not available on WebGL2
  vec4 off = vec4(-.5, .5, -.5, .5) * con1.xxyy;
  // textureGather to texture offsets
  // x=west y=east z=north w=south
  vec3 bC = FsrEasuCF(p0 + off.xw);
  float bL = bC.g + 0.5 * (bC.r + bC.b);
  vec3 cC = FsrEasuCF(p0 + off.yw);
  float cL = cC.g + 0.5 * (cC.r + cC.b);
  vec3 iC = FsrEasuCF(p1 + off.xw);
  float iL = iC.g + 0.5 * (iC.r + iC.b);
  vec3 jC = FsrEasuCF(p1 + off.yw);
  float jL = jC.g + 0.5 * (jC.r + jC.b);
  vec3 fC = FsrEasuCF(p1 + off.yz);
  float fL = fC.g + 0.5 * (fC.r + fC.b);
  vec3 eC = FsrEasuCF(p1 + off.xz);
  float eL = eC.g + 0.5 * (eC.r + eC.b);
  vec3 kC = FsrEasuCF(p2 + off.xw);
  float kL = kC.g + 0.5 * (kC.r + kC.b);
  vec3 lC = FsrEasuCF(p2 + off.yw);
  float lL = lC.g + 0.5 * (lC.r + lC.b);
  vec3 hC = FsrEasuCF(p2 + off.yz);
  float hL = hC.g + 0.5 * (hC.r + hC.b);
  vec3 gC = FsrEasuCF(p2 + off.xz);
  float gL = gC.g + 0.5 * (gC.r + gC.b);
  vec3 oC = FsrEasuCF(p3 + off.yz);
  float oL = oC.g + 0.5 * (oC.r + oC.b);
  vec3 nC = FsrEasuCF(p3 + off.xz);
  float nL = nC.g + 0.5 * (nC.r + nC.b);

  //------------------------------------------------------------------------------------------------------------------------------
  // Simplest multi-channel approximate luma possible (luma times 2, in 2
  // FMA/MAD). Accumulate for bilinear interpolation.
  vec2 dir = vec2(0);
  float len = 0.;

  FsrEasuSetF(dir, len, (1. - pp.x) * (1. - pp.y), bL, eL, fL, gL, jL);
  FsrEasuSetF(dir, len, pp.x * (1. - pp.y), cL, fL, gL, hL, kL);
  FsrEasuSetF(dir, len, (1. - pp.x) * pp.y, fL, iL, jL, kL, nL);
  FsrEasuSetF(dir, len, pp.x * pp.y, gL, jL, kL, lL, oL);

  //------------------------------------------------------------------------------------------------------------------------------
  // Normalize with approximation, and cleanup close to zero.
  vec2 dir2 = dir * dir;
  float dirR = dir2.x + dir2.y;
  bool zro = dirR < (1.0 / 32768.0);
  dirR = inversesqrt(dirR);
  dirR = zro ? 1.0 : dirR;
  dir.x = zro ? 1.0 : dir.x;
  dir *= vec2(dirR);
  // Transform from {0 to 2} to {0 to 1} range, and shape with square.
  len = len * 0.5;
  len *= len;
  // Stretch kernel {1.0 vert|horz, to sqrt(2.0) on diagonal}.
  float stretch = dot(dir, dir) / (max(abs(dir.x), abs(dir.y)));
  // Anisotropic length after rotation,
  //  x := 1.0 lerp to 'stretch' on edges
  //  y := 1.0 lerp to 2x on edges
  vec2 len2 = vec2(1. + (stretch - 1.0) * len, 1. - .5 * len);
  // Based on the amount of 'edge',
  // the window shifts from +/-{sqrt(2.0) to slightly beyond 2.0}.
  float lob = .5 - .29 * len;
  // Set distance^2 clipping point to the end of the adjustable window.
  float clp = 1. / lob;

  //------------------------------------------------------------------------------------------------------------------------------
  // Accumulation mixed with min/max of 4 nearest.
  //    b c
  //  e f g h
  //  i j k l
  //    n o
  vec3 min4 = min(min(fC, gC), min(jC, kC));
  vec3 max4 = max(max(fC, gC), max(jC, kC));
  // Accumulation.
  vec3 aC = vec3(0);
  float aW = 0.;
  FsrEasuTapF(aC, aW, vec2(0, -1) - pp, dir, len2, lob, clp, bC);
  FsrEasuTapF(aC, aW, vec2(1, -1) - pp, dir, len2, lob, clp, cC);
  FsrEasuTapF(aC, aW, vec2(-1, 1) - pp, dir, len2, lob, clp, iC);
  FsrEasuTapF(aC, aW, vec2(0, 1) - pp, dir, len2, lob, clp, jC);
  FsrEasuTapF(aC, aW, vec2(0, 0) - pp, dir, len2, lob, clp, fC);
  FsrEasuTapF(aC, aW, vec2(-1, 0) - pp, dir, len2, lob, clp, eC);
  FsrEasuTapF(aC, aW, vec2(1, 1) - pp, dir, len2, lob, clp, kC);
  FsrEasuTapF(aC, aW, vec2(2, 1) - pp, dir, len2, lob, clp, lC);
  FsrEasuTapF(aC, aW, vec2(2, 0) - pp, dir, len2, lob, clp, hC);
  FsrEasuTapF(aC, aW, vec2(1, 0) - pp, dir, len2, lob, clp, gC);
  FsrEasuTapF(aC, aW, vec2(1, 2) - pp, dir, len2, lob, clp, oC);
  FsrEasuTapF(aC, aW, vec2(0, 2) - pp, dir, len2, lob, clp, nC);
  //------------------------------------------------------------------------------------------------------------------------------
  // Normalize and dering.
  pix = min(max4, max(min4, aC / aW));
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

vec3 median(vec2 uv, vec2 iResolution, float offsetMult) {
  // vec3 arrayCol[9];

  vec3 colArray[9] = vec3[](vec3(0.), vec3(0.), vec3(0.), vec3(0.), vec3(0.),
                            vec3(0.), vec3(0.), vec3(0.), vec3(0.));

  for (int i = 0; i < 9; i++) {
    vec2 offset = vec2(float(i % 3) - 1., float(i / 3) - 1.) * offsetMult;
    vec2 coords = (uv * iResolution.xy + offset);
    // vec3 currCol = texelFetch(tex, ivec2(coords*0.5),0).rgb;
    // vec3 currCol = texture2D(tex, (coords/iResolution)*0.5).rgb;
    vec3 currCol = texture2D(den1, coords / iResolution).xyz;
    colArray[i] = currCol;
  }

  vec3 first = pickbetween3(colArray[0], colArray[1], colArray[2]);
  vec3 second = pickbetween3(colArray[3], colArray[4], colArray[5]);
  vec3 third = pickbetween3(colArray[6], colArray[7], colArray[8]);

  return pickbetween3(first, second, third);
}

vec3 median2(vec2 uv, vec2 iResolution, float offsetMult) {
  // vec3 arrayCol[9];

  vec3 colArray[9] = vec3[](vec3(0.), vec3(0.), vec3(0.), vec3(0.), vec3(0.),
                            vec3(0.), vec3(0.), vec3(0.), vec3(0.));

  for (int i = 0; i < 9; i++) {
    vec2 offset = vec2(float(i % 3) - 1., float(i / 3) - 1.) * offsetMult;
    vec2 coords = (uv * iResolution.xy + offset);
    // vec3 currCol = texelFetch(tex, ivec2(coords*0.5),0).rgb;
    // vec3 currCol = texture2D(tex, (coords/iResolution)*0.5).rgb;
    vec3 currCol = texture2D(colortexFog, coords / iResolution).xyz;
    colArray[i] = currCol;
  }

  vec3 first = pickbetween3(colArray[0], colArray[1], colArray[2]);
  vec3 second = pickbetween3(colArray[3], colArray[4], colArray[5]);
  vec3 third = pickbetween3(colArray[6], colArray[7], colArray[8]);

  return pickbetween3(first, second, third);
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

vec4 fog2(vec3 pos, vec3 dir, vec2 TC, inout uint r) {
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
    // accum += shadowS(finPos);

    float accum2 = length(cam - finPos);
    float heigh = exp(-max(reversePos.y - (mov + 1.), 0.) * .4);
    accum += stepCol * volAbs * .004 * dens * 1.5 * exp(-acci * 5.);

    accum += stepCol * volAbs * .0005 * dens * 1. * heigh;

    // volAbs *= exp(-length(finPos - cam)*.025);
    volAbs *= exp(-max(dens, 0.05) * heigh * length(finPos - cam) * .01);

    absorb *= 1.0 - exp(-0.7 * (1.0 - max(dot(ldir, vec3(0., 1., 0.)), 0.)) *
                        length(finPos - cam));
  }

  return vec4(accum, volAbs);
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
  }

  // if(TC.x < 0.5){
  // col = sunN;

  //}
  // col = texture2D(colorfog, TC).xyz;
  // col = median3(TC, iResolution, 1.);
  // col = median4(TC, iResolution, 1.);
  // if(TC.x > 0.499 && TC.x < 0.501){
  //    col = vec3(0.9, 0., 0.);
  //}
  // col = 0.5*median2(TC, iResolution, 1.)*F2;

  // col += texture2D(holdinfo, TC).xyz;
  // col += texture2D(holdinfo, TC).x * vec3(0.5,0.9,0.5)/220.;
  // texture2D(albedo, TC*2.).w > 0.5 ||
  // col += texture2D(reflAlb, TC).xyz*vec3(0.9,0.8,0.6)*5.5*(texture2D(albedo,
  // TC).xyz/3.14159); col += texture2D(reflnorm, TC).www*vec3(0.9,0.7,0.2)*0.4;
  // texture2D(albedo, TC).w > 0.5 ||

  if ((texture2D(albedo, TC - smallOffset2).w > 0.5 ||
       texture2D(position, TC - smallOffset2).w > 0.5) &&
      wp.w < 0.5) {
    col = texture2D(albedo, TC - smallOffset2).xyz;
  }
  ind = col;

  // vec4 fog(vec3 pos, vec3 dir, vec2 TC, inout uint r){

  vec4 fogs = fog(watPOS, wi, TC, r);
  col = col * fogs.w + pow(fogs.xyz * max(ldir.y * ldir.y, 0.5), vec3(1.));
  // col = fogs.xyz;
  // col = diff;
  // col = vec3(1.,0.,0.);

  // ind = col;
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
  // col = texture2D(den1, TC).xyz;

  // col += texture2D(inf, TC).w * vec3(0.9, 0.6, 0.6)/70.;
  //  vec3 c;
  //     vec4 con0,con1,con2,con3;

  //     // "rendersize" refers to size of source image before upscaling.
  //     vec2 rendersize = wh*2.;
  //     FsrEasuCon(
  //         con0, con1, con2, con3, rendersize, rendersize, iResolution.xy
  //     );
  //     FsrEasuF(c, TC*wh*2., con0, con1, con2, con3);
  // vec3 col = c;
  float RWf = texture2D(spatfog, TC).z;
  // col += clamp(texture2D(spatfogLO, TC).xyz, 0., 10000.)*clamp(RWf, 0.,
  // 200.)*0.4;

  float depth = texture2D(normal, TC).w;

  // col += 1./(1.+exp(-2.*(depth*0.1-2.)))*vec3(0.2,0.6,0.9);

  /*
     vec3 prevCol = (texture2D(prevTAA, ProjectedCoordinates).xyz);
  //vec4 currCol = adjust(col);

  vec3 minCol = (vec3(9999.));
  vec3 maxCol = (vec3(-9999.));

    for(int i = 0; i < 9; i++){
      vec2 coords = vec2(float(i%3)-1., float(i/3)-1.);
      vec3 currSample = (texture2D(prevTAA, (ProjectedCoordinates*iResolution +
  coords)/iResolution).xyz); minCol = min(minCol, currSample); maxCol =
  max(maxCol, currSample);
    }

  prevCol = ClipAABB(prevCol, minCol, maxCol);
    float currentVelocity = length(ProjectedCoordinates - TC);
    float prevVelocity = (texture2D(prevTAA, ProjectedCoordinates).w);
    float velocityWeigth = sqrt(prevVelocity*prevVelocity +
  currentVelocity*currentVelocity); float disocclusion = clamp((velocityWeigth -
  0.001)*1., 0., 1.);
  //if(ProjectedCoordinates.x >= 0. && ProjectedCoordinates.x <= 1. &&
  ProjectedCoordinates.y >= 0. && ProjectedCoordinates.y <= 1.0){

  vec2 velocity = (TC - ProjectedCoordinates.xy) * iResolution;
          float blendFactor = float(
                  ProjectedCoordinates.x > 0.0 && ProjectedCoordinates.x < 1.0
  && ProjectedCoordinates.y > 0.0 && ProjectedCoordinates.y < 1.0
          );
          blendFactor *= exp(-length(velocity)*0.1) * 0.6 + 0.3;


    col = mix(col, prevCol, blendFactor);
  */

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
  if (texture2D(albedo, TC - smallOffset2).w < 0.5 &&
      texture2D(position, TC - smallOffset2).w < 0.5) {
    col = mix(col, prevCol, clamp(blendFactor, 0.0, 1.));
  }
  //}

  //}

  TAA = vec4(col, clamp(sqrt(lum(ind)) * 0.1 +
                            texture2D(prevTAA, ProjectedCoordinates).w * 0.9,
                        0., 0.99));

  float frame = texture2D(colortexFogP, TC).w;
  col = texture2D(color, TC).xyz + (texture2D(colortexFogP, TC).xyz);
  if (length(viewPos - lastViewPos) > 0.01) {
    col = texture2D(colortexFog, TC).xyz;
    frame = 0.;
  }

  colortexFogPrev = vec4(col, frame + 1.0);

  prevPosition = texture2D(position, TC);
  prevSecondPosition = texture2D(secondpos, TC);
}