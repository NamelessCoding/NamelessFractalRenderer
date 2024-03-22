

layout(location = 0) out vec4 position;
layout(location = 1) out vec4 normal;
layout(location = 2) out vec4 albedo;
layout(location = 3) out vec4 holdinfo;
// layout (location = 4) out vec4 reflectionAlb;
layout(location = 4) out vec4 colorfog;
layout(location = 5) out vec4 watpos;
layout(location = 6) out vec4 watnorm;

// out vec4 outputColor;
in vec2 texCoord;

// uniform sampler2D texture0;

layout(binding = 0, Rgba16f) uniform image3D rcpos;
layout(binding = 1, Rgba16f) uniform image3D rcnorm;
uniform sampler2D skytex;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 invview;
uniform mat4 invproj;
uniform vec2 wh;
uniform float time;
uniform float time2;

uniform vec3 viewPos;
uniform vec3 lastViewPos;
uniform vec3 ldir;

const float phi2 = 1.32471795724474602596;
const vec2 a = vec2(1.0 / phi2, 1.0 / (phi2 * phi2));

vec2 R2(float n) { return fract(a * n + 0.5); }
#define RESOLUTION 0.5

vec2 jitter() {
  return ((R2(float(int(time) % 1000)) * 2. - 1.) / (wh * 2. * RESOLUTION));
}

void main() {
  vec2 fragCoord = texCoord * wh;
  uint r = uint(uint(fragCoord.x) * uint(1973) +
                uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) |
           uint(1);
  vec2 iResolution = wh;
  uint seedCam = uint(time);
  vec2 smallOffset =
      ((vec2(rndf(seedCam), rndf(seedCam))) * 2. - 1.) / iResolution;
  //  smallOffset = vec2(0.);
  //+jitter()
  vec4 p22 =
      vec4((clamp(texCoord + smallOffset, 0., 1.) * 2.0 - 1.0), 0.0, 1.0);
  // p22.xy += smallOffset*1.;

  vec3 dir = (invproj * p22).xyz / (invproj * p22).w;
  dir = normalize(mat3(invview) * dir);

  vec4 p22cam =
      vec4((clamp(vec2(0.5) + smallOffset, 0., 1.) * 2.0 - 1.0), 0.0, 1.0);
  // p22.xy += smallOffset*1.;

  vec3 dircam = (invproj * p22cam).xyz / (invproj * p22cam).w;
  dircam = normalize(mat3(invview) * dircam);

  vec3 wi = dir;
  // dir = dir.xzy;
  vec3 pos =
      viewPos; // +
               // (vec3(rndf(seedCam),rndf(seedCam),rndf(seedCam))*2.0-1.0)*0.001;

  /*
      float focaldist = 7.3;
          float radius = 0.09;

          vec3 n = dir;
          vec3 W = (abs(n.x)>0.99)?vec3(0.,1.,0.):vec3(1.,0.,0.);
          vec3 N = n;
          vec3 T = normalize(cross(N,W));
          vec3 B = cross(T,N);

          float ang = rndf(r)*2.0*3.14159;
          float scale = sqrt(rndf(r))*radius;
          vec2 offset = vec2(cos(ang), sin(ang))*scale;

          vec3 focuspoint = pos + ((dir*focaldist) / dot(dir,dircam)); //these
     will lie on the focal plane

          pos = pos + B*offset.x;
          pos = pos + -T*offset.y;

          dir = normalize(focuspoint - pos);
  */

  vec3 cam = pos;
  vec3 col2 = vec3(0.);
  vec3 camera = pos;
  // if(traceVolume(camera, dir, r)){
  //    col2 += l*cccc;
  //}
  colorfog = vec4(col2, 0.);
  float isk = 0.;
  float isk2 = 0.;
  vec3 col = vec3(0.);

  vec3 tt = vec3(1.);
  vec3 normals = vec3(0.);
  vec3 positions = vec3(0.);
  vec3 albedos = vec3(0.);
  vec3 secp = pos;
  float Depth = 0.;
  float isEM = 0.;
  float firstRough = 1.;
  float lengthRefl = 0.;
  vec3 reflectedNorm = vec3(1.);
  vec3 reflectedAlb = vec3(1.);
  vec3 shadow = vec3(0.);
  float keepk = 0.;

  // for(int i = 0; i < 1; i++){
  // bool trace(inout vec3 p, vec3 d, inout vec3 watp, inout vec3 watn, bool
  // renderWat, inout bool hitswat){

  bool iswat = false;
  vec3 watP = vec3(0.);
  vec3 watN = vec3(0.);

  if (trace(pos, dir, watP, watN, removeWater, iswat)) {

    vec3 n = norm(pos - dir * epsilon * 2.);

    // if(i == 0){
    //  keepk = k;
    positions = pos;
    normals = n;
    albedos = cccc;
    // if(iswat){
    //  albedos *= vec3(0.2, 0.6, 0.9);
    // }
    Depth = length(pos - cam);
    secp = pos;
    firstRough = rough;
    cccc = vec3(1.);
    vec3 newdir = ldir.xzy;
    vec3 newpos = pos + n * epsilon * 2.;
    float rough2 = rough;
    float l2 = l;
    vec3 cccc2 = cccc;
    /* if(!trace(newpos, newdir)){
         if(l2 < 0.01){
             float pdf2 = max(dot(newdir, n),0.)/3.14159;
             vec3 brdf2 = vec3(1.)/3.14159;
             if(rough2 < 0.5){
                 pdf2 = ggx_pdf(reflect(wi,n),newdir,firstRough);
             }
             vec3 d2 = newdir;
             vec3 h = normalize(-wi + d2);

             float D2=ggx_D(reflect(wi,n),d2,firstRough);
             float G2 = ggx_G2(h,n,wi,d2,firstRough);//cook torrance based
     geometry term vec3 F2 = ggx_F(vec3(0.04), max(dot(-wi, n), 0.));//schlicks
     approx to the fresnel term vec3 specular2 =
     (D2*G2*F2)/max(4.*max(dot(-wi,n),0.)*max(dot(d2,n),0.6),0.0001);
             if(firstRough < 0.5){
                 brdf2 = specular2;
                 brdf2 *=  (1.0+2.*(1.-firstRough)*max(dot(d2,n),0.));

             }

             pdf2 = max(pdf2, 0.001);
             shadow = tt*brdf2*max(dot(newdir, n),0.)/pdf2;
             //col += (tt*brdf2*max(dot(newdir,
     n),0.)/pdf2)*vec3(0.9,0.8,0.7)*0.5;
         }

     }*/
    rough = rough2;
    l = l2;
    cccc = cccc2;

    //  }

    if (l > 0.01) {
      // if(i == 0){
      isEM = 1.0;
      // }
      col += tt * l * cccc;
      // break;
    }
    // col = vec3(cccc)*max(dot(n, ldir),0.);
    pos += n * epsilon * 2.;

    dir = angledircos(n.xzy, r).xzy;
    float pdf = max(dot(dir, n), 0.) / 3.14159;
    vec3 brdf = cccc / 3.14159;

  } else {
    // vec3 skyp2(vec3 d, vec3 lig){//my code to begin with

    // col +=

    positions = pos;

    // if(i == 0){
    // keepk = k;
    isk2 = 1.0;
    if (renderSkyAndSun) {
      // albedos = tt*skyp2(dir.xzy, ldir)*5.;
      // vec3 sky(vec3 p, vec3 d, vec3 lig){

      // albedos = sky(vec3(0.), dir.xzy, normalize(ldir).xzy)*1.*
      // max(dot(normalize(ldir).xzy, vec3(0.,0.,1.)),0.34);
      vec3 rayDirection = dir.xzy;
      rayDirection = normalize(rayDirection);
      vec2 TX = vec2((atan(rayDirection.z, rayDirection.x) /
                      6.283185307179586476925286766559) +
                         0.5,
                     acos(rayDirection.y) / 3.1415926535897932384626433832795);
      albedos = texture2D(skytex, TX).rgb * 4.;
    }
    // albedos = vec3(1.);
    Depth = 100.;
    // }
  }

  //}
  // rcnorm
  vec3 delta = lastViewPos - viewPos;
  imageStore(rcpos, ivec3((positions + normals * 0. + 90.0 - floor(viewPos))),
             vec4(positions, isEM));
  imageStore(rcnorm, ivec3((positions + normals * 0. + 90.0 - floor(viewPos))),
             vec4(normals, 1.));

  position = vec4(positions, isEM);
  normal = vec4(normals, Depth);
  albedo = vec4(albedos, isk2);
  holdinfo = vec4(0., vec3(max(firstRough * firstRough, 0.001)));

  if (!removeWater) {
    iswat = false;
  }
  watpos = vec4(watP, (iswat) ? 1. : 0.);
  watnorm = vec4(watN, length(watP - cam));
  // reflectionAlb = vec4(clamp(shadow,0., 100.), 1.);
}