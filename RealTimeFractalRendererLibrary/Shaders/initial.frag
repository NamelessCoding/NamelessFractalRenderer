

layout(location = 0) out vec4 position;
layout(location = 1) out vec4 normal;
layout(location = 2) out vec4 albedo;
layout(location = 3) out vec4 holdinfo;
layout(location = 4) out vec4 colorfog;
layout(location = 5) out vec4 watpos;
layout(location = 6) out vec4 watnorm;

in vec2 texCoord;

layout(binding = 0, Rgba32f) uniform image3D rcpos;
layout(binding = 1, Rgba32f) uniform image3D rcnorm;
uniform sampler2D skytex;

uniform mat4 invview;
uniform mat4 invproj;
uniform vec2 wh;

uniform vec3 viewPos;


struct Level {
    uint isEmpty;
};

struct radianceCache{
    vec3 col;
    vec3 pos;
    vec3 normal;
    float isEM;
};


layout(std430, binding=6) buffer buffHigh {
    radianceCache nodesHigh[];
} nodesH;



uint maxIDLow = 120;

uint to1DLow(uint x, uint y, uint z){
  return (x + y*maxIDLow + z*maxIDLow*maxIDLow);
}

uint maxIDMid = 16;

uint to1DMid(uint x, uint y, uint z){
  return (x + y*maxIDMid + z*maxIDMid*maxIDMid);
}
uint maxIDHigh = 5;

uint to1DHigh(uint x, uint y, uint z){
  return (x + y*maxIDHigh + z*maxIDHigh*maxIDHigh);
}


uint to1DRad(uint x, uint y, uint z, uint w){
  return (x + y*32 + z*32*32 + w*32*32*32);
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
/*
bool traceNewTestBasic(inout vec3 p, vec3 d, float times) {
  vec3 cam = p;


  bool hasHit = true;


  vec3 rayDir = d;
  vec3 rayPos = p;
  vec3 mapPos = vec3((rayPos + 0.));
  float epsl = epsilon;

  
  //vec3 prevPos = mapPos; 
  bool managedToHit = true;

  p = rayPos;
  
  float boostEps = 1.;
  
 
  vec3 Step = vec3((rayDir.x < 0.)?-1.:1., (rayDir.y < 0.)?-1.:1., (rayDir.z < 0.)?-1.:1.);
  vec3 invDir = 1./rayDir;
  vec2 tts = boxIntersection(cam-vec3(0., 0., 0.), d, vec3(10., 10., 1.));

  vec3 T = 1.0 / abs(rayDir);
  vec3 voxelSize1 = vec3(10./2.);
  vec3 voxelSize2 = vec3(10./16.);
  vec3 voxelSize3 = vec3(10./128.);

  vec3 deltaT1 = voxelSize1*invDir;
  vec3 deltaT2 = voxelSize2*invDir;
  vec3 deltaT3 = voxelSize3*invDir;

  vec3 voxelIncr = vec3(0.);
  int state = 0;
  bool hasHitFilledCell = false;

  vec3 currPos = rayPos;

  for(int i = 0; i < 290; i++){
    voxelIncr.x = ((T.x <= T.y) && (T.x <= T.z))?1.:0.;
    voxelIncr.y = ((T.y <= T.x) && (T.y <= T.z))?1.:0.;
    voxelIncr.z = ((T.z <= T.x) && (T.z <= T.y))?1.:0.;

    currPos += d*0.1;
    /*if(state == 0){
      
      //  vec3 currentPosition = ((pos/128.)*2.-1.)*10.;
      T += voxelIncr*deltaT1;
      mapPos += voxelIncr*Step*voxelSize1;
      vec3 index = ((mapPos/10.)*0.5 + 0.5)*2.;
      if(index.x >= 0. && index.x < 2. && index.y >= 0. && index.y < 2. && index.z >= 0. && index.z < 2.){
        uint id = to1DHigh(uint(index.x), uint(index.y), uint(index.z));
        Level lev1 = nodesH.nodesHigh[id];
        if(lev1.isEmpty > 0){
          state = 1;
        }
      }
      
    }else if(state == 1){
      T += voxelIncr*deltaT2;
      mapPos += voxelIncr*Step*voxelSize2;
      vec3 index = ((mapPos/10.)*0.5 + 0.5)*16.;
      if(index.x >= 0. && index.x < 16. && index.y >= 0. && index.y < 16. && index.z >= 0. && index.z < 16.){
        uint id = to1DMid(uint(index.x), uint(index.y), uint(index.z));
        Level lev1 = nodesM.nodesMid[id];
        if(lev1.isEmpty > 0){
          state = 2;
        }else{
          state = 0;
        }
      }
      

    }else if(state == 2){
      T += voxelIncr*deltaT3;
      mapPos += voxelIncr*Step*voxelSize3;
      vec3 index = ((mapPos/10.)*0.5 + 0.5)*128.;
      if(index.x >= 0. && index.x < 128. && index.y >= 0. && index.y < 128. && index.z >= 0. && index.z < 128.){
        uint id = to1DLow(uint(index.x), uint(index.y), uint(index.z));
        Level lev1 = nodesL.nodesLow[id];
        if(lev1.isEmpty > 0){
          hasHitFilledCell = true;
          break;
        }else{
          state = 1;
        }
      }
      

    }
    vec3 index = ((currPos/20.)*0.5 + 0.5)*5.;
    if(index.x >= 0. && index.x < 5. && index.y >= 0. && index.y < 5. && index.z >= 0. && index.z < 5.){
      uint id = to1DHigh(uint(index.x), uint(index.y), uint(index.z));
      Level lev1 = nodesH.nodesHigh[id];
      if(lev1.isEmpty > 0){
        index = ((currPos/20.)*0.5 + 0.5)*16.;
        if(index.x >= 0. && index.x < 16. && index.y >= 0. && index.y < 16. && index.z >= 0. && index.z < 16.){
          id = to1DMid(uint(index.x), uint(index.y), uint(index.z));
          lev1 = nodesM.nodesMid[id];
          if(lev1.isEmpty > 0){
              index = ((currPos/20.)*0.5 + 0.5)*120.;
              if(index.x >= 0. && index.x < 120. && index.y >= 0. && index.y < 120. && index.z >= 0. && index.z < 120.){
                id = to1DLow(uint(index.x), uint(index.y), uint(index.z));
                lev1 = nodesL.nodesLow[id];
                if(lev1.isEmpty > 0){
                  hasHitFilledCell = true;
                  break;
                }
              }
            }
          }
      }
    }

  }

if(!hasHitFilledCell){
  return false;
}
  //float iterShit = 0.;
p = currPos-d*0.1;
int iters = 25;
//if(!managedToHit){
 // iters = 60;
//}

  for (int i = 0; i < iters; i++) {
    // p = cam + d*t;

    float A = map(p, times);
    float dist = A;
   
    // float precis = 0.001 * t;

    if (dist < epsl) {
      
      return true;
    }

    // t += dist;
    epsl = epsilon * (1.+ boostEps*sqrt(length(cam-p))) ;
 
    p += d * dist;
  }


  return false;
}*/

const float phi2 = 1.32471795724474602596;
const vec2 a = vec2(1.0 / phi2, 1.0 / (phi2 * phi2));

vec2 R2(float n) { return fract(a * n + 0.5); }
#define RESOLUTION 0.5

vec2 jitter() {
  return ((R2(float(int(time) % 1000)) * 2. - 1.) / (wh * 2. * RESOLUTION));
}

void main() {
  vec2 fragCoord = texCoord * wh;

   if(int(fragCoord.x) % 2 == int(time)%2 && int(fragCoord.y) % 2 == int(time)%2){
      //discard;
   }

  uint r = uint(uint(fragCoord.x) * uint(1973) +
                uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) |
           uint(1);
  vec2 iResolution = wh;
  uint seedCam = uint(time);
  vec2 smallOffset =
      ((vec2(rndf(seedCam), rndf(seedCam))) * 2. - 1.) / iResolution;
   //smallOffset = vec2(0.);
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
  float iskwater = 0.;
  float issubb = 0.;
  // inout vec3 glassP, inout vec3 glassN, inout bool hasHitGlass
  vec3 glassP = vec3(0.);
  vec3 glassN = vec3(0.);
  bool hasHitGlass = false;
  //  if (traceInit(pos, dir, watP, watN, removeWater, iswat, iskwater, float(time)*0.04, glassP, glassN, hasHitGlass, viewPos, false ) ) {

  if (trace(pos, dir, watP, watN, removeWater, iswat, iskwater, float(time)*0.04, viewPos, false ) ) {

    if(issubsurface){
      issubb = 1.;
    }
    vec3 n = norm(pos - dir * epsilon * 2.,float(time)*0.04);

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
    // vec3 newdir = ldir.xzy;
    vec3 newpos = pos + n * epsilon * 2.;
    float rough2 = rough;
    float l2 = l;
    vec3 cccc2 = cccc;

    rough = rough2;
    l = l2;
    cccc = cccc2;

    //  }

    if (l > 0.01) {
      // if(i == 0){
        if(!iswat || !removeWater){
          isEM = 1.0;
        }
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

      p22 = vec4((clamp(texCoord, 0., 1.) * 2.0 - 1.0), 0.0, 1.0);
      // p22.xy += smallOffset*1.;

      dir = (invproj * p22).xyz / (invproj * p22).w;
      dir = normalize(mat3(invview) * dir);

      vec3 rayDirection = dir.xzy;
                    rayDirection.y *= -1.;

      rayDirection = normalize(rayDirection);

      vec2 a = normalize(rayDirection.xy);
        float x = (a.y < 0.)?2.*3.14159 - acos(a.x):acos(a.x);
        x /= 2.*3.14159;
        float y = acos(rayDirection.z)/3.14159;
        vec2 TX = vec2(x,y); 
      albedos = texture2D(skytex, TX).rgb * 4.;
    }
    // albedos = vec3(1.);
    Depth = 100.;
    // }
  }

  //}
  // rcnorm
  // vec3 delta = lastViewPos - viewPos;
  imageStore(rcpos, ivec3((positions + normals * 0. + 90.0 - floor(viewPos))),
             vec4(positions, isEM));
  imageStore(rcnorm, ivec3((positions + normals * 0. + 90.0 - floor(viewPos))),
             vec4(normals, 1.));


    for(int i = 0; i < 12; i++){
      vec3 index = ((currPos/20.)*0.5 + 0.5)*5.;

      if(index.x >= 0. && index.x < 5. && index.y >= 0. && index.y < 5. && index.z >= 0. && index.z < 5.){
        uint id = to1DRad(uint(index.x), uint(index.y), uint(index.z));
        radianceCache lev1 = nodesH.nodesHigh[id];
      }
    }

//bool traceNewTestBasic(inout vec3 p, vec3 d, float times) {
//  pos =
//       viewPos;
//   if(traceNewTestBasic(pos, wi, 0.)){
//         vec3 n2 = norm(pos - wi * epsilon * 2.,float(time)*0.04);

//     albedos = cccc;
//   }else{
//     albedos = vec3(0.);
//   }


  position = vec4(positions, isEM);
  normal = vec4(normals, Depth);
  albedo = vec4(albedos, isk2);
  holdinfo = vec4(issubb, vec3(max(firstRough * firstRough, 0.001)));

  if (!removeWater) {
    iswat = false;
  }
  watpos = vec4(watP, (iswat) ? 1. : 0.);
  watnorm = vec4(watN, iskwater);

  // = time*0.1;
  // reflectionAlb = vec4(clamp(shadow,0., 100.), 1.);
}