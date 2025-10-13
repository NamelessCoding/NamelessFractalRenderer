

layout(location = 0) out vec4 position;
layout(location = 1) out vec4 normal;
layout(location = 2) out vec4 albedo;
layout(location = 3) out vec4 holdinfo;
layout(location = 4) out vec4 colorfog;
layout(location = 5) out vec4 watpos;
layout(location = 6) out vec4 watnorm;
layout(location = 7) out vec4 clll;

in vec2 texCoord;

layout(binding = 0, Rgba32f) uniform image3D rcpos;
layout(binding = 1, Rgba32f) uniform image3D rcnorm;
layout(binding = 5, Rgba16f) uniform image3D colX;
layout(binding = 7, r16f) uniform image3D worleyS;

uniform sampler2D skytex;

uniform mat4 invview;
uniform mat4 invproj;
uniform vec2 wh;

uniform vec3 viewPos;
uniform vec3 ldir;
uniform float cloudCoverage;
uniform sampler2D suntex;

uniform mat4 lightproj;
uniform mat4 lightview;
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
vec3 shadowS(vec3 pos) {
  vec4 sspace = lightproj * lightview * vec4(pos, 1.);
  sspace.xyz /= sspace.w;

  vec3 coords = sspace.xyz * 0.5 + 0.5;

  // float currDepth = texelFetch(suntex, ivec2(2048.*coords), 0).r;
  float currDepth = texture2D(suntex, coords.xy).r;
  // vec3 currPos = texture2D(suntex, coords.xy).xyz;
  vec3 col = vec3(1.);
  if (coords.z > currDepth + 0.0025 && coords.x >= 0. && coords.x <= 1. &&
      coords.y >= 0. && coords.y <= 1.) {
    col = vec3(0.);
  }



  return col;
}
// /https://github.com/btmxh/glsl-intersect/blob/master/3d/intersection/rayAABB.glsl

bool rayAABB(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax, out vec2 result) {
    vec3 rayInvDir = 1.0 / rayDir;
    vec3 tbot = rayInvDir * (boxMin - rayOrigin);
    vec3 ttop = rayInvDir * (boxMax - rayOrigin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    vec2 t = max(tmin.xx, tmin.yz);
    float t0 = max(t.x, t.y);
    t = min(tmax.xx, tmax.yz);
    float t1 = min(t.x, t.y);
    result = vec2(t0, t1);
    return t1 > max(t0, 0.0);
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

vec3 modul(vec3 pp){

return vec3(mod(pp.xy*1., 1024), mod(pp.z*0.5, 512));
}

float remap3(float x, float a, float b, float c, float d)
{
    return (((x - a) / (b - a)) * (d - c)) + c;
}
float dens(vec3 p){

p *= 2.;
float f2 =  imageLoad(worleyS,ivec3(modul(p))).x;


    
    float cloud = f2;

  float densityL= max(max(cloud - (cloudCoverage)  , 0.) - max(modul(p).z - 30.,0.)/40., 0.);

    float Srb = clamp(
      remap3(clamp(p.z / 1125., 0., 0.09), 0., 0.09, 0., 1.), 0., 1.);//a *= clamp(abs(length(p)-6500.)*0.00013, 0.0, 1.);
//densityL *= Srb;
  densityL = clamp(densityL, 0., 1.);
	    densityL = 1.0 - pow(1.0 - densityL, 4.);
      return densityL;

}
float PM(float cost, float g){
float a = 3./(8.*3.14159);
float b = (1.0-g*g)*(1.0+cost*cost);
float c = (2.0+g*g)*pow(1.0+g*g-2.*g*cost, 3./2.);
return a*(b/c);
}

float PR(float cost){
return (3./(16.*3.14159))*(1.0+cost*cost);
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
      vec4((clamp(texCoord , 0., 1.) * 2.0 - 1.0), 0.0, 1.0);
  // p22.xy += smallOffset*1.;

  vec3 dir = (invproj * p22).xyz / (invproj * p22).w;
  dir = normalize(mat3(invview) * dir);

  vec4 p22cam =
      vec4((clamp(vec2(0.5) , 0., 1.) * 2.0 - 1.0), 0.0, 1.0);
  // p22.xy += smallOffset*1.;

  vec3 dircam = (invproj * p22cam).xyz / (invproj * p22cam).w;
  dircam = normalize(mat3(invview) * dircam);

  vec3 wi = dir;
  // dir = dir.xzy;
  vec3 pos =
viewPos;


      // +
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
    Depth = 10000.;
    // }
  }

  //}
  // rcnorm
  // vec3 delta = lastViewPos - viewPos;
  imageStore(rcpos, ivec3((positions + normals * 0. + 90.0 - floor(viewPos))),
             vec4(positions, isEM));
  imageStore(rcnorm, ivec3((positions + normals * 0. + 90.0 - floor(viewPos))),
             vec4(normals, 1.));


    // for(int i = 0; i < 12; i++){
    //   vec3 index = ((currPos/20.)*0.5 + 0.5)*5.;

    //   if(index.x >= 0. && index.x < 5. && index.y >= 0. && index.y < 5. && index.z >= 0. && index.z < 5.){
    //     uint id = to1DRad(uint(index.x), uint(index.y), uint(index.z));
    //     radianceCache lev1 = nodesH.nodesHigh[id];
    //   }
    // }

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


    pos =  viewPos; 
        float maxXY = 300.;
  float maxZ = 100.;
  //vec2 bbb = boxIntersection(pos, dircam, vec3(maxXY.xx/2., maxZ/2.));
      //clll
//.bool rayAABB(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax, out vec2 result) {
vec2 bbb = vec2(0.);
  vec4 colClouds = vec4(0.,0.,0.,1.);

  if(rayAABB(pos.xzy, wi.xzy, vec3(-maxXY.xx/2., 0.), vec3(maxXY.xx/2., maxZ), bbb) && false){
    vec3 startPos = pos.xzy ;
   


    vec3 endPos = pos.xzy + wi.xzy*min(Depth, bbb.y);
    int iter = 200;

    float stepS = length(endPos - startPos)/float(iter);
    vec3 accum = vec3(0.);

    float ttt = 1.;
    vec3 accumH = vec3(0.);
    vec3 rayDirection = ldir.xzy;
        rayDirection.y *= -1.;

        rayDirection = normalize(rayDirection);
        vec2 a = normalize(rayDirection.xy);
        float  x = (a.y < 0.)?2.*3.14159 - acos(a.x):acos(a.x);
        x /= 2.*3.14159;
       float   y = acos(rayDirection.z)/3.14159;
        vec2 TX = vec2(x,y); 
        vec3 sky2 =  texture2D(skytex, TX).rgb * 12.;
  sky2 = mix(vec3(0.9, 0.7, 0.2), vec3(1.), ldir.y);


    for(int i =0; i < iter; i++){
      startPos += stepS*wi.xzy;

// /  vec3 worldPos = vec3((pos.xy/512. - 0.5)*maxXY, (pos.z/256.)*maxZ);

      vec3 idPos = vec3((startPos.xy/(maxXY)+0.5)*256., ((startPos.z)/maxZ)*128.); 
      idPos = vec3(clamp(idPos.xy, 0., 256.), clamp(idPos.z, 0., 128.));
      //float dens2 = max(imageLoad(colX, ivec3(idPos)).x,0.);
      float dens2 = dens(startPos);

          startPos += .4*(vec3(rndf(r), rndf(r), rndf(r))*2.-1.)*max(dens2,0.02);


      if(dens2 > 0.){
        
          float accum2 = 0.;
          vec3 cam2 = startPos;
          vec3 lig = ldir.xzy;
          if(rayAABB(cam2, lig, vec3(-maxXY.xx/2., 0.), vec3(maxXY.xx/2., maxZ), bbb)){
            vec3 startPos2 = cam2;
            vec3 endPos2 = cam2 + lig*bbb.y;
            int iter2 = 40;

            for(int k = 0; k < iter2; k++){
              float densityL = dens(startPos2);
              
          
                    // accum2 += ph(cam2.z, 1300.)*30.;
              accum2 += densityL * 2.4 ;
              startPos2 += lig*1.15;
              
            }
          }
        vec3 finCol = vec3(0.);
      vec3 camz = startPos;
      for(int k = 0; k < 11; k++){
        camz = startPos + 12.*(vec3(rndf(r), rndf(r), rndf(r))*2.-1.);
        vec3 idPos2 = vec3((camz.xy/(maxXY)+0.5)*256., ((camz.z)/maxZ)*128.); 
        idPos2 = vec3(mod(idPos2.xy, 256.), mod(idPos2.z, 128.));
        finCol+=max(imageLoad(colX, ivec3(idPos2)).xyz,0.)*dens2;

      }
       vec3 pm2 = vec3((PM(max(dot(wi.xzy, lig), 0.), pow(0.76, 1.))).xxx) * 13.  +                   
			                   vec3((PM(max(dot(wi.xzy, lig), 0.), -pow(0.5, 1.))).xxx) * 5.;
        accum += dens2*ttt*mix(finCol*0.2, finCol, exp(-accum2));
       // accumH += dens2*ttt*exp(-accum2*2.5)*sky2*pm2*shadowS(startPos.xzy);
        //accum += dens2*ttt*exp(-accum2*0.1);
                      ttt *= pow(1.-dens2,1.);

      }



    }



    colClouds = max(vec4((accum*14.+accumH*22.)*0.014, ttt),0.);
  }
clll = colClouds;





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