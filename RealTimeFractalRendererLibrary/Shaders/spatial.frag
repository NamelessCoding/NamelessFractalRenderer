

layout(location = 0) out vec4 weigths;
layout(location = 1) out vec4 outrad;
layout(location = 2) out vec4 outPos;

in vec2 texCoord;

uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D temppos;
uniform sampler2D prevW;
uniform sampler2D prevL;
uniform sampler2D tempW;
uniform sampler2D tempL;
uniform sampler2D prevN;
uniform sampler2D prevSPos;
uniform sampler2D holdinfo;

uniform mat4 prevview;
uniform mat4 prevproj;
uniform vec2 wh;
//uniform float time;
uniform vec3 viewPos;

float lum(vec3 c) {
  return sqrt(0.299 * c.x * c.x + 0.587 * c.y * c.y + 0.114 * c.z * c.z);
}

void main() {

  vec2 fragCoord = texCoord * wh;

 // vec2 fragCoord = texCoord * wh;

  if(int(fragCoord.x) % 2 == int(time)%2 && int(fragCoord.y) % 2 == int(time)%2){
      //discard;
   }
  uint r = uint(uint(fragCoord.x) * uint(1973) +
                uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) |
           uint(1);
  r = uint(time);
  // vec3 cameraOffset = viewPos - lastViewPos;
  vec3 View = texture2D(position, texCoord).xyz;
  vec4 Projected = vec4(View.xyz, 1.); // + vec4(cameraOffset, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;
  uint seedCamPrev = uint(max(time, 0));
  vec2 smallOffset2 =
      ((vec2(rndf(seedCamPrev), rndf(seedCamPrev))) * 2. - 1.) / wh;
  // smallOffset2 = vec2(0.);

  ProjectedCoordinates -= smallOffset2 * 1.;

  // vec3 InitSampleLO = texture2D(color, texCoord).xyz;
  bool isthesky = false;


    bool isSub = texture2D(holdinfo, texCoord).x > 0.5;


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
  // float isSpec = texture2D(reflAlb, texCoord).w;

  if (ProjectedCoordinates.x > 1. || ProjectedCoordinates.x < 0. ||
      ProjectedCoordinates.y > 1. || ProjectedCoordinates.y < 0. || disc) {
    ProjectedCoordinates = texCoord;
    outsc = true;
  }

  vec3 Rs = texture2D(prevW, ProjectedCoordinates).xyz;
  vec3 LO = texture2D(prevL, ProjectedCoordinates).xyz;
  float ADDLO = texture2D(prevL, ProjectedCoordinates).w;
  vec4 prevPos = texture2D(prevSPos, ProjectedCoordinates);
  // vec3 LO = texelFetch(colortex13,ivec2(iResolution*
  // ProjectedCoordinates),0).xyz;

  float RM = max(Rs.x, 1.);
  float RW = Rs.z;
  float Rw = Rs.y;

  bool lowSamp = false;
  if (outsc) {
    RM = 0.;
    RW = 0.;
    Rw = 0.;
    LO = texture2D(tempL, texCoord).xyz;
    ADDLO = texture2D(tempL, texCoord).w;
    // LO = InitSampleLO;
    lowSamp = true;
  }
  /* if(isthesky){
      RM = 0.;
      Rw = 0.;
      RW = 0.;

       // LO = InitSampleLO;
   }*/

  const int maxiterations = 9;
  // std::vector<RESERVOIR> QN;
  // QN.push_back(Rs);
  // Sample init = InitialSampleBuffer[index];
  float Z = RM;
  vec2 TexCoords = texCoord;
  vec2 iResolution = wh;
  for (int s = 0; s < maxiterations; s++) {
    if (texture2D(albedo, texCoord).w > 0.5 ||
        texture2D(position, texCoord).w > 0.5) {
      break;
    }
    float radius = (30.0) * sqrt(rndf(r));
    float angle = rndf(r) * 2.0 * 3.14159;
    float x = (cos(angle) * radius);
    float y = (sin(angle) * radius);

    vec2 fincords = (TexCoords * iResolution + vec2(x, y)) / iResolution;
    vec2 finn = (ProjectedCoordinates * iResolution + vec2(x, y)) / iResolution;

    if (texture2D(albedo, finn).w > 0.5 || texture2D(position, finn).w > 0.5) {
      continue;
    }
    vec3 R = texture2D(tempW, finn).xyz;
    // vec3 R = texelFetch(colortex4,ivec2(iResolution* finn),0).xyz;

    vec3 RnLo = texture2D(tempL, finn).xyz;
    float rnadd = texture2D(tempL, finn).w;
    // vec3 RnLo = texelFetch(colortex6,ivec2(iResolution* finn),0).xyz;
    // vec4 psandisk = texelFetch(colortex9,ivec2(iResolution* finn),0).xyzw;
    vec4 psandisk = texture2D(temppos, finn);
    // vec3 ns = texelFetch(colortex5,ivec2(iResolution* finn),0).xyz;

    float RnM = max(R.x, 0.01);
    float RnW = R.z;
    float Rnw = R.y;

    // vec4 info = vec4(texelFetch(colortex0,
    // ivec2(iResolution*TexCoords),0).xyz,
    // texelFetch(colortex1, ivec2(iResolution*TexCoords*0.5),0).w);
    // vec4 info2 = vec4(texelFetch(colortex0, ivec2(iResolution*finn),0).xyz,
    // texelFetch(colortex1, ivec2(iResolution*finn*0.5),0).w);
    vec4 info = texture2D(normal, TexCoords);
    vec4 info2 = texture2D(normal, finn);

    if (dot(normalize(info.xyz), normalize(info2.xyz)) < 0.56) {
      continue;
    }
    if ((info.w) > 1.1 * (info2.w) || (info.w) < 0.9 * (info2.w)) {
      // continue;
    }
    // vec3 pporigin = SceneSpaceToVoxelSpace(texelFetch(colortex2,
    // ivec2(iResolution*TexCoords),0).xyz);
    vec3 pporigin = texture2D(position, TexCoords).xyz;
    // vec3 pporigin2 = SceneSpaceToVoxelSpace(texelFetch(colortex2,
    // ivec2(iResolution*finn),0).xyz);

    vec3 initxv = pporigin + normalize(info.xyz) * epsilon * 2.;
    vec3 Rnxs = psandisk.xyz;
    vec3 direction = normalize(Rnxs - initxv);
    float cosT = max(dot(direction, info.xyz), 0.);
    float pq = lum(RnLo) ;
    /*if (isSpec < 0.5) {
          pq *= cosT;
    }*/

    /*
                    if(length(Rnxs - initxv) > 0.5 && psandisk.w < 0.5){
                        vec3 d2 = normalize(Rnxs - initxv);
                        vec3 p2 = initxv;
                        if(trace(p2, d2)){
                            if(length(p2 - Rnxs)>0.5 ){
                                pq = 0.;
                            }
                        }
                    }*/

    Rw /= max(RM, 1.);
    RM = min(RM, 500.);
    Rw *= RM;
    float M0 = RM;
    float upd = pq * RnW * RnM;
    Rw = Rw + upd;
    RM = M0 + RnM;
    // Rw /= max(RM,1.);
    // RM = min(RM, 500.);
    // Rw *= RM;
    if (rndf(r) < (upd / max(Rw, 0.000001))) {
      LO = RnLo;
      prevPos = psandisk;
      ADDLO = rnadd;
    }
    // RM = RM + 1.0;
     Z += RnM;
  }



  //Z = RM;
   vec3 Rnxs = prevPos.xyz;
  vec4 info = texture2D(normal, TexCoords);
  vec3 pporigin = texture2D(position, TexCoords).xyz;
  vec3 initxv = pporigin + normalize(info.xyz) * epsilon * 2.;
  float isShad = 1.;
  if(length(Rnxs - initxv) > 0.5 && prevPos.w < 0.5 ){
                        vec3 d2 = normalize(Rnxs - initxv);
                        vec3 p2 = initxv;
                        vec3 watp = vec3(0.);
                        vec3 watn = vec3(0.);
                        bool renderw = false;
                        bool hitsw = false;
                        float isk2 = 0.;
                        bool stp = false;
                        if(trace(p2, d2,watp,watn,renderw,hitsw,isk2,float(time)*0.04,viewPos,stp)){
                            if(length(p2 - Rnxs)>0.1 ){
                                isShad = 0.;
                            }
                        }
                    }
                    float cosT = max(dot(normalize(Rnxs - initxv), normalize(info.xyz)),0.);
  float  phat = lum(LO)*isShad;
  if (Z * phat < 0.001) {
    RW = 0.0;
  } else {
    RW = Rw / (Z * phat);
  }

  
  
//
/*
if(false){
  Z = RM;
  for (int s = 0; s < maxiterations; s++) {
    if (texture2D(albedo, texCoord).w > 0.5 ||
        texture2D(position, texCoord).w > 0.5) {
      break;
    }
    float radius = (30.0) * (rndf(r));
    float angle = rndf(r) * 2.0 * 3.14159;
    float x = (cos(angle) * radius);
    float y = (sin(angle) * radius);

    vec2 fincords = (TexCoords * iResolution + vec2(x, y)) / iResolution;
    vec2 finn = (ProjectedCoordinates * iResolution + vec2(x, y)) / iResolution;

    if (texture2D(albedo, finn).w > 0.5 || texture2D(position, finn).w > 0.5) {
      continue;
    }
    //vec3 R = texture2D(tempW, finn).xyz;
    vec3 R = texture2D(prevW, finn).xyz;
    // vec3 R = texelFetch(colortex4,ivec2(iResolution* finn),0).xyz;
   // R = vec3(RM, Rw, RW);
    //vec3 RnLo = texture2D(tempL, finn).xyz;
    vec3 RnLo = texture2D(prevL, finn).xyz;
   // RnLo = LO;
        float rnadd = texture2D(prevL, finn).w;

    // vec3 RnLo = texelFetch(colortex6,ivec2(iResolution* finn),0).xyz;
    // vec4 psandisk = texelFetch(colortex9,ivec2(iResolution* finn),0).xyzw;
    vec4 psandisk = texture2D(prevSPos, finn);
    // vec3 ns = texelFetch(colortex5,ivec2(iResolution* finn),0).xyz;

    float RnM = max(R.x, 0.01);
    float RnW = R.z;
    float Rnw = R.y;

    // vec4 info = vec4(texelFetch(colortex0,
    // ivec2(iResolution*TexCoords),0).xyz,
    // texelFetch(colortex1, ivec2(iResolution*TexCoords*0.5),0).w);
    // vec4 info2 = vec4(texelFetch(colortex0, ivec2(iResolution*finn),0).xyz,
    // texelFetch(colortex1, ivec2(iResolution*finn*0.5),0).w);
    vec4 info = texture2D(normal, TexCoords);
    vec4 info2 = texture2D(normal, finn);

    if (dot(normalize(info.xyz), normalize(info2.xyz)) < 0.996) {
      continue;
    }
    if ((info.w) > 1.1 * (info2.w) || (info.w) < 0.9 * (info2.w)) {
      // continue;
    }
    // vec3 pporigin = SceneSpaceToVoxelSpace(texelFetch(colortex2,
    // ivec2(iResolution*TexCoords),0).xyz);
    vec3 pporigin = texture2D(position, TexCoords).xyz;
    // vec3 pporigin2 = SceneSpaceToVoxelSpace(texelFetch(colortex2,
    // ivec2(iResolution*finn),0).xyz);

    vec3 initxv = pporigin + normalize(info.xyz) * epsilon * 2.;
    vec3 Rnxs = psandisk.xyz;
    vec3 direction = normalize(Rnxs - initxv);
    float cosT = max(dot(direction, info.xyz), 0.);
    float pq = lum(RnLo);
    /*if (isSpec < 0.5) {
          pq *= cosT;
    }
          //(inout vec3 p, vec3 d, inout vec3 watp, inout vec3 watn,
          // bool renderWat, inout bool hitswat, inout float isk2,
           //vec3 viewPos = vec3(0.), bool stopatwat = false)
                    

   
    float M0 = RM;
    float upd = pq * RnW * RnM;
    Rw = Rw + upd;
    RM = M0 + RnM;
    Rw /= max(RM,1.);
    RM = min(RM, 500.);
    Rw *= RM;
    if (rndf(r) < (upd / max(Rw, 0.000001))) {
      LO = RnLo;
      prevPos = psandisk;
      ADDLO = rnadd;

    }
    // RM = RM + 1.0;
      Z += RnM;
  }
}
*/
 /*vec3 R = texture2D(tempW, TexCoords).xyz;
    // vec3 R = texelFetch(colortex4,ivec2(iResolution* finn),0).xyz;

    vec3 RnLo = texture2D(tempL, TexCoords).xyz;
    float rnadd = texture2D(tempL, TexCoords).w;
    // vec3 RnLo = texelFetch(colortex6,ivec2(iResolution* finn),0).xyz;
    // vec4 psandisk = texelFetch(colortex9,ivec2(iResolution* finn),0).xyzw;
    vec4 psandisk = texture2D(temppos, TexCoords);
    // vec3 ns = texelFetch(colortex5,ivec2(iResolution* finn),0).xyz;
  LO = RnLo;
     RM = max(R.x, 0.01);
     RW = R.z;
     Rw = R.y;*/
  outPos = vec4(prevPos);
  weigths = vec4(RM, Rw, RW, ADDLO*RW);
  outrad = vec4(clamp(LO, 0., 1000.), ADDLO);
}