

layout(location = 0) out vec4 weigths;
layout(location = 1) out vec4 outrad;
layout(location = 2) out vec4 weigthsFog;
layout(location = 3) out vec4 outradFog;

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

uniform mat4 prevview;
uniform mat4 prevproj;
uniform vec2 wh;
uniform float time;

float lum(vec3 c) {
  return sqrt(0.299 * c.x * c.x + 0.587 * c.y * c.y + 0.114 * c.z * c.z);
}

void main() {

  vec2 fragCoord = texCoord * wh;
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

  // vec3 InitSampleLO = texture2D(color, texCoord).xyz;
  bool isthesky = false;

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

  // vec3 LO = texelFetch(colortex13,ivec2(iResolution*
  // ProjectedCoordinates),0).xyz;

  float RM = max(Rs.x, 1.);
  float RW = Rs.z;
  float Rw = Rs.y;

  if (outsc) {
    RM = 0.;
    RW = 0.;
    Rw = 0.;
    LO = texture2D(tempL, ProjectedCoordinates).xyz;
    // LO = InitSampleLO;
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
    float radius = (30.0) * (rndf(r));
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

    if (dot(normalize(info.xyz), normalize(info2.xyz)) < 0.9) {
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
    float pq = lum(RnLo) * cosT;
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
    }
    // RM = RM + 1.0;
    // Z += RnM;
  }
  //           if (Z*lum(LO) < 0.001) {
  //    	RW = 0.0;
  // }
  // else {
  // 	RW = Rw / (Z * lum(LO));
  // }
  // LO =  texture2D(tempL, ProjectedCoordinates).xyz;
  // RW = 1.;
  // if(isSpec > 0.5 ){
  //     LO =  texture2D(tempL, ProjectedCoordinates).xyz;
  //     RW = 1.;
  // }

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
    vec3 R = texture2D(tempW, finn).xyz;
    // vec3 R = texelFetch(colortex4,ivec2(iResolution* finn),0).xyz;

    vec3 RnLo = texture2D(tempL, finn).xyz;
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

    if (dot(normalize(info.xyz), normalize(info2.xyz)) < 0.9) {
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
    float pq = lum(RnLo) * cosT;
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
    }
    // RM = RM + 1.0;
    //  Z += RnM;
  }
  if (Z * lum(LO) < 0.001) {
    RW = 0.0;
  } else {
    RW = Rw / (Z * lum(LO));
  }

  weigths = vec4(RM, Rw, RW, 0.);
  outrad = vec4(clamp(LO, 0., 1000.), 1.0);
}