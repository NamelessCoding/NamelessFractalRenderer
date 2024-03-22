

layout(location = 0) out vec4 weigths;
layout(location = 1) out vec4 outrad;
layout(location = 2) out vec4 weigthsFog;
layout(location = 3) out vec4 outradFog;

in vec2 texCoord;

/*
_SpatialRestirShader.SetInt("color", 0);
            _SpatialRestirShader.SetInt("position", 1);
            _SpatialRestirShader.SetInt("normal", 2);
            _SpatialRestirShader.SetInt("albedo", 3);
            _SpatialRestirShader.SetInt("secondpos", 4);
            _SpatialRestirShader.SetInt("temppos", 5);
            _SpatialRestirShader.SetInt("prevW", 6);
            _SpatialRestirShader.SetInt("prevL", 7);

*/

uniform sampler2D color;
uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D secondpos;
uniform sampler2D temppos;
uniform sampler2D prevW;
uniform sampler2D prevL;
uniform sampler2D tempW;
uniform sampler2D tempL;
uniform sampler2D prevN;
uniform sampler2D reflAlb;
uniform sampler2D prevPosition;
uniform sampler2D prevSecondPosition;

uniform sampler2D tempfog;
uniform sampler2D tempLofog;
uniform sampler2D temppos2;
uniform sampler2D wightfog;
uniform sampler2D Lofog;
uniform sampler2D fosecg;

// fosecg
/*
 _SpatialRestirShader.SetInt("tempfog", 14);
            _SpatialRestirShader.SetInt("tempLofog", 15);
            _SpatialRestirShader.SetInt("temppos", 16);
            _SpatialRestirShader.SetInt("wightfog", 17);
            _SpatialRestirShader.SetInt("Lofog", 18);

*/

uniform mat4 view;
uniform mat4 projection;
uniform mat4 invview;
uniform mat4 invproj;
uniform mat4 prevview;
uniform mat4 prevproj;
uniform vec2 wh;
uniform float time;
uniform float time2;
uniform vec3 viewPos;
uniform vec3 lastViewPos;
uniform vec3 ldir;

float lum(vec3 c) {
  return sqrt(0.299 * c.x * c.x + 0.587 * c.y * c.y + 0.114 * c.z * c.z);
}

vec2 motionVectorSpecular(vec3 pos, vec2 motionpos, vec2 texcord,
                          vec3 prevpos) {
  vec2 iResolution = wh;

  float stepSize = 4.;
  vec2 bestPos = texcord;
  vec2 centerPos = texcord;
  float bestDist = length(pos - prevpos);
  // vec3 pm = texture2D(color, texCoord - motionpos).xyz;
  vec2 m = texcord - motionpos;
  vec3 prevposm = texture2D(prevSecondPosition, m).xyz;
  if (length(pos - prevposm) < bestDist) {
    bestDist = length(pos - prevposm);
    bestPos = m;
    centerPos = m;
  }

  for (int i = 0; i < 20; i++) {
    vec2 q =
        (centerPos * iResolution + (vec2(1., 0.) * stepSize)) / iResolution;
    vec3 prevposq = texture2D(prevSecondPosition, q).xyz;
    if (length(pos - prevposq) < bestDist) {
      bestDist = length(pos - prevposq);
      bestPos = q;
    }

    q = (centerPos * iResolution + (vec2(-1., 0.) * stepSize)) / iResolution;
    prevposq = texture2D(prevSecondPosition, q).xyz;
    if (length(pos - prevposq) < bestDist) {
      bestDist = length(pos - prevposq);
      bestPos = q;
    }

    q = (centerPos * iResolution + (vec2(0., 1.) * stepSize)) / iResolution;
    prevposq = texture2D(prevSecondPosition, q).xyz;
    if (length(pos - prevposq) < bestDist) {
      bestDist = length(pos - prevposq);
      bestPos = q;
    }

    q = (centerPos * iResolution + (vec2(0., -1.) * stepSize)) / iResolution;
    prevposq = texture2D(prevSecondPosition, q).xyz;
    if (length(pos - prevposq) < bestDist) {
      bestDist = length(pos - prevposq);
      bestPos = q;
    }

    q = (centerPos * iResolution + (vec2(0., 0.) * stepSize)) / iResolution;
    prevposq = texture2D(prevSecondPosition, q).xyz;
    if (length(pos - prevposq) < bestDist) {
      bestDist = length(pos - prevposq);
      bestPos = q;
    }

    if (length(bestPos * iResolution - centerPos * iResolution) < 0.001) {
      if (stepSize == 1.) {
        break;
      }
      stepSize *= 0.5;
    }
    centerPos = bestPos;
  }

  for (int i = 0; i < 9; i++) {
    // if(i == 4){continue;}
    vec2 offset = vec2(float(i % 3) - 1., float(i / 3) - 1.);
    vec3 prevposq = texture2D(prevSecondPosition,
                              (centerPos * iResolution + offset) / iResolution)
                        .xyz;
    if (length(pos - prevposq) < bestDist) {
      bestDist = length(pos - prevposq);
      bestPos = (centerPos * iResolution + offset) / iResolution;
    }
  }
  return bestPos;
}

void main() {
  /*
uniform sampler2D color;
uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D secondpos;
uniform sampler2D temppos;
uniform sampler2D prevW;
uniform sampler2D prevL;
uniform sampler2D tempW;
uniform sampler2D tempL;
*/

  float roughness = texture2D(color, texCoord).w;
  // if(roughness < 0.5){
  //     weigths = vec4(1., 1., 1., 0.);
  // 	//outrad = vec4(clamp(texture2D(tempL, ProjectedCoordinates).xyz,0.,
  // 100.) , 1.0);
  //     outrad = vec4(texture2D(color, texCoord).xyz, 1.);
  //  return;
  // }

  vec2 fragCoord = texCoord * wh;
  uint r = uint(uint(fragCoord.x) * uint(1973) +
                uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) |
           uint(1);
  r = uint(time);
  vec3 cameraOffset = viewPos - lastViewPos;
  vec3 View = texture2D(position, texCoord).xyz;
  vec4 Projected = vec4(View.xyz, 1.); // + vec4(cameraOffset, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;

  vec3 InitSampleLO = texture2D(color, texCoord).xyz;
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
  float isSpec = texture2D(reflAlb, texCoord).w;

  if (ProjectedCoordinates.x > 1. || ProjectedCoordinates.x < 0. ||
      ProjectedCoordinates.y > 1. || ProjectedCoordinates.y < 0. || disc) {
    ProjectedCoordinates = texCoord;
    outsc = true;
  }

  /*
  if(roughness < 0.5){
   View = texture2D(reflAlb , texCoord).xyz;
     Projected = vec4(View.xyz, 1.);// + vec4(cameraOffset, 0.);
    Projected = prevview * Projected;
    Projected = prevproj * Projected;
    Projected /= Projected.w;
     ProjectedCoordinates = Projected.xy * 0.5 + 0.5;


  ProjectedCoordinates = motionVectorSpecular(texture2D(secondpos,
  texCoord).xyz, texCoord-ProjectedCoordinates , texCoord,
  texture2D(prevSecondPosition, texCoord).xyz);

  if(texture2D(secondpos, texCoord).w > 0.5 || true){

      //    vec3 LO = InitSampleLO;
      //    weigths = vec4(1., 1., 1., 1.0);

          // outrad = vec4(clamp(LO,0.,100.) , 1.0);
      // return;
      isthesky = true;
  }

  if(texCoord.x > 0.5){
  //  vec3 LO = InitSampleLO;
  //      weigths = vec4(1., 1., 1., 1.0);

  // 	 outrad = vec4(clamp(LO,0.,100.) , 1.0);
  //      return;
  }

  }*/

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
    if (isSpec < 0.5) {
      //    pq *= cosT;
    }

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
    if (isSpec < 0.5) {
      //    pq *= cosT;
    }

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

  // FOG

  /*
              InitSampleLO = texture2D(tempLofog, texCoord).xyz;
              Rs = texture2D(wightfog, ProjectedCoordinates).xyz;
              LO = texture2D(Lofog, ProjectedCoordinates).xyz;

                          //vec3 LO = texelFetch(colortex13,ivec2(iResolution*
  ProjectedCoordinates),0).xyz;

                           RM = max(Rs.x,1.);
                           RW = Rs.z;
                           Rw = Rs.y;

               if(outsc ){
                  RM = 0.;
                  RW = 0.;
                  Rw = 0.;
                  LO =  texture2D(tempLofog, texCoord).xyz;
                // LO = InitSampleLO;
              }

              /*vec4 currN = texture2D(normal, texCoord);
              vec4 prevN = texture2D(prevN, ProjectedCoordinates);
              bool disc = false;
              if (dot(normalize(currN.xyz),normalize(prevN.xyz)) < 0.39) {
                  disc = true;
              }
              if (abs(currN.w - prevN.w) > 1.1)
              {
                  disc = true;
              }

               if(isthesky){
                  RM = 0.;
                  Rw = 0.;
                  RW = 0.;

                   // LO = InitSampleLO;
               }


               // maxiterations = 9;
              //std::vector<RESERVOIR> QN;
              //QN.push_back(Rs);
              //Sample init = InitialSampleBuffer[index];
               Z = RM;

              for (int s = 0; s < maxiterations; s++) {

                  float radius = (30.0)*sqrt(rndf(r));
                  float angle = rndf(r)*2.0*3.14159;
                  float x =  (cos(angle) * radius);
                  float y =  (sin(angle) * radius);

                  vec2 fincords = (TexCoords*iResolution + vec2(x,y)) /
  iResolution; vec2 finn = (ProjectedCoordinates*iResolution + vec2(x,y)) /
  iResolution;


  /*
  uniform sampler2D tempfog;
  uniform sampler2D tempLofog;
  uniform sampler2D temppos;
  uniform sampler2D wightfog;
  uniform sampler2D Lofog;


                  vec3 R = texture2D(tempfog, finn).xyz;
                  //vec3 R = texelFetch(colortex4,ivec2(iResolution*
  finn),0).xyz;

                  vec3 RnLo = texture2D(tempLofog, finn).xyz;
                              //vec3 RnLo =
  texelFetch(colortex6,ivec2(iResolution* finn),0).xyz;
                  //vec4 psandisk = texelFetch(colortex9,ivec2(iResolution*
  finn),0).xyzw; vec4 psandisk = texture2D(temppos2, finn);
                  //vec3 ns = texelFetch(colortex5,ivec2(iResolution*
  finn),0).xyz;

                              float RnM = max(R.x,0.01);
                              float RnW = R.z;
                              float Rnw = R.y;



                          //vec4 info = vec4(texelFetch(colortex0,
  ivec2(iResolution*TexCoords),0).xyz,
                  //texelFetch(colortex1,
  ivec2(iResolution*TexCoords*0.5),0).w);
                  //vec4 info2 = vec4(texelFetch(colortex0,
  ivec2(iResolution*finn),0).xyz,
                  //texelFetch(colortex1, ivec2(iResolution*finn*0.5),0).w);
                  vec4 info = texture2D(normal, TexCoords);
                  vec4 info2 = texture2D(normal, finn);
                  float pq = lum(RnLo);

                  /*if(texture2D(albedo, finn).w < 0.5 && texture2D(position,
  finn).w < 0.5 ){

                      if (dot(normalize(info.xyz),normalize(info2.xyz)) < 0.926)
  { continue;
                      }
                      if ( (info.w) > 1.1 * (info2.w) || (info.w) < 0.9 *
  (info2.w))
                      {
                          continue;
                      }
                  }
                  if(texture2D(albedo, TexCoords).w < 0.5 && texture2D(position,
  TexCoords).w < 0.5 ){ if (dot(normalize(info.xyz),normalize(info2.xyz)) <
  0.926) { continue;
                      }
                      if ( (info.w) > 1.1 * (info2.w) || (info.w) < 0.9 *
  (info2.w))
                      {
                          continue;
                      }
                  }
                  //vec3 org = texture2D(fosecg, texCoord).xyz;
                  //vec3 sec = texture2D(temppos2, finn).xyz;

                  //if(max(dot(normalize(org - viewPos), normalize(sec -
  viewPos)),0.) < 0.5){
                  //    continue;
                  //}


                  //vec3 pporigin = SceneSpaceToVoxelSpace(texelFetch(colortex2,
  ivec2(iResolution*TexCoords),0).xyz); vec3 pporigin = texture2D(fosecg,
  TexCoords).xyz;
                  //vec3 pporigin2 =
  SceneSpaceToVoxelSpace(texelFetch(colortex2, ivec2(iResolution*finn),0).xyz);

                  vec3 initxv = pporigin;
                  vec3 Rnxs = psandisk.xyz;


                  /*if(length(Rnxs - initxv) > 0.5){
                      float lngh = length(Rnxs - initxv);
                      vec3 d2 = normalize(Rnxs - initxv);
                      vec3 p2 = initxv;
                      if(trace(p2, d2)){
                          if(length(p2 - Rnxs)<lngh){
                              pq = 0.;
                          }
                      }
                  }

   Rw /= max(RM, 0.01);
      RM = min(RM, 500.);
      Rw *= max(RM, 0.01);
      float M0 = RM;
      float upd = pq * RnW * max(RnM, 1.);
      Rw = Rw + upd;
      RM = M0 + RnM;
      // Rw /= max(RM,1.);
      // RM = min(RM, 500.);
      // Rw *= RM;
     if(rndf(r) < (upd/max(Rw,0.001))){
                  LO = RnLo;
                  }
      // RM = RM + 1.0;
      Z += RnM;


              }
                if (Z*lum(LO) < 0.001) {
          RW = 0.0;
      }
      else {
          RW = Rw / (Z * lum(LO));
      }

              weigthsFog = vec4(RM, Rw, RW, 0.);
                          outradFog = vec4(clamp(LO,0., 100.) , 1.0);



  */

  // vec3 col = texture2D(color, texCoord).xyz;
  // outputColor = vec4(col, 1.);
}