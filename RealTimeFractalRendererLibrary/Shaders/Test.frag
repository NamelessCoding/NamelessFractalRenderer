

layout(location = 0) out vec4 color;
layout(location = 1) out vec4 secondp;
layout(location = 2) out vec4 reflection2;
layout(location = 3) out vec4 reflectionAlb;
layout(location = 4) out vec4 colortexFog;
layout(location = 5) out vec4 ssstex;
layout(location = 6) out vec4 sssTexDenoise;

//sssTexDenoise
in vec2 texCoord;

layout(binding = 2, Rgba16f) uniform image3D rcrad;

uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D holdinfo;
uniform sampler2D skytex;
uniform sampler2D suntex;
uniform sampler2D watpos;
uniform sampler2D watnorm;
uniform sampler2D sssprev;
uniform sampler2D prevN;

uniform sampler2D den1;

uniform mat4 invview;
uniform mat4 invproj;
uniform vec2 wh;
//uniform float time;

uniform vec3 viewPos;

uniform vec3 ldir;
uniform mat4 lightproj;
uniform mat4 lightview;
uniform mat4 prevview;
uniform mat4 prevproj;

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

  if (length(pos.xz - viewPos.xz) > 90.) {
    //  col = vec3(1.);
  }

  return col;
}
float PR(float cost) { return (3. / (16. * 3.14159)) * (1.0 + cost * cost); }
float PM(float cost, float g) {
  float a = 3. / (8. * 3.14159);
  float b = (1.0 - g * g) * (1.0 + cost * cost);
  float c = (2.0 + g * g) * pow(1.0 + g * g - 2. * g * cost, 3. / 2.);
  return a * (b / c);
}

vec3 tonemap_uchimura2(vec3 v) {
  const float P = 1.0;  // max display brightness
  const float a = 1.9;  // contrast
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
  vec4 p22 = vec4(((texCoord)*2.0 - 1.0), 0.0, 1.0);
  vec3 dir = (invproj * p22).xyz / (invproj * p22).w;
  dir = normalize(mat3(invview) * dir);
  vec3 wi = dir;
  vec3 holdwi = wi;
  // dir = dir.xzy;
  // uint seedCam = uint(time);
  // + (vec3(rndf(seedCam),rndf(seedCam),rndf(seedCam))*2.0-1.0)*0.001;
  vec3 p = texture2D(position, texCoord).xyz;

  vec3 camera = p;
  if (texture2D(albedo, texCoord).w > 0.5) {
    //  p = camera + wi * 100.;
    return;
  }
  vec3 direc = (p - camera) / 10.;
  float accumFog = 0.;
  vec3 holdp = p;

  vec3 forgum = vec3(0.);

  vec3 n = texture2D(normal, texCoord).xyz;
  float rou = texture2D(holdinfo, texCoord).w;
  if (texture2D(position, texCoord).w > 0.5) {
    return;
  }

  float lengthOfReflection = 0.;
  float firstRough = rou;
  vec3 newdir = normalize(ldir);
  newdir = sampleSun(normalize(ldir).xzy, r, 0.002).xzy;
  vec3 newp = p + n * epsilon;
  float rough2 = rough;
  float l2 = l;
  vec3 cccc2 = cccc;
  vec3 shadow = vec3(0.);
  float shadowLength = 0.;

   bool isSub = texture2D(holdinfo, texCoord).x > 0.5;
vec3 sssv = vec3(0.);
    vec3 keepAfter = p;
bool isSK = false;
vec3 keepThething = p;
    float stepsInside = 0.;
isSub = isSub && rndf(r)< 0.6;
 vec4 wp = texture2D(watpos, texCoord);
  vec4 wn = texture2D(watnorm, texCoord);
  vec3 F23 = vec3(0.);

  if (renderSkyAndSun && ldir.y > 0.) {
    bool isSP2 = false;
    // if(!trace(newp, newdir)){
    //  if(l2 < 0.01 && l < 0.01){
    float pdf2 = max(dot(newdir, n), 0.) / 3.14159;
    vec3 brdf2 = vec3(1.) / 3.14159;
   // isSP2 = rndf(r) > firstRough;
    // if(rough2 < 0.5 && rndf(r) > rough){
   // if (isSP2) {
      pdf2 = ggx_pdf(reflect(wi, n), newdir, firstRough);
   // }
    // pdf2 = mix(, pdf2, firstRough);
    // }
    vec3 d2 = newdir;
    vec3 h = normalize(-wi + d2);
  float loh = max(dot(d2, h),0.);
    float D2 = ggx_D(reflect(wi, n), d2, firstRough);
    float G2 =
        ggx_G2(h, n, wi, d2, firstRough); // cook torrance based geometry term
     vec3 F2 =
        ggx_F(vec3(0.99 - firstRough),
              max(dot(-wi, n),0.)); // schlicks approx to the fresnel term
    // vec3 F2 = fresnelSchlick(vec3(metalAlb), vec3(1.-firstRough),
    // max(dot(-wi, n), 0.));

    vec3 specular2 =
        (D2 * G2 * F2) /
        max(4. * max(dot(-wi, n), 0.) * max(dot(d2, n), 0.), 0.0001);
    // if(firstRough < 0.5){
   // if (isSP2) {
      brdf2 = specular2 * 1.;
  //  }
    // brdf2 = mix(specular2,brdf2,firstRough);

    // }
    //brdf2 *= (1.0 + 2. * (1. - firstRough) * max(dot(d2, n), 0.));

    pdf2 = max(pdf2, 0.00001);
    
    // shadow = tt*brdf2*max(dot(newdir, n),0.)/pdf2;

    // col += (brdf2*max(dot(newdir, n),0.)/pdf2)*sunColor*sunStrength;

    vec3 Albedo = pow(texture2D(albedo, texCoord).xyz, vec3(1.2));

    Albedo = rgbtohsv(Albedo);
    Albedo.g *= 1.;
    // Albedo.r *= 1.2;
    Albedo = hsvtorgb(vec3(clamp(Albedo.x, 0., 360.), clamp(Albedo.y, 0., 1.),
                           clamp(Albedo.z, 0., 1.)));


/*
sunN = max(pow(shadowsLight, 16.), 0.)*rain3*cosT*isinshad*
    6.*specularFinal*sunCol*F2;
sunN += clamp(Albedo,0.,100.)*2.*(1.0-F2)*(1.-doMetal)*max(pow(shadowsLight, 16.), 0.)*rain3*cosT*isinshad*sunCol;
*/

vec3 sunCol = mix(vec3(0.9, 0.7, 0.5), vec3(0.9, 0.85, 0.8), max(ldir.y, 0.)) *
               pow(max(dot(normalize(ldir), vec3(0., 1., 0.)), 0.), 2.) *
               sunStrength * max(dot(newdir, n), 0.);

vec3 shad = shadowS(texture2D(position, texCoord).xyz +
                      texture2D(normal, texCoord).xyz * epsilon *2.);

shadow = brdf2*sunCol*5.*shad;
shadow += Albedo*4.*(1.-F2)*sunCol*shad;


    // col += (brdf2/pdf2)*sunColor*sunStrength*max(dot(newdir, n),0.);
   /* shadow += shadowS(texture2D(position, texCoord).xyz +
                      texture2D(normal, texCoord).xyz * 0.1) *
              (((brdf2) + (Albedo / 3.14159) * (1.0 - F2)) *
               
*/


    shadow = clamp(shadow, 0., 100.);
    
    //  }

    // }else{
    shadowLength = length(newp - camera);

    //  }
  }
  rough = firstRough;
  l = l2;
  cccc = cccc2;
  vec3 cam = p;

  float isk = 0.;
  float isk2 = 0.;

  vec3 tt = vec3(1.);
  // float epsilon = 0.02;

  vec3 secp = p;

  float lengthRefl = 0.;
  vec3 reflectedNorm = vec3(1.);
  vec3 reflectedAlb = vec3(1.);
  rough = rou;
  cccc = vec3(1.);

  bool specIS = false;
  float cosT = 1.;

  //tt *= brdf * max(dot(dir, n), 0.) / pdf;

     bool breakloop = false;
      vec3 stpp =  vec3(0.);
  bool isitskyforsss = false;
  /*
if(isSub){
    vec3 newpM = p;
    newpM -= n*epsilon*4.;
    vec3 cccm = newpM;
    bool hasexited = false;
    vec3 wid = wi;
    newpM += wid*0.01;
   // wid = ldir;
   if(rndf(r) < 0.5){
    //wid = ldir;
   }
    //if(dot(wid, n) > 0.){
    //  breakloop = true;
    //}
   // wid = angledircos(wid.xzy, r).xzy;
    bool first = true;
    vec3 mc = vec3(1.);
    float constthingy = .0944;
    for(int i = 0; i < 55; i++){
    //  if(breakloop){
    //    break;
     // }
           newpM += wid*sssStepSize;
                 float dist = map(newpM);

      if(l > 0.001){
              stpp = vec3(1.)*pow(max(exp(-constthingy*stepsInside), 0.0)*1., 2.);

        col += l*cccc*2.*stpp;
        first = false;
        break;

      }
      if(dist < epsilon*4.){
       stepsInside += 2.;
      }
      if(!issubsurface || dist > epsilon*3.){
        break;
      }
      
      if(rndf(r) < 0.35 ){
        vec3 dir2 = angledircos(wid.xzy, r).xzy;
         wid = normalize(wid + dir2);
      }
    }
    //float dist = map(newpM);
    //if(dist > epsilon*4.){
    //  hasexited = true;
   // }
    //p = ;
     newpM += wid*0.025;
    vec3 sjh = shadowS(newpM);
    if(first){
      //p = newpM+wid*0.1;
      vec3 n2 = norm(newpM+wid*0.01);
      stpp = vec3(1.)*pow(max(exp(-constthingy*stepsInside), 0.0)*1., 2.);
      //      stpp = vec3(1.)*pow(max(exp(-8.744*stepsInside*max(1.-cccc,0.)), 0.0)*1., vec3(2.));

      vec3 dir2 = angledircos(wid.xzy, r).xzy;
      //dir2 = normalize(ggx_S(wid.xzy, r, 0.24).xzy);
      //col = sjh*stpp*1.*max(dot(wid, n2),0.)*sunStrength *
        //     mix(vec3(0.9, 0.2, 0.2), vec3(0.9, 0.8, 0.7), max(ldir.y, 0.)) ;
             // tt *=  pow(texture2D(albedo, texCoord).xyz, vec3(2.2));
//tt *= vec3(0.9,0.8,0.2);
if(renderSkyAndSun){
      col += sjh.x*stpp*0.5*tt*sunStrength * max(dot(ldir, n2),0.) *
             mix(vec3(0.9, 0.4, 0.2), vec3(0.9, 0.8, 0.7), max(ldir.y, 0.));
}
      //col *= stpp;
      vec3 wpzz = vec3(0.);
        vec3 wnzz = vec3(0.);
        bool hitswat = false;
        float ss = 0.;
        if (trace(newpM, dir2, wpzz, wnzz, false, hitswat, ss,float(time)*0.04 ,viewPos)  ) {
          if (l > 0.01) {
              //col += tt * l * (cccc / 3.14159)*stpp*1.;
              //sssv = tt * l * (cccc / 3.14159)*stpp*4.;
          }
                 ivec3 newP = ivec3((newpM + 90. - floor(viewPos)));
            if (useRadCache) {
                vec3 cm = imageLoad(rcrad, newP).xyz * tt / 3.14159;
                //col += cm*stpp*1.;
                //sssv = cm*stpp;
            }

        }else if(renderSkyAndSun){
             vec3 rayDirection = dir2.xzy;
                                 rayDirection.y *= -1.;

              rayDirection = normalize(rayDirection);
             vec2 a = normalize(rayDirection.xy);
        float x = (a.y < 0.)?2.*3.14159 - acos(a.x):acos(a.x);
        x /= 2.*3.14159;
        float y = acos(rayDirection.z)/3.14159;
        vec2 TX = vec2(x,y); 
            vec3 skk = tt * texture2D(skytex, TX).rgb *
             skyStrength;
             // if(isSub){
                //    skk *= stpp*4.;
                 // }
                  col += skk*stpp*1.; 
                  //sssv = skk*stpp*4.;
        }
       
      //isitskyforsss
    }
     sssv = col*1.;
      stpp *= sjh;
    col = vec3(0.);
      /*vec3 wpzz = vec3(0.);
        vec3 wnzz = vec3(0.);
        bool hitswat = false;
        float ss = 0.;
        if (trace(p, wid, wpzz, wnzz, false, hitswat, ss,float(time)*0.04 ,viewPos)  ) {

             n = norm(p - wid * epsilon * 2.);
              p += n*epsilon*2.;
               dir = angledircos(n.xzy, r).xzy;

        }else{
          isSK = true;
           dir = wid;

        }
   

      }
*/
      //stpp = 1.;
     // col = vec3(0.9,0.,0.);
//p = keepAfter;
    //}

  //}

  /* if(i > 1){
       float t_max = max(tt.x, max(tt.y, tt.z));
       if(rndf(r) > t_max){
           break;
       }
       tt *= 1./t_max;
   }*/
  
  ivec3 prevPos = ivec3((p + 90. - floor(viewPos)));

  float reflectionLength = 0.;

  if (firstRough < 0.25 || wp.w > 0.5) {
    if (wp.w < 0.5) {
      p = camera;
      n = texture2D(normal, texCoord).xyz;
      p += n * epsilon * 3.;
      rough = firstRough;

    } else {
      p = wp.xyz + wn.xyz * epsilon * 2.;
      n = wn.xyz;
      rough = 0.025;
    }
    vec3 holdN = n;
    // dir = reflect(wi,n);
    // dir = ggx_S(reflect(wi,n).xzy,r,firstRough).xzy;
    tt = vec3(1.);
    wi = holdwi;

    // float prob = rough;
    // specIS = rndf(r) > prob;
    // if(specIS){

    dir = ggx_S(reflect(wi, n).xzy, r, rough).xzy;
    float pdf = ggx_pdf(reflect(wi, n), dir, rough);
    //}
    // pdf = mix(ggx_pdf(reflect(wi,n),dir,rough), pdf, rough);

    vec3 d = dir;
    vec3 h = normalize(-wi + d);
    float loh = max(dot(d, h),0.);
    float D = ggx_D(reflect(wi, n), d, rough);
    float G = ggx_G2(h, n, wi, d, rough); // cook torrance based geometry term

    vec3 cols = pow(texture2D(albedo, texCoord).xyz, vec3(2.2));
    if(wp.w > 0.5){
      cols = vec3(1.);

    }
    vec3 fresn = mix(cols, vec3(1.), 1.-rou);

     F23 = ggx_F(vec3(1.-rough)*fresn,
                 loh); // schlicks approx to the fresnel term

                   if (wp.w < 0.5) {
                    F23 = vec3(1.);
                   }
    // vec3 F = fresnelSchlick(vec3(metalAlb), vec3(1.-rough), max(dot(-wi, n),
    // 0.));
    vec3 specular = vec3( F23 * D * G) /
                    max(4. * max(dot(-wi, n), 0.) * max(dot(d, n), 0.), 0.0001);
    // if(specIS){
    vec3 brdf = specular * 1.;
    //}
    pdf = max(pdf, 0.0001);

    tt *= brdf * max(dot(dir, n), 0.) / pdf;

    cosT = 1.;
    vec3 reflcam = p;
    ////bool trace(inout vec3 p, vec3 d, inout vec3 watp, inout vec3 watn, bool
    /// renderWat, inout bool hitswat){
    vec3 wps = vec3(0.);
    vec3 wns = vec3(0.);
    bool hitswat = false;
    // bool trace(inout vec3 p, vec3 d, inout vec3 watp, inout vec3 watn, bool
    // renderWat, inout bool hitswat, bool stopatwat = false){
    float sss = 0.;
    if (trace(p, dir, wps, wns, removeWater, hitswat, sss,float(time)*0.04 ,viewPos ,false)) {
      cosT = max(dot(dir, n), 0.);
      reflectionLength = length(p - reflcam);
      vec3 cccc23 = cccc;

      n = norm(p - dir * epsilon * 2., float(time)*0.04);
      if(hitswat){
        n = wns;
      }
      if (mapPl(p,float(time)*0.04 ) <= 0. && removeWater) {
        //cccc *= vec3(0.2, 0.6, 0.9);
      }

      if (l > 0.01) {
        forgum += tt * l * (cccc / 3.14159);
      }
      reflectedNorm = n;

      p += n * epsilon * 3.;
      // vec3 newdir = ldir.xzy;

      vec3 newpZ = p;
      float rough2 = rough;
      float l2 = l;
           

      if (renderSkyAndSun && dot(normalize(ldir), n) > 0. ) {

        // if(!trace(newpZ, normalize(ldir))){
        //   if(l2 < 0.01 ){
        //           finalColor += GetShadow(
        //           VoxelSpaceToSceneSpace(WorldSpaceToVoxelSpace(p2
        //           )))*10.*max(dot(ldir,
        //           sunN),0.)*tt*glassColSun*(sunCol/3.14159)*max(dot(ldir,
        //           vec3(0., 1., 0.)),0.0);

        forgum +=
            shadowS(newpZ + n * 0.1) * tt * cccc23* max(dot(normalize(ldir), n), 0.) *
            sunStrength *
            mix(vec3(0.9, 0.7, 0.5), vec3(0.9, 0.85, 0.8), max(ldir.y, 0.)) *
            pow(max(dot(normalize(ldir), vec3(0., 1., 0.)), 0.),
                2.); /// max(length(p-cmmm)*length(p-cmmm),0.1);
        //  }
        //}
      }

      tt *= cccc23;
      ivec3 newP = ivec3((p + 90. - floor(viewPos)));
      if (newP.x != prevPos.x || newP.y != prevPos.y || newP.z != prevPos.z) {
        if (useRadCache) {
          forgum += imageLoad(rcrad, newP).xyz * tt ;
        }
      }

      // forgum = cccc;
    } else {
      cosT = max(dot(dir, n), 0.);

      if (renderSkyAndSun) {
        // col += tt*skyp2(dir.xzy, ldir)*skyStrength;
        // forgum += tt*sky(vec3(0.), dir.xzy, normalize(ldir).xzy)*
        // max(dot(normalize(ldir).xzy, vec3(0.,0.,1.)),0.34)*skyStrength;
        vec3 rayDirection = dir;
                            rayDirection.y *= -1.;

        rayDirection = normalize(rayDirection);
        vec2 a = normalize(rayDirection.xy);
        float x = (a.y < 0.)?2.*3.14159 - acos(a.x):acos(a.x);
        x /= 2.*3.14159;
        float y = acos(rayDirection.z)/3.14159;
        vec2 TX = vec2(x,y); 
        forgum += tt * texture2D(skytex, TX).rgb *
                  max(dot(normalize(ldir).xzy, vec3(0., 0., 1.)), 0.34) *
                  skyStrength;
                 
      }
    }
  }


  p = texture2D(position, texCoord).xyz;
n = texture2D(normal, texCoord).xyz;
  p += n * epsilon * 2.;

  vec3 cmmm = p;

  dir = angledircos(n.xzy, r).xzy;
  float pdf = max(dot(dir, n), 0.) / 3.14159;
  vec3 brdf = cccc / 3.14159;

  pdf = max(pdf, 0.001);
  float normalLength = 160.;
  ////bool trace(inout vec3 p, vec3 d, inout vec3 watp, inout vec3 watn, bool
  /// renderWat, inout bool hitswat){
  vec3 wpzz = vec3(0.);
  vec3 wnzz = vec3(0.);
  bool hitswat = false;
  float ss = 0.;
  tt = vec3(1.)*max(dot(dir, n), 0.);



  vec3 View = texture2D(position, texCoord).xyz;
  vec4 Projected = vec4(View.xyz, 1.); // + vec4(cameraOffset, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;
  uint seedCamPrev = uint(max(time, 0));
  vec2 smallOffset2 =
      ((vec2(rndf(seedCamPrev), rndf(seedCamPrev))) * 2. - 1.) / iResolution;
  // smallOffset2 = vec2(0.);

  ProjectedCoordinates -= smallOffset2 * 1.;
  
  bool outsc = false;
  // float isSpec = texture2D(reflAlb, texCoord).w;

  if (ProjectedCoordinates.x > 1. || ProjectedCoordinates.x < 0. ||
      ProjectedCoordinates.y > 1. || ProjectedCoordinates.y < 0.) {
    //ProjectedCoordinates = texCoord;
    outsc = true;
  }
  vec4 nx = texture2D(normal, texCoord);

  vec4 nDij = texture2D(prevN, ProjectedCoordinates);

  vec3 toproject = camera;


  bool outScreen = outsc;

  float depthWeight = pow(exp(-abs((nDij.w) - (nx.w)) * 0.001), 12.);

  float normalWeight = pow(max(dot(nDij.xyz, nx.xyz), 0.), 5.);
  float totalWeight = (1.0 - float(outScreen)) * normalWeight * depthWeight;

  float accumulation = texture2D(sssprev, ProjectedCoordinates).w;

  accumulation = clamp(accumulation + 1.0, 0.0, 42.);
  accumulation *= totalWeight;

  float frameWeight = (1.0 / max(1.0, accumulation));

    vec3 previousC = texture2D(den1, ProjectedCoordinates).rgb;

  
  bool tryDiffuse = true;
  if(!outsc && abs((nDij.w) - (nx.w)) < 0.1 ){
    tryDiffuse = false;
  }

  vec3 col = vec3(0.);


  //if(tryDiffuse){
    if (trace(p, dir, wpzz, wnzz, false, hitswat, ss,float(time)*0.04 ,viewPos) && !isSK   ) {
      // lengthOfReflection = length(holdp - p);
      normalLength = length(p - camera);
      vec3 kpn = vec3(0.);
            vec3 cccc2 = cccc;

      n = norm(p - dir * epsilon * 2., float(time)*0.04);
      if (mapPl(p,float(time)*0.04 ) <= 0. && removeWater) {
      // cccc *= vec3(0.2, 0.6, 0.9);
      }

      if (l > 0.01) {
        //if(isSub){
        //  col += tt * l * 4. * (cccc / 3.14159) * stpp;
      // }else{
          col += tt * l * (cccc / 3.14159);
      //  }
      }

      wi = dir;
    

      // if(i == 0){
        //if(!isSub){
            secp = p;
      // }
      //}
      reflectedAlb = cccc;
      //p += n * epsilon * 2.;
      // vec3 newdir = ldir.xzy;
      ivec3 newP = ivec3((p + 90. - floor(viewPos)));

      vec3 newpZ = p;
      float rough2 = rough;
      float l2 = l;
      if (renderSkyAndSun && dot(normalize(ldir), n) > 0. && ldir.y > 0.) {

        // if(!trace(newpZ, normalize(ldir))){
        //     if(l2 < 0.01 ){
        //           finalColor += GetShadow(
        //           VoxelSpaceToSceneSpace(WorldSpaceToVoxelSpace(p2
        //           )))*10.*max(dot(ldir,
        //           sunN),0.)*tt*glassColSun*(sunCol/3.14159)*max(dot(ldir,
        //           vec3(0., 1., 0.)),0.0);

        vec3 sjh = shadowS(newpZ + n * 0.1) *2.* cccc2 * tt * max(dot(normalize(ldir), n), 0.) *
              sunStrength *
              mix(vec3(0.9, 0.8, 0.5), vec3(0.9, 0.85, 0.8), max(ldir.y, 0.)) *
              pow(max(dot(normalize(ldir), vec3(0., 1., 0.)), 0.),
                  2.); /// max(length(p-cmmm)*length(p-cmmm),0.1);
        //    }


        //if(isSub){
        //  sjh *=  stpp*4.;
        //}
        col += sjh;
        // }
      }
  tt *= cccc2;
      // rcrad
      // if(length(p-cmmm) > 1.){
      if (newP.x != prevPos.x || newP.y != prevPos.y || newP.z != prevPos.z) {
        if (useRadCache) {
          vec3 cm = max(dot(texture2D(normal, texCoord).xyz, dir),0.)*imageLoad(rcrad, newP).xyz * tt / 3.14159;
          //if(isSub){
          // cm *= stpp*4.;
        // }
          col += cm*0.85;
        }
      }

    
      

      //}
      // col = vec3(cccc)*max(dot(n, ldir),0.);
      // tt *= cccc;

    } else {
      // vec3 skyp2(vec3 d, vec3 lig){//my code to begin with
      if (renderSkyAndSun) {

        // col += tt*skyp2(dir.xzy, ldir)*skyStrength;

        // col += tt*sky(vec3(0.), dir.xzy, normalize(ldir).xzy)*
        // max(dot(normalize(ldir).xzy, vec3(0.,0.,1.)),0.34)*skyStrength;
        //maybe remove the .xzy
        vec3 rayDirection = dir;
                            rayDirection.y *= -1.;

        rayDirection = normalize(rayDirection);
        vec2 a = normalize(rayDirection.xy);
          float x = (a.y < 0.)?2.*3.14159 - acos(a.x):acos(a.x);
          x /= 2.*3.14159;
          float y = acos(rayDirection.z)/3.14159;
          vec2 TX = vec2(x,y); 
        vec3 skk = tt * texture2D(skytex, TX).rgb  *
              skyStrength;
              // if(isSub){
                  //    skk *= stpp*4.;
                  // }
                    col += skk;
      }
      // lengthOfReflection = 1000.;

      isk = 1.;
      //if(!isSub){
            secp = p;
        //}
    }

   // col *= (1.-F23);
  //}else{

  //  col = previousC;
 // }
  float AO = 0.;
  p = camera;
  n = texture2D(normal, texCoord).xyz;
  p += n * epsilon * 2.;

  for (int i = 0; i < 7; i++) {
    p += n * 0.01 * (2.*float(i));
    float dist = map(p, float(time)*0.04);
    AO += dist;
  }
  // AO = max(1.-AO, 0.01);
  AO = max(pow(AO * 1.5, 2.), 0.);
  AO = clamp(AO, 0.1, 1.);
  //+ texture2D(colorfog, texCoord).xyz
  // col = tonemap_uchimura2(col);
  // col = pow(col, vec3(1.6))*1.8;
  //      col = pow(col, vec3(1./2.2));

  //col = sssv;
 // col = vec3( isSK?1.:0.);

//sssprev


  

  vec3 prevSSS = texture2D(sssprev, ProjectedCoordinates).xyz;
  sssv = sssv*frameWeight + prevSSS*(1.-frameWeight);
  sssTexDenoise = vec4(sssv, texture2D(holdinfo, texCoord).x);
  ssstex = vec4(vec3(sssv), accumulation);
  color = vec4(clamp(col, 0., 1000.), (tryDiffuse)?0.:1.);
  secondp = vec4(secp, isk);
  reflection2 = vec4(reflectedNorm, shadowLength);
  reflectionAlb = vec4(shadow, cosT);
  colortexFog = vec4(clamp(forgum, 0., 1000.), AO);
}