

layout (location = 0) out vec4 color;
layout (location = 1) out vec4 secondp;
layout (location = 2) out vec4 reflection2;
layout (location = 3) out vec4 reflectionAlb;
layout (location = 4) out vec4 colortexFog;
layout (location = 5) out vec4 colortexFogp;

//colortexFogp
//colortexFog

//out vec4 outputColor;
in vec2 texCoord;

/*
_testShader.SetInt("position ", 1);
            _testShader.SetInt("normal", 2);
            _testShader.SetInt("albedo", 3);
*/
layout(binding = 5, Rgba16f) uniform image3D rcrad; 

uniform sampler2D position ;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D holdinfo;
uniform sampler2D colorfog;
uniform sampler2D skytex;
uniform sampler2D sunTex;
uniform sampler2D watpos;
uniform sampler2D watnorm;


uniform mat4 view;
uniform mat4 projection;
uniform mat4 invview;
uniform mat4 invproj;
uniform vec2 wh;
uniform float time;
uniform float time2;

uniform vec3 viewPos;

uniform vec3 ldir;
uniform mat4 lightproj;
uniform mat4 lightview;
uniform vec3 lpos;


vec3 shadowS(vec3 pos) {
    vec4 sspace = lightproj * lightview * vec4(pos, 1.);
            sspace.xyz /= sspace.w;
    
    vec3 coords = sspace.xyz * 0.5 + 0.5;
    
    
    //float currDepth = texelFetch(sunTex, ivec2(2048.*coords), 0).r;
    float currDepth = texture2D(sunTex, coords.xy).r;
    //vec3 currPos = texture2D(sunTex, coords.xy).xyz;
    vec3 col = vec3(1.);
    if(coords.z > currDepth + 0.001 && coords.x >= 0. && coords.x <= 1. && coords.y >= 0. && coords.y <= 1.){
        col = vec3(0.);
    }
    
    if(length(pos.xz - viewPos.xz) > 90.){
      //  col = vec3(1.);
    }

    return col;
}
float PR(float cost){
return (3./(16.*3.14159))*(1.0+cost*cost);
}
float PM(float cost, float g){
float a = 3./(8.*3.14159);
float b = (1.0-g*g)*(1.0+cost*cost);
float c = (2.0+g*g)*pow(1.0+g*g-2.*g*cost, 3./2.);
return a*(b/c);
}

vec3 tonemap_uchimura2(vec3 v)
{
    const float P = 1.0;  // max display brightness
    const float a = 1.9;  // contrast
    const float m = 0.1; // linear section start
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



vec3 rgbtohsv(vec3 col){
float cmax = max(max(col.x, col.y), col.z);
float cmin = min(min(col.x, col.y), col.z);
float delta = cmax - cmin;
float H = 0.;
if(delta < 0.0001){
    H = 0.;
}else if(cmax == col.x){
    H = 60. * mod((col.g - col.b)/delta, 6.);
}else if(cmax == col.g){
    H = 60. * ((col.b - col.r)/delta + 2.);
}else if(cmax == col.b){
    H = 60. * ((col.r - col.g)/delta + 4.);
}
float S = 0.;
if(cmax > 0.){
    S = delta/cmax;
}
float V = cmax;

return vec3(clamp(H,0.,360.),clamp(S, 0., 1.),clamp(V, 0., 1.));
}

vec3 hsvtorgb(vec3 hsv){
    
    float C = hsv.b * hsv.g;
    float X = C * (1.- abs(mod(hsv.r / 60., 2.)-1.));
    float m = hsv.b - C;

    vec3 rgb = vec3(0.);
    float H = hsv.r;
    if(H < 60.){
        rgb = vec3(C, X, 0.);
    }else if(H < 120.){
        rgb = vec3(X, C, 0.);
    }else if(H < 180.){
        rgb = vec3(0., C, X);
    }else if(H < 240.){
        rgb = vec3(0., X, C);
    }else if(H < 300.){
        rgb = vec3(X, 0., C);
    }else if(H < 360.){
        rgb = vec3(C, 0., X);
    }

    return rgb+m;
}

vec4 rgbtoCMYK(vec3 col){
    float K = max(1.-max(col.r, max(col.g, col.b)),0.);
    float C = (1.-col.r-K) / (1.-K);
    float M = (1.-col.g - K)/(1.-K);
    float Y = (1.-col.b-K)/(1.-K);
    return vec4(C,M,Y,K);
}

vec3 CMYKtorgb(vec4 col){
    col = clamp(col, 0., 1.);
float R = (1.-col.x)*(1.-col.w);
float G = (1.-col.g)*(1.-col.w);
float B = (1.-col.b)*(1.-col.w);
return vec3(R,G,B);
}

void main()
{
    vec2 fragCoord = texCoord*wh;
    uint r = uint(uint(fragCoord.x) * uint(1973) + uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) | uint(1);
    vec2 iResolution = wh;
   // vec2 smallOffset = ((vec2(rndf(r), rndf(r)))*2.-1.)/iResolution;
/*
vec2 iResolution = wh;
    uint seedCam = uint(time);
    vec2 smallOffset = ((vec2(rndf(seedCam), rndf(seedCam)))*2.-1.)/iResolution;
//+jitter()
    vec4 p22 = vec4(((texCoord+smallOffset) * 2.0 - 1.0), 0.0, 1.0);

*/
    uint seedCam = uint(time);
    vec2 smallOffset = ((vec2(rndf(seedCam), rndf(seedCam)))*2.-1.)/iResolution;
    vec4 p22 = vec4(((texCoord) * 2.0 - 1.0), 0.0, 1.0);
    vec3 dir =
	  (invproj * p22).xyz / (invproj * p22).w;
    dir = normalize(mat3(invview) * dir);
    vec3 wi = dir;
    vec3 holdwi = wi;
    //dir = dir.xzy;
   // uint seedCam = uint(time);
    // + (vec3(rndf(seedCam),rndf(seedCam),rndf(seedCam))*2.0-1.0)*0.001;
        vec3 p = texture2D(position , texCoord).xyz;

    vec3 camera = p;
    if(texture2D(albedo, texCoord).w > 0.5){
      //  p = camera + wi * 100.;
              return;

    }
    vec3 direc = (p - camera)/10.;
    float accumFog = 0.;
    vec3 holdp = p;

    vec3 forgum = vec3(0.);
/*
    for(int i = 0; i < 10; i++){
        vec3 currCam = camera + direc*rndf(r)*float(i+1);
        if(!trace2(currCam, ldir.xzy)){
            accumFog += 0.5*PM(max(dot(wi.xzy, ldir),0.), 0.76);
        }
    }
// */
// vec3 fogtt = vec3(1.);
// vec3 secpFog = camera;
// float isLight = 1.;
//     if(traceVolume(camera, holdwi, r, fogtt, secpFog, isLight)){
//             forgum += cccc*l*fogtt;
//     }
//colortexFogp = vec4(secpFog, isLight);
    vec3 n = texture2D(normal, texCoord).xyz;
    float rou = texture2D(holdinfo, texCoord).w;
    if(texture2D(position , texCoord).w > 0.5){
        return;
    }
    
        vec3 col = vec3(0.);
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
                if(renderSkyAndSun && ldir.y > 0.){
                    bool isSP2 = false;
                    //if(!trace(newp, newdir)){
                      //  if(l2 < 0.01 && l < 0.01){
                            float pdf2 = max(dot(newdir, n),0.)/3.14159;
                            vec3 brdf2 = vec3(1.)/3.14159;
                            isSP2 = rndf(r) > firstRough;
                            //if(rough2 < 0.5 && rndf(r) > rough){
                                if(isSP2){
                                    pdf2 = ggx_pdf(reflect(wi,n),newdir,firstRough);
                                }
                                //pdf2 = mix(, pdf2, firstRough);
                           // }
                            vec3 d2 = newdir;
                            vec3 h = normalize(-wi + d2);
        
                            float D2=ggx_D(reflect(wi,n),d2,firstRough);
                            float G2 = ggx_G2(h,n,wi,d2,firstRough);//cook torrance based geometry term
                            vec3 F2 = ggx_F(vec3(0.99-firstRough), max(dot(-wi, n), 0.));//schlicks approx to the fresnel term
                            //vec3 F2 = fresnelSchlick(vec3(metalAlb), vec3(1.-firstRough), max(dot(-wi, n), 0.));

                            vec3 specular2 = (D2*G2*F2)/max(4.*max(dot(-wi,n),0.)*max(dot(d2,n),0.),0.0001);
                           // if(firstRough < 0.5){
                                if(isSP2){
                                    brdf2 = specular2*1.;
                                }
                               // brdf2 = mix(specular2,brdf2,firstRough);

                           // }
                                    brdf2 *=  (1.0+2.*(1.-firstRough)*max(dot(d2,n),0.));

                            pdf2 = max(pdf2, 0.001);
                        // shadow = tt*brdf2*max(dot(newdir, n),0.)/pdf2;
                        
                            //col += (brdf2*max(dot(newdir, n),0.)/pdf2)*sunColor*sunStrength;

                            vec3 Albedo = pow(texture2D(albedo, texCoord).xyz, vec3(2.2));

                            Albedo = rgbtohsv(Albedo);
                            Albedo.g *= 1.;
                            //Albedo.r *= 1.2;
                            Albedo = hsvtorgb(vec3(clamp(Albedo.x,0.,360.),clamp(Albedo.y, 0., 1.),clamp(Albedo.z, 0., 1.)));


                           // col += (brdf2/pdf2)*sunColor*sunStrength*max(dot(newdir, n),0.);
                            shadow += shadowS(texture2D(position, texCoord).xyz+texture2D(normal, texCoord).xyz*0.1)*(((brdf2) + (Albedo/3.14159)*(1.0-F2))*mix(vec3(0.9, 0.7, 0.5), vec3(0.9,0.85, 0.8), max(ldir.y,0.))*pow(max(dot(normalize(ldir), vec3(0., 1., 0.)),0.),3.)*sunStrength*max(dot(newdir, n),0.));
                            shadow = clamp(shadow, 0., 10.);
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
    //float epsilon = 0.02;

    vec3 secp = p;
   
    float lengthRefl = 0.;
    vec3 reflectedNorm = vec3(1.);
    vec3 reflectedAlb = vec3(1.);
    rough = rou;
    cccc = vec3(1.);

    
    bool specIS = false;
    float cosT = 1.;
            p += n * epsilon*2.;

            vec3 cmmm = p;
    
           
        dir = angledircos(n.xzy, r).xzy;
        float pdf = max(dot(dir, n),0.)/3.14159;
        vec3 brdf = cccc/3.14159;
                /*float prob = rough;
                specIS = rndf(r) > prob;
                if(specIS){
                    dir = ggx_S(reflect(wi,n).xzy,r,rough).xzy;
                    cosT = max(dot(dir, n),0.);
                    pdf = ggx_pdf(reflect(wi,n),dir,rough);
                }
                //pdf = mix(ggx_pdf(reflect(wi,n),dir,rough), pdf, rough);

                vec3 d = dir;
                vec3 h = normalize(-wi + d);
      
                float D=ggx_D(reflect(wi,n),d,rough);
                float G = ggx_G2(h,n,wi,d,rough);//cook torrance based geometry term
                vec3 F = ggx_F(vec3(1.-rough), max(dot(-wi, n), 0.));//schlicks approx to the fresnel term
                //vec3 F = fresnelSchlick(vec3(metalAlb), vec3(1.-rough), max(dot(-wi, n), 0.));
                vec3 specular = (D*G*F)/max(4.*max(dot(-wi,n),0.)*max(dot(d,n),0.),0.0001);
                if(specIS){
                    brdf = specular*1.;
                }
                //brdf = mix(specular*3., brdf, rough);

                                  //  brdf *=  (1.0+2.*(1.-rough)*max(dot(d,n),0.));

            
*/
            pdf = max(pdf, 0.001);
            tt *= brdf*max(dot(dir,n),0.)/pdf;

           /* if(i > 1){
                float t_max = max(tt.x, max(tt.y, tt.z));
                if(rndf(r) > t_max){
                    break;
                }
                tt *= 1./t_max;
            }*/
        ivec3 prevPos = ivec3((p+90.-floor(viewPos)));
        float normalLength = 160.;
        ////bool trace(inout vec3 p, vec3 d, inout vec3 watp, inout vec3 watn, bool renderWat, inout bool hitswat){
  vec3 wpzz = vec3(0.);
    vec3 wnzz = vec3(0.);
  bool hitswat = false;
        if(trace(p,dir, wpzz, wnzz, false, hitswat)){
               // lengthOfReflection = length(holdp - p);
                normalLength = length(p-camera);

            
            vec3 kpn = vec3(0.);
            n = norm(p-dir*epsilon*2.);
            if(mapPl(p) < 0.){
                cccc *= vec3(0.2, 0.6, 0.9);
            }

            if(l > 0.01){
                col += tt*l*(cccc/3.14159);
            }
       
        wi = dir;
                    tt *= cccc;

                //if(i == 0){
                 secp = p;
                //}
                reflectedAlb = cccc;
                p += n * epsilon*3.;
                //vec3 newdir = ldir.xzy;
                ivec3 newP = ivec3((p+90.-floor(viewPos)));

                vec3 newpZ = p ;
                float rough2 = rough;
                float l2 = l;
                vec3 cccc2 = cccc;
                    if(renderSkyAndSun && dot(normalize(ldir), n) > 0. && length(p-cmmm) > 1. && ldir.y > 0. ){

                       // if(!trace(newpZ, normalize(ldir))){
                       //     if(l2 < 0.01 ){
//           finalColor += GetShadow( VoxelSpaceToSceneSpace(WorldSpaceToVoxelSpace(p2 )))*10.*max(dot(ldir, sunN),0.)*tt*glassColSun*(sunCol/3.14159)*max(dot(ldir, vec3(0., 1., 0.)),0.0);

                                col += shadowS(newpZ + n *0.1)*tt*max(dot(normalize(ldir), n),0.)*sunStrength*mix(vec3(0.9, 0.8, 0.5), vec3(0.9,0.85, 0.8), max(ldir.y,0.))*pow(max(dot(normalize(ldir), vec3(0., 1., 0.)),0.),3.);///max(length(p-cmmm)*length(p-cmmm),0.1);
                        //    }
                       // }
                    }
      
            
            //rcrad
            //if(length(p-cmmm) > 1.){
                        if(newP.x != prevPos.x || newP.y != prevPos.y || newP.z != prevPos.z){
                            if(useRadCache){
                                col += imageLoad(rcrad, newP).xyz*tt/3.14159;
                            }
                        }
            //}
            //col = vec3(cccc)*max(dot(n, ldir),0.);
           // tt *= cccc;
            
        }else{
            //vec3 skyp2(vec3 d, vec3 lig){//my code to begin with
                            if(renderSkyAndSun){

                            //col += tt*skyp2(dir.xzy, ldir)*skyStrength;

                            //col += tt*sky(vec3(0.), dir.xzy, normalize(ldir).xzy)* max(dot(normalize(ldir).xzy, vec3(0.,0.,1.)),0.34)*skyStrength;
                            vec3 rayDirection = dir.xzy;
                            rayDirection = normalize(rayDirection);
                            vec2 TX = vec2((atan(rayDirection.z, rayDirection.x) / 6.283185307179586476925286766559) + 0.5, acos(rayDirection.y) / 3.1415926535897932384626433832795);
                            col += tt*texture2D(skytex, TX).rgb* max(dot(normalize(ldir).xzy, vec3(0.,0.,1.)),0.34)*skyStrength;

                            }
        //lengthOfReflection = 1000.;

                isk = 1.;
                secp = p;
           
        }
       
 
           float reflectionLength = 0.;
    vec4 wp = texture2D(watpos, texCoord);
    vec4 wn = texture2D(watnorm, texCoord);

    if(firstRough < 0.25 || wp.w > 0.5){
        if( wp.w < 0.5){
            p = camera;
            n = texture2D(normal, texCoord).xyz;
            p+=n*epsilon*3.;
            rough = firstRough;

        }else{
            p = wp.xyz + wn.xyz*epsilon*2.;
            n = wn.xyz;
            rough = 0.001;

        }
        //dir = reflect(wi,n);
        //dir = ggx_S(reflect(wi,n).xzy,r,firstRough).xzy;
        tt = vec3(1.);
         wi = holdwi;

                //float prob = rough;
                //specIS = rndf(r) > prob;
                //if(specIS){
                    dir = ggx_S(reflect(wi,n).xzy,r,rough).xzy;
                    cosT = max(dot(dir, n),0.);
                    pdf = ggx_pdf(reflect(wi,n),dir,rough);
                //}
                //pdf = mix(ggx_pdf(reflect(wi,n),dir,rough), pdf, rough);

                vec3 d = dir;
                vec3 h = normalize(-wi + d);
      
                float D=ggx_D(reflect(wi,n),d,rough);
                float G = ggx_G2(h,n,wi,d,rough);//cook torrance based geometry term
                vec3 F = ggx_F(vec3(1.-rough), max(dot(-wi, n), 0.));//schlicks approx to the fresnel term
                //vec3 F = fresnelSchlick(vec3(metalAlb), vec3(1.-rough), max(dot(-wi, n), 0.));
                vec3 specular = (D*G*F)/max(4.*max(dot(-wi,n),0.)*max(dot(d,n),0.),0.0001);
                //if(specIS){
                    brdf = specular*1.;
                //}
        pdf = max(pdf, 0.0001);

        tt *= brdf*max(dot(dir,n),0.)/pdf;

        cosT = 1.;
        vec3 reflcam = p;
           ////bool trace(inout vec3 p, vec3 d, inout vec3 watp, inout vec3 watn, bool renderWat, inout bool hitswat){
  vec3 wps = vec3(0.);
    vec3 wns = vec3(0.);
  bool hitswat = false;
  //bool trace(inout vec3 p, vec3 d, inout vec3 watp, inout vec3 watn, bool renderWat, inout bool hitswat, bool stopatwat = false){

        if(trace(p,dir, wps, wns, removeWater, hitswat, true)){
            cosT = max(dot(dir, n),0.);
            reflectionLength = length(p - reflcam);
            
            n = norm(p-dir*epsilon*2.);
            if(mapPl(p) < 0.){
                cccc *= vec3(0.2, 0.6, 0.9);
            }

            if(l > 0.01){
                forgum += tt*l*(cccc/3.14159);
            }
            reflectedNorm = n;

            p += n * epsilon*3.;
                //vec3 newdir = ldir.xzy;

                vec3 newpZ = p ;
                float rough2 = rough;
                float l2 = l;
                vec3 cccc2 = cccc;
                    if(renderSkyAndSun && dot(normalize(ldir), n) > 0. && length(p-cmmm) > 1. ){

                        //if(!trace(newpZ, normalize(ldir))){
                         //   if(l2 < 0.01 ){
//           finalColor += GetShadow( VoxelSpaceToSceneSpace(WorldSpaceToVoxelSpace(p2 )))*10.*max(dot(ldir, sunN),0.)*tt*glassColSun*(sunCol/3.14159)*max(dot(ldir, vec3(0., 1., 0.)),0.0);

                                forgum += shadowS(newpZ + n *0.1)*tt*max(dot(normalize(ldir), n),0.)*sunStrength*mix(vec3(0.9, 0.7, 0.5), vec3(0.9,0.85, 0.8), max(ldir.y,0.))*pow(max(dot(normalize(ldir), vec3(0., 1., 0.)),0.),3.);///max(length(p-cmmm)*length(p-cmmm),0.1);
                          //  }
                        //}
                    }
      

            tt *= cccc;

            ivec3 newP = ivec3((p+90.-floor(viewPos)));
            if(newP.x != prevPos.x || newP.y != prevPos.y || newP.z != prevPos.z){
                if(useRadCache){
                    forgum += imageLoad(rcrad, newP).xyz*tt/3.14159;
                }
            }

            
           // forgum = cccc;
        }else{
            cosT = max(dot(dir, n),0.);

            if(renderSkyAndSun){
                            //col += tt*skyp2(dir.xzy, ldir)*skyStrength;
                //forgum += tt*sky(vec3(0.), dir.xzy, normalize(ldir).xzy)* max(dot(normalize(ldir).xzy, vec3(0.,0.,1.)),0.34)*skyStrength;
                vec3 rayDirection = dir.xzy;
                rayDirection = normalize(rayDirection);
                vec2 TX = vec2((atan(rayDirection.z, rayDirection.x) / 6.283185307179586476925286766559) + 0.5, acos(rayDirection.y) / 3.1415926535897932384626433832795);
                forgum += tt*texture2D(skytex, TX).rgb* max(dot(normalize(ldir).xzy, vec3(0.,0.,1.)),0.34)*skyStrength;
            }
        }

    }

    float AO = 0.;   
    p = camera;
    n = texture2D(normal, texCoord).xyz;
    p+=n*epsilon*2.;

    for(int i = 0; i < 7; i++){
        p += n*epsilon*(2.+float(i));
        float dist = map(p);
        AO += dist;
    }
//AO = max(1.-AO, 0.01);
AO = max(pow(AO*3., 2.),0.);
AO = clamp(AO, 0., 1.);
//+ texture2D(colorfog, texCoord).xyz
//col = tonemap_uchimura2(col);
        //col = pow(col, vec3(1.6))*1.8;
  //      col = pow(col, vec3(1./2.2));
    color = vec4(clamp(col,0.,10.) , normalLength);
    secondp = vec4(secp, isk);
    reflection2 = vec4(reflectedNorm, shadowLength);
    reflectionAlb = vec4(shadow, cosT);
    colortexFog = vec4(clamp(forgum,0., 10.), AO);
}