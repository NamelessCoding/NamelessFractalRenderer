

layout (location = 0) out vec4 position;
layout (location = 1) out vec4 normal;
layout (location = 2) out vec4 albedo;
layout (location = 3) out vec4 holdinfo;
//layout (location = 4) out vec4 reflectionAlb;
layout (location = 4) out vec4 colorfog;


//out vec4 outputColor;
in vec2 texCoord;

//uniform sampler2D texture0;

layout(binding = 0, Rgba32f) uniform image3D rcpos; 
layout(binding = 1, Rgba32f) uniform image3D rcnorm; 

uniform mat4 view;
uniform mat4 projection;
uniform mat4 invview;
uniform mat4 invproj;
uniform vec2 wh;
uniform float time;
uniform float time2;

uniform vec3 viewPos;
uniform vec3 lastViewPos;


void main()
{
    vec2 fragCoord = texCoord*wh;
    uint r = uint(uint(fragCoord.x) * uint(1973) + uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) | uint(1);
    vec2 iResolution = wh*1.5;
    vec2 smallOffset = ((vec2(rndf(r), rndf(r)))*2.-1.)/iResolution;

    vec4 p22 = vec4((texCoord) * 2.0 - 1.0, 0.0, 1.0);
    vec3 dir =
	  (invproj * p22).xyz / (invproj * p22).w;
    dir = normalize(mat3(invview) * dir);
    vec3 wi = dir;
    //dir = dir.xzy;
    uint seedCam = uint(time);
    vec3 pos = viewPos;// + (vec3(rndf(seedCam),rndf(seedCam),rndf(seedCam))*2.0-1.0)*0.001;
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
    float epsilon = 0.02;
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
    for(int i = 0; i < 1; i++){
        if(trace(pos,dir)){
            float rough22 = rough;
                float l22 = l;
                vec3 cccc22 = cccc;
            vec3 n = norm(pos-dir*epsilon);
            rough = rough22;
                l = l22;
                cccc = cccc22;
            if(i == 0){
                keepk = k;
                positions = pos;
                normals = n;
                albedos = cccc;
                Depth = length(pos-cam);
                secp = pos;
                firstRough = rough;
                cccc = vec3(1.);
                vec3 newdir = ldir.xzy;
                vec3 newpos = pos + n * epsilon;
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
                        float G2 = ggx_G2(h,n,wi,d2,firstRough);//cook torrance based geometry term
                        vec3 F2 = ggx_F(vec3(0.04), max(dot(-wi, n), 0.));//schlicks approx to the fresnel term
                        vec3 specular2 = (D2*G2*F2)/max(4.*max(dot(-wi,n),0.)*max(dot(d2,n),0.6),0.0001);
                        if(firstRough < 0.5){
                            brdf2 = specular2;
                            brdf2 *=  (1.0+2.*(1.-firstRough)*max(dot(d2,n),0.));

                        }
   
                        pdf2 = max(pdf2, 0.001);
                        shadow = tt*brdf2*max(dot(newdir, n),0.)/pdf2;
                        //col += (tt*brdf2*max(dot(newdir, n),0.)/pdf2)*vec3(0.9,0.8,0.7)*0.5;
                    }
                     
                }*/
                rough = rough2;
                l = l2;
                cccc = cccc2;
                
            }

           
            if(l > 0.01){
                if(i == 0){
                    isEM = 1.0;
                }
                col += tt*l*cccc;
                break;
            }
            //col = vec3(cccc)*max(dot(n, ldir),0.);
            pos += n * epsilon;

           
            dir = angledircos(n.xzy, r).xzy;
            float pdf = max(dot(dir, n),0.)/3.14159;
            vec3 brdf = cccc/3.14159;
            
            
        }else{
            //vec3 skyp2(vec3 d, vec3 lig){//my code to begin with
            
           //col += 

           

            if(i == 0){
                keepk = k;
                isk2 = 1.0;
                albedos = tt*skyp2(dir.xzy, ldir);
               // albedos = vec3(1.);
                Depth = 100.;
            }
        }

    }
//rcnorm
vec3 delta =  lastViewPos - viewPos;
    imageStore(rcpos, ivec3((positions+40.0 - floor(viewPos))*.9), vec4(positions, 1.));
    imageStore(rcnorm, ivec3((positions+40.0 - floor(viewPos))*.9), vec4(normals, 1.));


    position = vec4(positions,isEM);
    normal = vec4(normals, Depth);
    albedo = vec4(albedos, isk2);
    holdinfo = vec4(firstRough, firstRough, firstRough, keepk);
    //reflectionAlb = vec4(clamp(shadow,0., 100.), 1.);

}