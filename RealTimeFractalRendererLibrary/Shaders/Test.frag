

layout (location = 0) out vec4 color;
layout (location = 1) out vec4 secondPos;
layout (location = 2) out vec4 reflection2;
layout (location = 3) out vec4 reflectionAlb;


//out vec4 outputColor;
in vec2 texCoord;

/*
_testShader.SetInt("position", 1);
            _testShader.SetInt("normal", 2);
            _testShader.SetInt("albedo", 3);
*/
layout(binding = 5, Rgba32f) uniform image3D rcrad; 

uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D holdinfo;
uniform sampler2D colorfog;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 invview;
uniform mat4 invproj;
uniform vec2 wh;
uniform float time;
uniform float time2;

uniform vec3 viewPos;



float PR(float cost){
return (3./(16.*3.14159))*(1.0+cost*cost);
}
float PM(float cost, float g){
float a = 3./(8.*3.14159);
float b = (1.0-g*g)*(1.0+cost*cost);
float c = (2.0+g*g)*pow(1.0+g*g-2.*g*cost, 3./2.);
return a*(b/c);
}

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
    vec3 holdwi = wi;
    //dir = dir.xzy;
    uint seedCam = uint(time);
    vec3 pos = viewPos;// + (vec3(rndf(seedCam),rndf(seedCam),rndf(seedCam))*2.0-1.0)*0.001;
    vec3 camera = pos;
    pos = texture(position, texCoord).xyz;
    if(texture(albedo, texCoord).w > 0.5){
        pos = camera + wi * 100.;
    }
    vec3 direc = (pos - camera)/10.;
    float accumFog = 0.;
    vec3 holdpos = pos;
/*
    for(int i = 0; i < 10; i++){
        vec3 currCam = camera + direc*rndf(r)*float(i+1);
        if(!trace2(currCam, ldir.xzy)){
            accumFog += 0.5*PM(max(dot(wi.xzy, ldir),0.), 0.76);
        }
    }
*/
    vec3 n = texture(normal, texCoord).xyz;
    float rou = texture(holdinfo, texCoord).x;
    if(texture(position, texCoord).w > 0.5){
        return;
    }
        vec3 col = vec3(0.);
    float epsilon = 0.02;
float lengthOfReflection = 0.;
float firstRough = rou;
                vec3 newdir = ldir.xzy;
                vec3 newpos = pos + n * epsilon;
                float rough2 = rough;
                float l2 = l;
                vec3 cccc2 = cccc;
                if(!trace(newpos, newdir)){
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
                        vec3 F2 = ggx_F(vec3(0.4), max(dot(-wi, n), 0.));//schlicks approx to the fresnel term
                        vec3 specular2 = (D2*G2*F2)/max(4.*max(dot(-wi,n),0.)*max(dot(d2,n),0.6),0.0001);
                        if(firstRough < 0.5){
                            brdf2 = specular2;
                            brdf2 *=  (1.0+2.*(1.-firstRough)*max(dot(d2,n),0.));

                        }
   
                        pdf2 = max(pdf2, 0.001);
                       // shadow = tt*brdf2*max(dot(newdir, n),0.)/pdf2;
                        col += (brdf2*max(dot(newdir, n),0.)/pdf2)*vec3(0.9,0.8,0.7)*2.5;
                    }
                     
                }

    vec3 cam = pos;

    float isk = 0.;
    float isk2 = 0.;


    vec3 tt = vec3(1.);
    //float epsilon = 0.02;
    vec3 normals = vec3(0.);
    vec3 positions = vec3(0.);
    vec3 albedos = vec3(0.);
    vec3 secp = pos;
   
    float lengthRefl = 0.;
    vec3 reflectedNorm = vec3(1.);
    vec3 reflectedAlb = vec3(1.);
    rough = rou;
    cccc = vec3(1.);


    
    for(int i = 0; i < 1; i++){
        pos += n * epsilon;

           
        dir = angledircos(n.xzy, r).xzy;
        float pdf = max(dot(dir, n),0.)/3.14159;
        vec3 brdf = cccc/3.14159;
            if(i == 0){
                if(rough < 0.5){
                    dir = ggx_S(reflect(wi,n).xzy,r,rough).xzy;
                    pdf = ggx_pdf(reflect(wi,n),dir,rough);

                }
                vec3 d = dir;
                vec3 h = normalize(-wi + d);
      
                float D=ggx_D(reflect(wi,n),d,rough);
                float G = ggx_G2(h,n,wi,d,rough);//cook torrance based geometry term
                vec3 F = ggx_F(vec3(0.4), max(dot(-wi, n), 0.));//schlicks approx to the fresnel term
                vec3 specular = (D*G*F)/max(4.*max(dot(-wi,n),0.)*max(dot(d,n),0.6),0.0001);
                if(rough < 0.5){
                    brdf = specular*4.;
                    brdf *=  (1.0+2.*(1.-rough)*max(dot(d,n),0.));
                    
                }
            }
            pdf = max(pdf, 0.001);
            tt *= brdf*max(dot(dir,n),0.)/pdf;

            if(i > 1){
                float t_max = max(tt.x, max(tt.y, tt.z));
                if(rndf(r) > t_max){
                    break;
                }
                tt *= 1./t_max;
            }
        if(trace(pos,dir)){
            lengthOfReflection = length(holdpos - pos);
            if(l > 0.01){
                col += tt*l*cccc;
                break;
            }

            //rcrad
            ivec3 newP = ivec3((pos+40.-floor(viewPos))*.9);
            col += imageLoad(rcrad, newP).xyz;
            //col = vec3(cccc)*max(dot(n, ldir),0.);
           // tt *= cccc;
            
        }else{
            //vec3 skyp2(vec3 d, vec3 lig){//my code to begin with
            
          col += tt*skyp2(dir.xzy, ldir);
        lengthOfReflection = 1000.;

            if(i == 0){
                isk = 1.;
            }
            break;
        }
        float rough22 = rough;
                float l22 = l;
                vec3 cccc22 = cccc;
        n = norm(pos-dir*epsilon);
        rough = rough22;
                l = l22;
                cccc = cccc22;
        wi = dir;
        if(i < 2){     
                secp = pos;
                reflectedNorm = n;
                reflectedAlb = cccc;

                vec3 newdir = ldir.xzy;
                vec3 newpos = pos + n * epsilon;
                float rough2 = rough;
                float l2 = l;
                vec3 cccc2 = cccc;
                if(!trace(newpos, newdir)){
                    if(l2 < 0.01){

                         col += tt*max(dot(newdir, n),0.)*2.;
                    }
                }

                rough = rough2;
                l = l2;
                cccc = cccc2;
            }
    }

//+ texture(colorfog, texCoord).xyz
    color = vec4(col , rou);
    secondPos = vec4(secp, isk);
    reflection2 = vec4(reflectedNorm, accumFog);
    reflectionAlb = vec4(holdpos + holdwi*lengthOfReflection, 1.);
}