



layout (location = 0) out vec4 weigths;
layout (location = 1) out vec4 outrad;

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



float lum(vec3 c){
return sqrt( 0.299*c.x*c.x + 0.587*c.y*c.y + 0.114*c.z*c.z );
}

void main()
{
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
if(texture(albedo, texCoord).w > 0.5 || texture(position, texCoord).w > 0.5){
    return;
}

            float roughness = texture(color, texCoord).w;
// if(roughness < 0.5){
//     weigths = vec4(1., 1., 1., 0.);
// 	//outrad = vec4(clamp(texture(tempL, ProjectedCoordinates).xyz,0., 100.) , 1.0);
//     outrad = vec4(texture(color, texCoord).xyz, 1.);
//  return;   
// }

 vec2 fragCoord = texCoord*wh;
    uint r = uint(uint(fragCoord.x) * uint(1973) + uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) | uint(1);

            vec3 cameraOffset = viewPos - lastViewPos;
            vec3 View = texture(position, texCoord).xyz;
            vec4 Projected = vec4(View.xyz, 1.);// + vec4(cameraOffset, 0.);
            Projected = prevview * Projected;
            Projected = prevproj * Projected;
            Projected /= Projected.w;
            vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;



if(roughness < 0.5){
    //tempAccum = vec4(col, 0.);
    //den1 = vec4(col, 0.);  
    //var1 = vec4(1.);
 //return;   
 View = texture(reflAlb, texCoord).xyz;
   Projected = vec4(View.xyz, 1.);// + vec4(cameraOffset, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
   ProjectedCoordinates = Projected.xy * 0.5 + 0.5;


 
}

            vec3 Rs = texture(prevW, ProjectedCoordinates).xyz;
            vec3 LO = texture(prevL, ProjectedCoordinates).xyz;

			//vec3 LO = texelFetch(colortex13,ivec2(iResolution* ProjectedCoordinates),0).xyz;
            vec3 InitSampleLO = texture(color, texCoord).xyz;

			float RM = max(Rs.x,1.);
			float RW = Rs.z;
			float Rw = Rs.y;

            vec4 currN = texture(normal, texCoord);
            vec4 prevN = texture(prevN, ProjectedCoordinates);
            bool disc = false;
            if (dot(normalize(currN.xyz),normalize(prevN.xyz)) < 0.39) {
                disc = true;
            }
            if (abs(currN.w - prevN.w) > 1.1)
            {
                disc = true;
            }
            if(ProjectedCoordinates.x > 1. || ProjectedCoordinates.x < 0. || ProjectedCoordinates.y > 1. || ProjectedCoordinates.y < 0. || disc){
                RM = 0.;
                RW = 0.;
                Rw = 0.;
                LO =  texture(tempL, ProjectedCoordinates).xyz;
              // LO = InitSampleLO;
            }

            const int maxiterations = 9;
            //std::vector<RESERVOIR> QN;
            //QN.push_back(Rs);
            //Sample init = InitialSampleBuffer[index];
            float Z = RM;
            vec2 TexCoords = texCoord;
            vec2 iResolution = wh*1.5;
            for (int s = 0; s < maxiterations; s++) {
               
                float radius = (30.0*sqrt(sqrt(roughness)))*rndf(r);
                float angle = rndf(r)*2.0*3.14159;
                float x =  (cos(angle) * radius);
                float y =  (sin(angle) * radius);
             
               	vec2 fincords = (TexCoords*iResolution + vec2(x,y)) / iResolution;
                vec2 finn = (TexCoords*iResolution + vec2(x,y)) / iResolution;
                if(texture(albedo, finn).w > 0.5 || texture(position, finn).w > 0.5){
                    continue;
                }
                vec3 R = texture(tempW, finn).xyz;
                //vec3 R = texelFetch(colortex4,ivec2(iResolution* finn),0).xyz;
                
                vec3 RnLo = texture(tempL, finn).xyz;
			    //vec3 RnLo = texelFetch(colortex6,ivec2(iResolution* finn),0).xyz;
                //vec4 psandisk = texelFetch(colortex9,ivec2(iResolution* finn),0).xyzw;
                vec4 psandisk = texture(temppos, finn);
                //vec3 ns = texelFetch(colortex5,ivec2(iResolution* finn),0).xyz;

			    float RnM = max(R.x,1.);
			    float RnW = R.z;
			    float Rnw = R.y;
                


		        //vec4 info = vec4(texelFetch(colortex0, ivec2(iResolution*TexCoords),0).xyz, 
                //texelFetch(colortex1, ivec2(iResolution*TexCoords*0.5),0).w);
                //vec4 info2 = vec4(texelFetch(colortex0, ivec2(iResolution*finn),0).xyz, 
                //texelFetch(colortex1, ivec2(iResolution*finn*0.5),0).w);
                vec4 info = texture(normal, TexCoords);
                vec4 info2 = texture(normal, finn);

                if (dot(normalize(info.xyz),normalize(info2.xyz)) < 0.926) {
                    continue;
                }
                if ( (info.w) > 1.1 * (info2.w) || (info.w) < 0.9 * (info2.w))
                {
                   continue;
                }
                //vec3 pporigin = SceneSpaceToVoxelSpace(texelFetch(colortex2, ivec2(iResolution*TexCoords),0).xyz);
                vec3 pporigin = texture(position, TexCoords).xyz;
                //vec3 pporigin2 = SceneSpaceToVoxelSpace(texelFetch(colortex2, ivec2(iResolution*finn),0).xyz);
                float pq = lum(RnLo);

                vec3 initxv = pporigin+normalize(info.xyz)*0.01;
                vec3 Rnxs = psandisk.xyz;


                if(length(Rnxs - initxv) > 1. && psandisk.w < 0.5){
                    vec3 d2 = normalize(Rnxs - initxv);
                    vec3 p2 = initxv;
                    if(trace(p2, d2)){ 
                        if(length(p2 - Rnxs)>0.5 && l < 0.01){
                            pq = 0.;
                        }
                    }
                }


        if(roughness > 0.5){
            Rw /= max(RM,0.001);
   	    	RM = min(RM, 500.); 
   		    Rw *= max(RM,0.001);
        }else{
            Rw /= max(RM,0.001);    
   	    	RM = min(RM, 500.); 
   		    Rw *= max(RM,0.001);
        }
        float M0 = RM;
		float upd = pq * RnW * max(RnM,0.001);
		Rw = Rw + upd;
		RM = M0 + RnM;
		//Rw /= max(RM,1.);
   		//RM = min(RM, 500.); 
   		//Rw *= RM;
        if(rndf(r) <= (upd/max(Rw,0.000001))){
        	LO = RnLo;

		}
	//	RM = RM + 1.0;
		Z += RnM;
            }
              if (Z*lum(LO) < 0.001) {
       	RW = 0.0;
    }
    else {
    	RW = Rw / (Z * lum(LO));
    } 
         
            weigths = vec4(RM, Rw, RW, 0.);
			outrad = vec4(clamp(LO,0., 100.) , 1.0);

    //vec3 col = texture(color, texCoord).xyz;
    //outputColor = vec4(col, 1.);
}