#version 330


layout (location = 0) out vec4 weigths;
layout (location = 1) out vec4 outrad;
layout (location = 2) out vec4 outpos;


in vec2 texCoord;

uniform sampler2D color;
uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D secondpos;
uniform sampler2D prevW;
uniform sampler2D prevL;
uniform sampler2D prevP;
uniform sampler2D prevN;
uniform sampler2D reflAlb;
uniform sampler2D prevPosition;

/*
_TemporalRestirShader.SetInt("prevW", 5);
            _TemporalRestirShader.SetInt("prevL", 6);
            _TemporalRestirShader.SetInt("prevP", 7);
             */


uniform mat4 view;
uniform mat4 projection;
uniform mat4 invview;
uniform mat4 invproj;
uniform mat4 prevview;
uniform mat4 prevproj;
uniform vec2 wh;
uniform float time;
uniform vec3 viewPos;
uniform vec3 lastViewPos;

//NOT MY CODE///////////////
uint wang_hash(inout uint seed)
{
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}
 
float rndf(inout uint state)
{
    return float(wang_hash(state)) / 4294967296.0;
}
///////////////////////////



float lum(vec3 c){
return sqrt( 0.299*c.x*c.x + 0.587*c.y*c.y + 0.114*c.z*c.z );
}



vec2 motionVectorSpecular(vec3 pos, vec2 motionpos, vec2 texcord, vec3 prevpos){
    vec2 iResolution = wh*1.5;

    float stepSize = 4.;
    vec2 bestPos = texcord;
    vec2 centerPos = texcord;
    float bestDist = length(pos-prevpos);
    //vec3 pm = texture(color, texCoord - motionpos).xyz;
    vec2 m = texcord - motionpos;
    vec3 prevposm = texture(prevPosition, m).xyz;
    if(length(pos - prevposm) < bestDist){
        bestDist = length(pos - prevposm);
        bestPos = m;
        centerPos = m;
    }


    for(int i = 0; i < 20; i++){
        vec2 q = (centerPos*iResolution + (vec2(1., 0.)*stepSize))/iResolution;
        vec3 prevposq = texture(prevPosition, q).xyz;
        if(length(pos-prevposq) < bestDist){
            bestDist = length(pos-prevposq);
            bestPos = q;
        }

        q = (centerPos*iResolution + (vec2(-1., 0.)*stepSize))/iResolution;
        prevposq = texture(prevPosition, q).xyz;
        if(length(pos-prevposq) < bestDist){
            bestDist = length(pos-prevposq);
            bestPos = q;
        }

        q = (centerPos*iResolution + (vec2(0., 1.)*stepSize))/iResolution;
         prevposq = texture(prevPosition, q).xyz;
        if(length(pos-prevposq) < bestDist){
            bestDist = length(pos-prevposq);
            bestPos = q;
        }

       q = (centerPos*iResolution + (vec2(0., -1.)*stepSize))/iResolution;
         prevposq = texture(prevPosition, q).xyz;
        if(length(pos-prevposq) < bestDist){
            bestDist = length(pos-prevposq);
            bestPos = q;
        }

       q = (centerPos*iResolution + (vec2(0., 0.)*stepSize))/iResolution;
         prevposq = texture(prevPosition, q).xyz;
        if(length(pos-prevposq) < bestDist){
            bestDist = length(pos-prevposq);
            bestPos = q;
        }

        if(length(bestPos*iResolution - centerPos*iResolution) < 0.01){
            if(stepSize >= 0.9999 && stepSize <= 1.0001){
                break;
            }else{
                stepSize *= 0.5;
            }
        }
        centerPos = bestPos;
    }


    for(int i = 0; i < 9; i++){
        //if(i == 4){continue;}
        vec2 offset = vec2(float(i%3)-1., float(i/3)-1.);
        vec3 prevposq = texture(prevPosition, (centerPos*iResolution + offset)/iResolution).xyz;
        if(length(pos - prevposq) < bestDist){
            bestDist = length(pos - prevposq);
            bestPos = (centerPos*iResolution + offset)/iResolution;
        }
    }
    return bestPos;
}

void main()
{
    vec2 fragCoord = texCoord*wh;
    uint r = uint(uint(fragCoord.x) * uint(1973) + uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) | uint(1);

    vec3 InitSampleLO = texture(color, texCoord).xyz;
    vec4 pst = texture(secondpos, texCoord).xyzw;

    vec3 cameraOffset = viewPos - lastViewPos;
    vec3 View = texture(position, texCoord).xyz;
    vec4 Projected = vec4(View.xyz, 1.);// + vec4(cameraOffset, 0.);
    Projected = prevview * Projected;
    Projected = prevproj * Projected;
    Projected /= Projected.w;
    vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;

            float roughness = texture(color, texCoord).w;

if(roughness < 0.5){
    //tempAccum = vec4(col, 0.);
    //den1 = vec4(col, 0.);  
    //var1 = vec4(1.);
 //return;   
 //vec2 motionVectorSpecular(vec3 pos, vec2 motionpos, vec2 texcord, vec3 prevpos){

ProjectedCoordinates = motionVectorSpecular(texture(position, texCoord).xyz, texCoord-ProjectedCoordinates , texCoord, texture(prevPosition, texCoord).xyz);
}
    /*
    prevW;
uniform sampler2D prevL;
uniform sampler2D prevP;
    */
    vec3 weigthR = texture(prevW, ProjectedCoordinates).xyz;
    float RM = weigthR.x;
    float Rw = weigthR.y;
    float RW = weigthR.z;
    vec3 LO = texture(prevL, ProjectedCoordinates).xyz;
  //  vec3 test = LO*0.9 + InitSampleLO*0.1;
    vec4 _pst = texture(prevP, ProjectedCoordinates);
    float w = lum(InitSampleLO);  




    if(roughness > 0.5){
            Rw /= max(RM,0.001);
   	    	RM = min(RM, 20.); 
   		    Rw *= max(RM,0.001);
        }else{
            Rw /= max(RM,0.001);    
   	    	RM = min(RM, 20.); 
   		    Rw *= max(RM,0.001);
        }     
    Rw = Rw + w;
	RM = RM + 1.0;
	///bool changed = false;
	if(rndf(r) < w/max(Rw,0.001)){
		LO = InitSampleLO;
		//isK = globalIllum.w;
        _pst = pst;
        //_nst = nst;
		//changed = true;
	}
            //R.Update(S, w, rndf(r));  
     
     if(RM * lum(LO) < 0.001){
         RW = 0.;
    }else{
        RW = Rw / max(RM * lum(LO), 0.001);
    }
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
    if(ProjectedCoordinates.x > 1. || ProjectedCoordinates.x < 0. || ProjectedCoordinates.y > 1. || ProjectedCoordinates.y < 0. || disc ){
        RM = 1.;
        RW = 1.;
        Rw = 1.;
        LO = InitSampleLO;
    }
    weigths = vec4(RM, Rw, RW, 1.0);
	outrad = vec4(clamp(LO,0.,100.) , 1.0);
    outpos = _pst;
    //weigths;
    //outrad;
    //outpos;
}