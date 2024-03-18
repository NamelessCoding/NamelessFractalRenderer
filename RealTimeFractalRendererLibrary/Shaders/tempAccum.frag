#version 330


layout (location = 0) out vec4 tempAccum;
layout (location = 1) out vec4 den1;
layout (location = 2) out vec4 var1;

in vec2 texCoord;

/*
_swapShader.SetInt("tempWeights", 0);
            _swapShader.SetInt("tempOutL", 1);
            _swapShader.SetInt("tempPos", 2);
            _swapShader.SetInt("spatWe", 3);
            _swapShader.SetInt("spatLO", 4);
*/
/*
_tempAccumShader.SetInt("color", 0);
            _tempAccumShader.SetInt("position", 1);
            _tempAccumShader.SetInt("normal", 2);
            _tempAccumShader.SetInt("albedo", 3);
            _tempAccumShader.SetInt("secondpos", 4);
            _tempAccumShader.SetInt("weigth", 5);
            _tempAccumShader.SetInt("outgoingr", 6);
            _tempAccumShader.SetInt("weightS", 7);
            _tempAccumShader.SetInt("outgoingrS", 8);
            _tempAccumShader.SetInt("prevN", 9);
            _tempAccumShader.SetInt("prevAcc", 10);
*/
uniform sampler2D color;
uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D secondpos;
uniform sampler2D weigth;
uniform sampler2D outgoingr;
uniform sampler2D weightS;
uniform sampler2D outgoingrS;
uniform sampler2D prevN;
uniform sampler2D prevAcc;
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
vec3 lerp(vec3 a, vec3 b, float t){
return mix(a,b,t);
}
float lerp(float a, float b, float t){
return mix(a,b,t);
}

vec3 interpolateHistory(
    vec3 prev, vec3 current,
    float antilagAlpha, float invHistLen, float minAlpha
) {
    float alpha = lerp(max(minAlpha, invHistLen), 1., antilagAlpha);
    return lerp(prev, current, alpha);
}
float lum(vec3 c){
return sqrt( 0.299*c.x*c.x + 0.587*c.y*c.y + 0.114*c.z*c.z );
}

vec3 ClipAABB(vec3 q,vec3 aabb_min, vec3 aabb_max){
	vec3 p_clip = 0.5 * (aabb_max + aabb_min);
	vec3 e_clip = 0.5 * (aabb_max - aabb_min) + 0.00000001;

	vec3 v_clip = q - vec3(p_clip);
	vec3 v_unit = v_clip.xyz / e_clip;
	vec3 a_unit = abs(v_unit);
	float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

	if (ma_unit > 1.0)
		return vec3(p_clip) + v_clip / ma_unit;
	else
		return q;
}


void main()
{
    vec2 iResolution = wh*1.5;
if(texture(albedo, texCoord).w > 0.5){
    return;
}

  vec3 cameraOffset = viewPos - lastViewPos;
  vec3 View = texture(position, texCoord).xyz;
  vec4 Projected = vec4(View.xyz, 1.);// + vec4(cameraOffset, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;

 float RW = texture(weightS, texCoord).z;
 vec3 col = texture(outgoingrS, texCoord).xyz*clamp(RW, 0., 200.);
            float roughness = texture(color, texCoord).w;

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
/*
vec2 xyDelta = ProjectedCoordinates*iResolution - texCoord*iResolution;
float depthCurr = texture(normal, texCoord).w;
float depthPrev = texture(prevN, ProjectedCoordinates).w;

vec4 motionFilterWidth = 
vec4(xyDelta, depthPrev - depthCurr, 0.01);

vec2 _weights = vec2(0.1, 0.5);
vec4 _weights1 = vec4(2., 1., 0.02, 0.);		//HistLen weights (2), min alpha color, min alpha moments
vec2 _momentsWeights = vec2(2., 128.);		//zMax, nDotRaise

vec4 prevColHistLen = vec4(0.);
vec4 prevColHistLenNoRESTIR = vec4(0.);

vec2 prevMoments = vec2(0.);
float sumW = 0.;
vec2 prevPos = ProjectedCoordinates*iResolution ;
vec2 toCenter = prevPos ;
vec2 pix = floor(toCenter);

vec2 px = fract(toCenter - pix);
vec2 ipx = 1.0 - px;
vec2 pixelWeights[2] = vec2[]( ipx, px );

vec4 nD = texture(normal, texCoord);
for(int i = 0; i < 4; ++i) {

    	//Reject if out of screen

		vec2 loc2 = pix + vec2(float(uint(i) & uint(1)), float(uint(i) >> uint(1)));


		//Grab difference between previous depth and use motion vector
		//to detect occlusion
		//No need to upscale, since nDepthPrev is already half res
		vec4 nDij = texture(prevN, loc2/iResolution);
		//float4 nDij = _normalDepthPrev[loc2];

		float distDepth = abs(nD.w - nDij.w + motionFilterWidth.z) / abs(nD.w);
		
		float normalDot = clamp(dot(nD.xyz, nDij.xyz),0.,1.);

		float w = pixelWeights[i & 1].x * pixelWeights[i >> 1].y;
		w *= pow(normalDot, 1 /*normalPower);
		
	    if(distDepth < _weights.x && normalDot > _weights.y) {
			vec3 momHistLen = texture(prevAcc, loc2/iResolution).www;
			prevColHistLen += vec4(texture(prevAcc, loc2/iResolution).xyz, momHistLen.z) * w;
			sumW += w;
		}
	}

	prevColHistLen /= sumW;
    prevColHistLenNoRESTIR /= sumW;

	prevMoments /= sumW;

	bool isInvalid = sumW < 1e-6;		//Ensure we don't use NaN, because apparently the pixel doesn't exist anymore
	
	//Compute moments by grabbing average from neighbors (but using depth and normal to ignore irrelevant neighbors)
	
	sumW = 1.;

	float histLen = 1;


	if(!isInvalid) {
		
    	float antilagAlpha = clamp(lerp(1, _weights1.x * 0., _weights1.y),0.,1.);
    	histLen = min(256., 1. + prevColHistLen.w * pow(1. - antilagAlpha, 10.));

		float invHistLen = 1. / histLen;
		col =  interpolateHistory(prevColHistLen.xyz, col, antilagAlpha, invHistLen, _weights1.z);
        //col2 = interpolateHistory(prevColHistLen.xyz, col2, antilagAlpha, invHistLen, _weights1.z);
		//reflectedColor = interpolateHistory(prevColHistLen.xyz, reflectedColor, antilagAlpha, invHistLen, _weights1.z);
		//spatialMoments = interpolateHistory(prevMoments.xyy, spatialMoments.xyy, antilagAlpha, invHistLen, _weights1.w).xy;
	}

   
	*/


vec4 n = texture(normal, texCoord);

vec4 nDij = texture(prevN, ProjectedCoordinates);
 bool outScreen = (ProjectedCoordinates.x > 1.0 || ProjectedCoordinates.x < 0. || ProjectedCoordinates.y > 1.0 || ProjectedCoordinates.y < 0.0);


    float depthWeight  =  pow(exp(-abs((nDij.w)-(n.w))*0.01), 12.);
    float normalWeight = pow(max(dot(nDij.xyz, n.xyz), 0.), 5.);
    float totalWeight  = (1.0-float(outScreen )) * normalWeight * depthWeight;


    float accumulation = texture(prevAcc, ProjectedCoordinates).w;
    accumulation  = clamp(accumulation+1.0, 0.0, 42.);
    accumulation *= totalWeight;



    float frameWeight = (1.0 / max(1.0, accumulation));

//

    vec3 prevFrame = texture(prevAcc, ProjectedCoordinates).xyz;
    vec3 kk = vec3(0.);
vec3 minCol = (vec3(9999.));
vec3 maxCol = (vec3(-9999.));
  for(int i = 0; i < 9; i++){
    vec2 coords = vec2(float(i%3)-1., float(i/3)-1.)*min(1.0-min(accumulation*0.1,1.),1.);
    vec2 newc = (ProjectedCoordinates*iResolution + coords)/iResolution;
    vec3 currSample = texture(prevAcc, newc).xyz;
    minCol = min(minCol, currSample);
    maxCol = max(maxCol, currSample);
    kk += currSample/9.;
  }
prevFrame = ClipAABB(prevFrame, minCol, maxCol);


//if(true){

//col = col*(frameWeight) + (1.0-frameWeight)*prevFrame;
tempAccum = vec4(col, accumulation);
den1 = vec4(col, 0.);
 vec2 nm = vec2(lum(col), pow(lum(col),2.));
    float variance = (abs(nm.r - nm.g * nm.g));

var1 = vec4(variance);
}