#version 450 compatibility
#extension GL_ARB_shading_language_packing: enable

layout (location = 0) out vec4 den2;
layout (location = 1) out vec4 var2;
layout (location = 2) out vec4 colorfog2;

in vec2 texCoord;
/*
_denoiseShader.Use();
            _denoiseShader.SetInt("color", 0);
            _denoiseShader.SetInt("position", 1);
            _denoiseShader.SetInt("normal", 2);
            _denoiseShader.SetInt("albedo", 3);
            _denoiseShader.SetInt("secondpos", 4);
            _denoiseShader.SetInt("weigth", 5);
            _denoiseShader.SetInt("outgoingr", 6);
            _denoiseShader.SetInt("weightS", 7);
            _denoiseShader.SetInt("outgoingrS", 8);
            _denoiseShader.SetInt("prevN", 9);
            _denoiseShader.SetInt("ACC", 10);
            _denoiseShader.SetInt("den1", 11);

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
uniform sampler2D ACC;
uniform sampler2D den1;
uniform sampler2D var1;
uniform sampler2D reflN;
uniform sampler2D reflectionAlb;
uniform sampler2D colorfog;

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

float g3x3(vec2 coords, vec2 iResolution) {
	const float atrous_kernel_weights[9] = {
			1./16., 1./8.,1./16.,
			1./8.,1./4.,1./8.,
			1./16.,1./8.,1./16.
			 };
	//return texture2D(colortex15, coords*0.5).x;
	vec2 p = coords * iResolution;
	float ppp2 = 0.;
	float ppp = 0.;
	for (int i = 0; i < 9; i++) {
		vec2 coords2 = vec2(float(i % 3) - 1., float(i / 3) - 1.)*4.;
		ppp += (texture(var1, coords2).x) * atrous_kernel_weights[i];
		//ppp2 += atrous_kernel_weights[i];
	}
	return (ppp/max(1.,0.001));
}

void main()
{
    vec2 fragCoord = texCoord*wh;
    uint r = uint(uint(fragCoord.x) * uint(1973) + uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) | uint(1);
if(texture(albedo, texCoord).w > 0.5){
    return;
}
    vec3 InitSampleLO = texture(color, texCoord).xyz;
    vec4 pst = texture(secondpos, texCoord).xyzw;

    vec3 cameraOffset = viewPos - lastViewPos;
    vec3 View = texture(position, texCoord).xyz;
    vec4 Projected = vec4(View.xyz, 1.);// + vec4(cameraOffset, 0.);
    Projected = prevview * Projected;
    Projected = prevproj * Projected;
    Projected /= Projected.w;
    vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;

    vec2 iResolution = wh*1.5;
    
	/*const float atrous_kernel_weights[25] = {
			1.0 / 256.0, 4.0 / 256.0,  6.0 / 256.0,  4.0 / 256.0,  1.0 / 256.0,
			4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
			6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0,
			4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
			1.0 / 256.0, 4.0 / 256.0,  6.0 / 256.0,  4.0 / 256.0,  1.0 / 256.0};

	const float atrous_kernel_weights[49] = {
		0.0,0.,1.,2.,1.,0.,0.,
		0.,3.,13.,22.,13.,3.,0.,
		1.,13.,59.,97.,59.,13.,1.,
		2.,22.,97.,159.,97.,22.,2.,
		1.,13.,59.,97.,59.,13.,1.,
		0.,3.,13.,22.,13.,3.,0.,
		0.0,0.,1.,2.,1.,0.,0.
	};	*/
		
const float atrous_kernel_weights[25] = {
  1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0,
  4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
  6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0,
  4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
  1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0 };

 /*float atrous_kernel_weights[9] = {
			 
			 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 
			 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 
			16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0
			 };*/

	

	float mult =  pow(2.,texture(den1, texCoord).w);
    vec3 col = vec3(0.);
    float w = 0.;
	vec2 currCoors = texCoord*iResolution;
	float hist = texture(ACC, texCoord).w;
hist = 1.0-step(hist/42., 0.2);
    float moment1 = 0.;
    float moment2 = 0.;
    float momentCount = 0.;
	vec3 currrp = texture(den1, texCoord).xyz;
    vec4 ND = texture(normal, texCoord);
vec3 fogcol = vec3(0.);
    float wf = 0.;
    float wf2 = 0.;
    float fog = 0.;
    vec3 currfog = texture(colorfog, texCoord).xyz;
    float currFog2 = texture(reflN, texCoord).w;
    for (int i = 0; i < 25; i++) {
		vec2 coords = vec2(float(i % 5) - 2., float(i / 5) - 2.) *mult  ;
		
		vec2 fincords = (currCoors + coords) / iResolution;

        vec3 nextrp = texture(colorfog, fincords).xyz;
        float currFog2next = texture(reflN, fincords).w;

       // wrp = max(wrp, 1.0-clamp(hist*0.01, 0., 1.));
       
        float wrp = exp(-(abs((lum(currfog)) - lum(nextrp)) / (1.1 + 0.001)));

		float weigth = wrp;

		fogcol += pow(atrous_kernel_weights[i],1.) * ( weigth  ) * nextrp;
		wf += pow((atrous_kernel_weights[i]) * weigth, 1.);


        float wrp2 = exp(-(abs(((currFog2)) - (currFog2next)) / (1.1 + 0.001)));

		float weigth2 = wrp2;
        fog += pow(atrous_kernel_weights[i],1.) * ( weigth2  ) * currFog2next;
        wf2 += pow((atrous_kernel_weights[i]) * weigth2, 1.);

	}
    colorfog2 = vec4(fogcol/max(wf,0.0001), 0.);

    float var = 0.;
    float w2 = 0.;
    float curvar = texture(var1, texCoord).x;
    float roughness = texture(color, texCoord).w;
    vec3 curN = texture(reflN, texCoord).xyz;
   
	for (int i = 0; i < 25; i++) {
		vec2 coords = vec2(float(i % 5) - 2., float(i / 5) - 2.) *mult * pow(roughness, 8.) ;
		
		vec2 fincords = (currCoors + coords) / iResolution;

        vec4 ND2 = texture(normal, fincords);
		float wd = exp(-(abs(ND.w - ND2.w) / 122.945));
		float wp = max(pow(max(dot(ND.xyz, ND2.xyz), 0.), 6.), 0.0);
        float nextrp = texture(var1, fincords).w;
        float wrp = exp(-(abs(((curvar)) - (nextrp)) / (1.2 + 0.001)));
       // wrp = max(wrp, 1.0-clamp(hist*0.01, 0., 1.));

		float weigth = 1.0;
		weigth =  wp ;

		var += pow(atrous_kernel_weights[i],2.) * ( weigth * weigth ) * nextrp;
		w2 += pow((atrous_kernel_weights[i]) * weigth, 2.);
	}
    vec4 currVAR = vec4(fog/max(wf2, 0.001), vec3(var/max(pow(max(w2,0.001),1.),0.001)));
    var2 = currVAR;

    

	for (int i = 0; i < 25; i++) {
		vec2 coords = vec2(float(i % 5) - 2., float(i / 5) - 2.) * mult ;
		
		vec2 fincords = (currCoors + coords) / iResolution;
        if(texture(albedo, fincords).w > 0.5){
            continue;
        }
        vec4 ND2 = texture(normal, fincords);
		float wd = exp(-(abs(ND.w - ND2.w) / 122.945));
		float wp = max(pow(max(dot(ND.xyz, ND2.xyz), 0.), 126.), 0.0);
        vec3 nextrp = texture(den1, fincords).xyz;
        vec3 nextN = texture(reflN, fincords).xyz;
        moment1 += lum(nextrp);
        moment2 += lum(nextrp) * lum(nextrp);
        momentCount += 1.0;
        float dll = sqrt(currVAR.x);
        
        float wrp = exp(-(abs((lum(currrp)) - lum(nextrp)) / (3.2*dll + 0.001)));
        //wrp = max(wrp, 1.0-clamp(hist*0.5, 0., 1.));
		wrp = max(wrp, hist);

		float weigth = 1.0;
		weigth =   wp  * wrp;
		//weigth = max(weigth, hist);

        if(roughness < 0.5 && texture(secondpos, texCoord).w < 0.5){
            weigth *= max(pow(max(dot(normalize(curN), normalize(nextN)),0.0), 126.), sqrt(sqrt(roughness)));
            //nextrp *= texture(reflectionAlb, fincords).xyz;
        }

		col += (atrous_kernel_weights[i]) * weigth * nextrp;
		w += ((atrous_kernel_weights[i]) * weigth);
	}


         vec3 ret = col / max(w, 0.000001);
        // //ret = currrp;

        // float mean = moment1 / momentCount;
        // float variance = moment2 / momentCount - pow(mean, 2.0);

        // if(lum(currrp) > mean + 1.6 * max(sqrt(variance), 0.001)) {
        //     ret = mean / lum(currrp) * ret;
        // }
    //ret = texture(reflN, texCoord).xyz;
	den2 = vec4(ret, texture(den1, texCoord).w+1.);



    
    
}