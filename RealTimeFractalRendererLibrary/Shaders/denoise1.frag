#version 450 compatibility
#extension GL_ARB_shading_language_packing : enable

layout(location = 0) out vec4 den2;
layout(location = 1) out vec4 var2;
layout(location = 2) out vec4 colorfog2;
layout(location = 3) out vec4 secalb;
layout(location = 4) out vec4 sssTexDenoise;

in vec2 texCoord;

uniform sampler2D color;
//uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D secondpos;
//uniform sampler2D weigth;
//uniform sampler2D outgoingr;
//uniform sampler2D weightS;
//uniform sampler2D outgoingrS;
//uniform sampler2D prevN;
//uniform sampler2D ACC;
uniform sampler2D den1;
uniform sampler2D var1;
uniform sampler2D reflN;
uniform sampler2D reflectionAlb;
uniform sampler2D colorfog;
uniform sampler2D holdinfo;
uniform sampler2D watpos;
uniform sampler2D watnorm;
uniform sampler2D tempacc;
uniform sampler2D sstexden;
uniform sampler2D colorTex;
/*
_TemporalRestirShader.SetInt("prevW", 5);
            _TemporalRestirShader.SetInt("prevL", 6);
            _TemporalRestirShader.SetInt("prevP", 7);
             */

//uniform mat4 view;
//uniform mat4 projection;
//uniform mat4 invview;
//uniform mat4 invproj;
//uniform mat4 prevview;
//uniform mat4 prevproj;
uniform vec2 wh;
uniform float time;
//uniform vec3 viewPos;
//uniform vec3 lastViewPos;

//uniform vec3 ldir;

// NOT MY CODE///////////////
uint wang_hash(inout uint seed) {
  seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
  seed *= uint(9);
  seed = seed ^ (seed >> 4);
  seed *= uint(0x27d4eb2d);
  seed = seed ^ (seed >> 15);
  return seed;
}

float rndf(inout uint state) { return float(wang_hash(state)) / 4294967296.0; }
///////////////////////////

float lum(vec3 c) {
  return sqrt(0.299 * c.x * c.x + 0.587 * c.y * c.y + 0.114 * c.z * c.z);
}

float g3x3(vec2 coords, vec2 iResolution) {
  const float atrous_kernel_weights[25] = {
      1.0 / 273.0, 4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0,  1.0 / 273.0,
      4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0,
      7.0 / 273.0, 26.0 / 273.0, 41.0 / 273.0, 26.0 / 273.0, 7.0 / 273.0,
      4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0,
      1.0 / 273.0, 4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0,  1.0 / 273.0};
  // return texture2D(colortex15, coords*1.0).x;
  vec2 p = coords * iResolution;
  float ppp2 = 0.;
  float ppp = 0.;
  for (int i = 0; i < 25; i++) {
    vec2 coords2 = vec2(float(i % 5) - 2., float(i / 5) - 2.) * 1.;
    // ppp += (texelFetch(colortex6, ivec2((p + coords2)*0.5 ), 0).x) ;
    ppp += texture2D(var1, ((p + coords2) / iResolution)).x *
           atrous_kernel_weights[i];
  }
  return ppp;
}

float g3x3v2(vec2 coords, vec2 iResolution) {
  const float atrous_kernel_weights[25] = {
      1.0 / 273.0, 4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0,  1.0 / 273.0,
      4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0,
      7.0 / 273.0, 26.0 / 273.0, 41.0 / 273.0, 26.0 / 273.0, 7.0 / 273.0,
      4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0,
      1.0 / 273.0, 4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0,  1.0 / 273.0};
  // return texture2D(colortex15, coords*1.0).x;
  vec2 p = coords * iResolution;
  float ppp2 = 0.;
  float ppp = 0.;
  for (int i = 0; i < 25; i++) {
    vec2 coords2 = vec2(float(i % 5) - 2., float(i / 5) - 2.) * 0.65;
    // ppp += (texelFetch(colortex6, ivec2((p + coords2)*0.5 ), 0).x) ;
    ppp += texture2D(color, ((p + coords2) / iResolution)).w *
           atrous_kernel_weights[i];
  }
  return ppp;
}


int midean(vec3 a){
    float maxx = max(a.x, max(a.y,a.z));
    float minx = min(a.x, min(a.y,a.z));

    return (maxx == a.x)?(minx == a.z)?1:2:(maxx == a.y)?
    (minx == a.x)?2:0:(minx == a.x)?1:0;
}

vec3 medianof9s(vec2 uv, vec2 iResolution, float offsetS, sampler2D tex){
    vec3 colArray[9];  
	

	for(int i = 0; i < 9; i++){
		vec2 offset = vec2(float(i%3)-1., float(i/3)-1.)*offsetS;
		//vec3 currCol =  texelFetch(tex, ivec2((uv*iResolution + offset)),0).rgb;
    vec3 currCol = texture2D(tex, (uv*iResolution + offset)/iResolution).rgb;
		colArray[i] = currCol;
	}

    int ms1 = (midean(vec3(lum(colArray[0]), lum(colArray[1]),lum(colArray[2]))));
    int ms2 = (midean(vec3(lum(colArray[3]), lum(colArray[4]),lum(colArray[5]))));
    int ms3 = (midean(vec3(lum(colArray[6]), lum(colArray[7]),lum(colArray[8]))));


	int id1 = ms1;
	int id2 = 3+ms2;
	int id3 = 6+ms3;
    
    int finid = (midean(vec3(lum(colArray[id1]),lum(colArray[id2]),lum(colArray[id3]))));

	//return (finid == 0)?colArray[id1]:(finid == 1)?colArray[id2]:colArray[id3];
    
	int as = id1 + (id2 - id1)*finid;
	return colArray[as + (id3-as)*max(finid-1,0)]; 

}

void main() {
  vec2 fragCoord = texCoord * wh;

  //vec2 fragCoord = texCoord * wh;

  if(int(fragCoord.x) % 2 == int(time)%2 && int(fragCoord.y) % 2 == int(time)%2){
      //discard;
   }
//  uint r = uint(uint(fragCoord.x) * uint(1973) +
//                uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) |
//           uint(1);
  if (texture2D(albedo, texCoord).w > 0.5) {
    return;
  }
  vec3 InitSampleLO = texture2D(color, texCoord).xyz;
  vec4 pst = texture2D(secondpos, texCoord).xyzw;

  /*
  vec3 cameraOffset = viewPos - lastViewPos;
  vec3 View = texture2D(position, texCoord).xyz;
  vec4 Projected = vec4(View.xyz, 1.); // + vec4(cameraOffset, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;
  */

  vec2 iResolution = wh;

  /*const float atrous_kernel_weights[25] = {
                  1.0 / 256.0, 4.0 / 256.0,  6.0 / 256.0,  4.0 / 256.0,  1.0 /
  256.0, 4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0, 6.0
  / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0, 4.0 /
  256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0, 4.0
  / 256.0,  6.0 / 256.0,  4.0 / 256.0,  1.0 / 256.0};

  const float atrous_kernel_weights[49] = {
          0.0,0.,1.,2.,1.,0.,0.,
          0.,3.,13.,22.,13.,3.,0.,
          1.,13.,59.,97.,59.,13.,1.,
          2.,22.,97.,159.,97.,22.,2.,
          1.,13.,59.,97.,59.,13.,1.,
          0.,3.,13.,22.,13.,3.,0.,
          0.0,0.,1.,2.,1.,0.,0.
  };	*/
  /*
const float atrous_kernel_weights[25] = {
1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0,
4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0,
4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0 };
*/
  float atrous_kernel_weights[9] = {

      16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0,
      24.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0};

  float mult = pow(2., texture2D(den1, texCoord).w);
  vec3 col = vec3(0.);
  float w = 0.;
  vec2 currCoors = texCoord * iResolution;
  //float hist = texture2D(ACC, texCoord).w;
  //float hist = max(1.0 - hist / 22., 0.);
  float hist = 0.;
  float moment1 = 0.;
  float moment2 = 0.;
  float momentCount = 0.;

  float dodenois = texture2D(colorTex, texCoord).w;
  
  vec3 currrp = texture2D(den1, texCoord).xyz;
  //vec3 medianof9s(vec2 uv, vec2 iResolution, float offsetS, sampler2D tex){
    currrp = medianof9s(texCoord, iResolution, .175, den1);
  vec4 ND = texture2D(normal, texCoord);

  float var = 0.;
  float w2 = 0.;
  float curvar = texture2D(var1, texCoord).x;
  // float roughness = texture2D(color, texCoord).w;
  float roughness = texture2D(holdinfo, texCoord).w;
  vec3 curN = texture2D(reflN, texCoord).xyz;
  float depth1 = texture2D(normal, texCoord).w;

  for (int i = 0; i < 9; i++) {
    vec2 coords = vec2(float(i % 3) - 1., float(i / 3) - 1.) * mult;

    vec2 fincords = (currCoors + coords) / iResolution;

    vec4 ND2 = texture2D(normal, fincords);
    float wd = exp(-(abs(ND.w - ND2.w) / 122.945));
    float wp = max(pow(max(dot(ND.xyz, ND2.xyz), 0.), 6.), 0.0);
    float nextrp = texture2D(var1, fincords).x;
    float wrp = exp(-(abs(((curvar)) - (nextrp)) / (1.2 + 0.001)));
    // wrp = max(wrp, 1.0-clamp(hist*0.01, 0., 1.));

    float weigth = 1.0;

    float depth2 = texture2D(normal, fincords).w;

    float depthWeight = pow(exp(-abs((depth1) - (depth2)) * 0.001), 12.);
    float dll = max((g3x3(texCoord, iResolution)), 0.);
    dll = texture2D(var1, texCoord).x;
    float dll2 = max((g3x3(fincords, iResolution)), 0.);
        dll2 = texture2D(var1, fincords).x;

		float lumWeigth = exp(-abs(dll-dll2)/(dll*.5+0.00001));
       lumWeigth = max(lumWeigth, 0.7*exp(-texture(tempacc, texCoord).w));

    weigth = wp * depthWeight * lumWeigth;

    var += pow(atrous_kernel_weights[i], 2.) * (weigth * weigth) * nextrp;
    w2 += pow((atrous_kernel_weights[i]) * weigth, 2.);
  }
  vec4 currVAR =
      vec4((var / max(pow(w2, 1.), 0.000001)), texture2D(var1, texCoord).yzw);
  var2 = currVAR;

  float DP1 = texture2D(reflectionAlb, texCoord).w;

  vec3 dn1 = texture2D(colorfog, texCoord).xyz;
  vec3 reflct = vec3(0.);
  float reflctdiv = (0.);

  vec3 shad1 = vec3(0.);
  float shad1we = 0.;

  vec3 shd1 = texture2D(reflectionAlb, texCoord).xyz;
  float rflLength = texture2D(reflN, texCoord).w;

  float normLength = texture2D(color, texCoord).w;

  vec3 subsurf = vec3(0.);
  float subsurfweigth = 0.;
  vec3 currsub = texture2D(sstexden, texCoord).xyz;
              currsub = medianof9s(texCoord, iResolution, .175, sstexden);

  //normLength = g3x3v2(texCoord, iResolution);
  //if(dodenois < 0.5){
      for (int i = 0; i < 9; i++) {
      vec2 coords = vec2(float(i % 3) - 1., float(i / 3) - 1.) * mult;

      vec2 fincords = (currCoors + coords) / iResolution;
      if (texture2D(albedo, fincords).w > 0.5) {
        continue;
      }
      vec4 ND2 = texture2D(normal, fincords);
      // float wd = exp(-(abs(ND.w - ND2.w) / 122.945));
      float wd = pow(exp(-abs((ND.w) - (ND2.w)) * 0.01), 1.);
      float wp = max(pow(max(dot(ND.xyz, ND2.xyz), 0.), 28.), 0.0);
      vec3 nextrp = texture2D(den1, fincords).xyz;
          nextrp = medianof9s(fincords, iResolution, .175, den1);

      vec3 nextN = texture2D(reflN, fincords).xyz;

      // float g3x3(vec2 coords, vec2 iResolution) {

      float dll = max(sqrt(g3x3(texCoord, iResolution)), 0.);
      //dll = max(dll, 1. - clamp(texture(tempacc, texCoord).w/20.,0., 1.));
      //   dll *= pow(clamp(normLength / 62., 0.0001, 1.), 2.);

      float wrp =
          exp(-(abs((lum(currrp)) - lum(nextrp)) / (.451 * currVAR.x + 0.00001)));

       wrp = max(wrp, 0.7*exp(-texture(tempacc, texCoord).w));
      // wrp = max(wrp, hist);
      float weigth = 1.0;
      float depth2 = texture2D(normal, fincords).w;

      float depthWeight = pow(exp(-abs((depth1) - (depth2)) * 0.0001), 12.);

      weigth = wp * wrp * depthWeight;
      vec3 nextsub = texture2D(sstexden, fincords).xyz;
              nextsub = medianof9s(fincords, iResolution, .175, sstexden);


      float issub = texture2D(sstexden, fincords).w;
      float wrpsub =
          exp(-(abs((lum(currsub)) - lum(nextsub)) / (1.51 + 0.00001)));
      float weightsub = wp*depthWeight*wrpsub;

      if(issub < 0.5){
        weightsub = 0.;
      }

      subsurf += (atrous_kernel_weights[i]) * weightsub * nextsub;
      subsurfweigth += ((atrous_kernel_weights[i]) * weightsub);

      col += (atrous_kernel_weights[i]) * weigth * nextrp;
      w += ((atrous_kernel_weights[i]) * weigth);
    }
  //}
  
  vec4 wpz = texture2D(watpos, texCoord);
  vec4 wn = texture2D(watnorm, texCoord);
  if (wpz.w > 0.5) {
    ND = wn;
    roughness = 0.001;
  }
  for (int i = 0; i < 9; i++) {
    vec2 coords =
        vec2(float(i % 3) - 1., float(i / 3) - 1.) * mult * pow(roughness, 0.5);

    vec2 fincords = (currCoors + coords) / iResolution;
    if (texture2D(albedo, fincords).w > 0.5) {
      continue;
    }

    vec4 ND2 = texture2D(normal, fincords);
    if (wpz.w > 0.5) {
      ND2 = texture2D(watnorm, fincords);
    }
    float wd = exp(-(abs(ND.w - ND2.w) / 122.945));
    float wp = max(pow(max(dot(ND.xyz, ND2.xyz), 0.), 26.), 0.0);
    vec3 nextrp = texture2D(den1, fincords).xyz;
    vec3 nextN = texture2D(reflN, fincords).xyz;

    vec3 dn2 = texture2D(colorfog, fincords).xyz;

    float dll = max(sqrt(g3x3(texCoord, iResolution)), 0.);
    dll = max(dll, 1. - clamp(texture(tempacc, texCoord).w/20.,0., 1.));
    float wrp2 = exp(-(abs((lum(dn1)) - lum(dn2)) / (12.072 * dll + 0.00001)));

    // wrp = max(wrp, 1.0-clamp(hist*0.5, 0., 1.));
    wrp2 = max(wrp2, hist);

    float DP2 = texture2D(reflectionAlb, fincords).w;
    float dotWeigth = max(exp(-abs(DP1 - DP2) / 0.0001), sqrt(roughness));
    // weigth = max(weigth, hist);
    float weigth2 = (wp * dotWeigth);
    if (roughness < 0.5 && texture2D(secondpos, texCoord).w < 0.5) {

      weigth2 *= max(pow(max(dot(normalize(curN), normalize(nextN)), 0.0), 6.),
                     sqrt(sqrt(roughness)));
      // nextrp *= texture2D(reflectionAlb, fincords).xyz;
    }

    // weigth2 *= dotWeigth;

    reflct += (atrous_kernel_weights[i]) * weigth2 * dn2;
    reflctdiv += (atrous_kernel_weights[i]) * weigth2;
  }

  vec3 ret = col / max(w, 0.001);
  // //ret = currrp;

  float mean = moment1 / momentCount;
  float variance = moment2 / momentCount - pow(mean, 2.0);

  if (lum(currrp) > mean + 2.5 * max(sqrt(variance), 0.001)) {
     //ret = mean / lum(currrp) * ret;
  }
  // ret = texture2D(reflN, texCoord).xyz;
  // ret = currVAR.xxx;
  colorfog2 =
      vec4(reflct / max(reflctdiv, 0.0001), texture2D(colorfog, texCoord).w);


//  den2 = texture2D(den1, texCoord).xyzw;
 // if(dodenois < 0.5){
    den2 = vec4(ret, max(texture2D(den1, texCoord).w - 1., 1.));
 // }else{
 //   den2 = texture2D(den1, texCoord).xyzw;
 // }
  

/*
vec3 subsurf = vec3(0.);
  float subsurfweigth = 0.;
  vec3 currsub = texture2D(sstexden, texCoord).xyz;
*/
sssTexDenoise = vec4(subsurf/max(subsurfweigth, 0.0001), texture2D(sstexden, texCoord).w);


  // secalb = vec4(shad1/max(shad1we,0.001), texture2D(reflectionAlb,
  // texCoord).w);
}