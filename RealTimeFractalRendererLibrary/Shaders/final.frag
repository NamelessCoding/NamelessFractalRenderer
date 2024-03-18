
out vec4 outputColor;
in vec2 texCoord;


layout(binding = 0, Rgba32f) uniform image3D rcpos; 

uniform sampler2D color;
uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D secondpos;
/*
_finalShader.SetInt("weigth", 5);
            _finalShader.SetInt("outgoingr", 6);
*/
uniform sampler2D weigth;
uniform sampler2D outgoingr;
uniform sampler2D weightS;
uniform sampler2D outgoingrS;
uniform sampler2D Acc;
uniform sampler2D den1;
uniform sampler2D var1;
uniform sampler2D TAA;
uniform sampler2D colorfog;



/*
weigthS", 7);
            _finalShader.SetInt("outgoingrS", 8);
*/

uniform mat4 view;
uniform mat4 projection;
uniform mat4 invview;
uniform mat4 invproj;
uniform vec2 wh;
uniform float time;
uniform vec3 viewPos;



vec3 tonemap_uchimura2(vec3 v)
{
    const float P = 1.0;  // max display brightness
    const float a = 1.8;  // contrast
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
//////////////////////////////////


float L(float x, float a){
if(x == 0.0){return 1.0;}
if(x != 0.0 && x < a && x > -a){return (a*sin(3.14159*x)*sin((3.14159*x)/a))/(pow(3.14159,2.)*x*x);}
return 0.0;
}

vec3 upscaleIndirect(vec2 iResolution, vec2 texcc){
vec3 col = vec3(0.);
float weight = 0.;
for(int i = 0; i < 25; i++){
      vec2 coords = vec2(float(i%5)-2., float(i/5)-2.);
      vec2 newCords = (texcc*iResolution + coords)/iResolution;
        
      float L_W = L(length(coords)/1., 2.0);
    col += texture(TAA, newCords).xyz*L_W;
   // weight += L_W;
  }
return col/max(1.,0.001);
}

vec3 blur2(vec2 p, float dist, vec2 iResolution) {
   // vec2 iResolution = vec2(viewWidth, viewHeight);

p*=iResolution.xy;
    vec3 s = vec3(0.);
    
    vec3 div = vec3(0.);
    //vec2 off = vec2(0.0, r);
    float k = 0.61803398875;
    for(int i = 0; i < 150; i++){
     float m = float(i)*0.01;
    float r = 2.*3.14159*k*float(i);
    vec2 coords = vec2(m*cos(r), m*sin(r))*dist;
    //vec4 c2 = texture(iChannel0, (p+coords)/iResolution.xy).xyzw;
    	vec2 cir = (p + coords) / iResolution.xy;
    vec3 ccc = texture(albedo, cir).xyz;

//vec3 c = upscaleIndirect(iResolution, cir);
vec3 c = texture(TAA, cir).xyz;
//texture(albedo, cir).w > 0.5 || 
//if(texture(position, cir).w > 0.5){
  //      c = ccc;
   // }

    //c = c*c *1.8;
    //vec3 bok = pow(c,vec3(4.));
      vec3 bok = vec3(1.);
      s+=c*bok;
      div += bok;
    }
        
    s/=div;
    
    return s;
    
}



vec3 blur23(vec2 p, float dist, vec2 iResolution) {

p*=iResolution.xy;
    vec3 s = vec3(0.);
    
    vec3 div = vec3(0.);
    //vec2 off = vec2(0.0, r);
    float k = 0.61803398875;
    for(int i = 0; i < 20; i++){
    float m = 1.;
    float r = 2.*3.14159*k*float(i);
    vec2 coords = vec2(m*cos(r), m*sin(r))*dist;
    //vec4 c2 = texture(iChannel0, (p+coords)/iResolution.xy).xyzw;
    	vec2 cir = (p + coords) / iResolution.xy;

//vec3 c = texture2D(colortex5, cir).xyz;
 vec2 uv2 = cir;
     vec3 rad2 = vec3(0.);
    vec2 offset2 = (cir*iResolution - iResolution.xy/2.)*1.;
    for(int i = 0; i < 20; i++){
       vec2 offset = cir*iResolution + offset2*smoothstep(0.,15.-length(uv2*2.0-1.)*1.5+dist, float(i)/20.)*1.;
       rad2.x += texture(TAA, (offset+offset2*0.0064*dist)/iResolution.xy).x;
       rad2.y += texture(TAA, (offset)/iResolution.xy).y;
       rad2.z += texture(TAA, (offset-offset2*0.0064*dist)/iResolution.xy).z;

    }
    rad2 /= 16.;
    
    vec3 c = rad2*0.8;
    //c = c*c *1.8;
    //vec3 bok = pow(c,vec3(4.));
      vec3 bok = vec3(1.);
      s+=c*bok;
      div += bok;
    }
        
    s/=div;
    
    return s;
    
}

//NOT MY CODE//////////////////////
vec3 ACESFilm(vec3 x)
{
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.,1.);
}

/***** RCAS *****/
#define FSR_RCAS_LIMIT (0.25-(1.0/16.0))
//#define FSR_RCAS_DENOISE

// Input callback prototypes that need to be implemented by calling shader
vec4 FsrRcasLoadF(vec2 p);
//------------------------------------------------------------------------------------------------------------------------------
void FsrRcasCon(
    out float con,
    // The scale is {0.0 := maximum, to N>0, where N is the number of stops (halving) of the reduction of sharpness}.
    float sharpness
){
    // Transform from stops to linear value.
    con = exp2(-sharpness);
}

vec3 FsrRcasF(
    vec2 ip, // Integer pixel position in output.
    float con
)
{
    // Constant generated by RcasSetup().
    // Algorithm uses minimal 3x3 pixel neighborhood.
    //    b 
    //  d e f
    //    h
    vec2 sp = vec2(ip);
    vec3 b = FsrRcasLoadF(sp + vec2( 0,-1)).rgb;
    vec3 d = FsrRcasLoadF(sp + vec2(-1, 0)).rgb;
    vec3 e = FsrRcasLoadF(sp).rgb;
    vec3 f = FsrRcasLoadF(sp+vec2( 1, 0)).rgb;
    vec3 h = FsrRcasLoadF(sp+vec2( 0, 1)).rgb;
    // Luma times 2.
    float bL = b.g + .5 * (b.b + b.r);
    float dL = d.g + .5 * (d.b + d.r);
    float eL = e.g + .5 * (e.b + e.r);
    float fL = f.g + .5 * (f.b + f.r);
    float hL = h.g + .5 * (h.b + h.r);
    // Noise detection.
    float nz = .25 * (bL + dL + fL + hL) - eL;
    nz=clamp(
        abs(nz)
        /(
            max(max(bL,dL),max(eL,max(fL,hL)))
            -min(min(bL,dL),min(eL,min(fL,hL)))
        ),
        0., 1.
    );
    nz=1.-.5*nz;
    // Min and max of ring.
    vec3 mn4 = min(b, min(f, h));
    vec3 mx4 = max(b, max(f, h));
    // Immediate constants for peak range.
    vec2 peakC = vec2(1., -4.);
    // Limiters, these need to be high precision RCPs.
    vec3 hitMin = mn4 / (4. * mx4);
    vec3 hitMax = (peakC.x - mx4) / (4.* mn4 + peakC.y);
    vec3 lobeRGB = max(-hitMin, hitMax);
    float lobe = max(
        -FSR_RCAS_LIMIT,
        min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.)
    )*con;
    // Apply noise removal.
    #ifdef FSR_RCAS_DENOISE
    lobe *= nz;
    #endif
    // Resolve, which needs the medium precision rcp approximation to avoid visible tonality changes.
    return (lobe * (b + d + h + f) + e) / (4. * lobe + 1.);
} 


vec4 FsrRcasLoadF(vec2 p) {
    return texture(TAA,p/(wh*1.5));
}
float lum(vec3 c){
return sqrt( 0.299*c.x*c.x + 0.587*c.y*c.y + 0.114*c.z*c.z );
}

vec3 sharpn(vec2 iResolution){
float str = 1.;
float strength = mix(-1./9., -1./6., str);
//vec3 col = texture(TAA, texCoord).xyz;
vec3 minCol = (vec3(9999.));
vec3 maxCol = (vec3(-9999.));
float minPixel = 9999.;
float maxPixel = -9999.;
for(int i = 0; i < 9; i++){
    vec2 offset = vec2(float(i%3), float(i/3))-1.;
    vec2 newCords = (texCoord*iResolution + offset)/iResolution;
    //if(i != 4){
        vec3 currSample = (texture(TAA, newCords).xyz);
         minCol = min(minCol, currSample);
         minPixel = min(min(currSample.x, min(currSample.y, currSample.z)), minPixel);
        maxPixel = max(max(currSample.x, max(currSample.y, currSample.z)), maxPixel);

        maxCol = max(maxCol, currSample);
    //}
}
//float amplitude = min(lum(minCol), 2.0-lum(maxCol))/max(lum(maxCol),0.001);
float amplitude = min((minPixel), max(2.0-(maxPixel),0.))/max((maxPixel),0.001);
float amp = amplitude*strength;

vec3 c1 =  texture(TAA, texCoord).xyz*1.;
vec3 c2 =  texture(TAA, (texCoord*iResolution + vec2(1., 0.))/iResolution).xyz*amp;
vec3 c3 =  texture(TAA, (texCoord*iResolution + vec2(-1., 0.))/iResolution).xyz*amp;
vec3 c4 =  texture(TAA, (texCoord*iResolution + vec2(0., 1.))/iResolution).xyz*amp;
vec3 c5 =  texture(TAA, (texCoord*iResolution + vec2(0., -1.))/iResolution).xyz*amp;

vec3 c = c1+c2+c3+c4+c5;

return c/max(1.0 + 4.*amp,0.001);
}

vec3 sharpn2(vec2 iResolution){
float amp = -1.;

vec3 c1 =  texture(TAA, texCoord).xyz*5.;
vec3 c2 =  texture(TAA, (texCoord*iResolution + vec2(1., 0.))/iResolution).xyz*amp;
vec3 c3 =  texture(TAA, (texCoord*iResolution + vec2(-1., 0.))/iResolution).xyz*amp;
vec3 c4 =  texture(TAA, (texCoord*iResolution + vec2(0., 1.))/iResolution).xyz*amp;
vec3 c5 =  texture(TAA, (texCoord*iResolution + vec2(0., -1.))/iResolution).xyz*amp;

vec3 c = c1+c2+c3+c4+c5;

return c/max(1.0 + 4.*amp,0.001);
}

vec3 sRGB(vec3 t) {
  return mix(1.055*pow(t, vec3(1./2.4)) - 0.055, 12.92*t, step(t, vec3(0.0031308)));
}

#define ROT(a)          mat2(cos(a), sin(a), -sin(a), cos(a))

#define PI          3.141592654
#define TAU         (2.0*PI)

const mat2 brot = ROT(2.399);
// License: Unknown, author: Dave Hoskins, found: Forgot where
vec3 dblur(vec2 q,float rad) {
  vec2 RESOLUTION = wh;
  vec3 acc=vec3(0);
  const float m = 0.0025;
  vec2 pixel=vec2(m*RESOLUTION.y/RESOLUTION.x,m);
  vec2 angle=vec2(0,rad);
  rad=1.;
  const int iter = 30;
  for (int j=0; j<iter; ++j) {  
    rad += 1./rad;
    angle*=brot;
    vec4 col=texture(TAA,q+pixel*(rad-1.)*angle);
    acc+=clamp(col.xyz, 0.0, 10.0);
  }
  return acc*(1.0/float(iter));
}

#define FXAA_SPAN_MAX 8.0
#define FXAA_REDUCE_MUL   (1.0/FXAA_SPAN_MAX)
#define FXAA_REDUCE_MIN   (1.0/128.0)
#define FXAA_SUBPIX_SHIFT (1.0/4.0)

vec3 FxaaPixelShader( vec4 uv, sampler2D tex, vec2 rcpFrame) {
    
    vec3 rgbNW = texture(tex, uv.zw, 0.0).xyz;
    vec3 rgbNE = texture(tex, uv.zw + vec2(1,0)*rcpFrame.xy, 0.0).xyz;
    vec3 rgbSW = texture(tex, uv.zw + vec2(0,1)*rcpFrame.xy, 0.0).xyz;
    vec3 rgbSE = texture(tex, uv.zw + vec2(1,1)*rcpFrame.xy, 0.0).xyz;
    vec3 rgbM  = texture(tex, uv.xy, 0.0).xyz;

    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM,  luma);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max(
        (lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL),
        FXAA_REDUCE_MIN);
    float rcpDirMin = 1.0/(min(abs(dir.x), abs(dir.y)) + dirReduce);
    
    dir = min(vec2( FXAA_SPAN_MAX,  FXAA_SPAN_MAX),
          max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
          dir * rcpDirMin)) * rcpFrame.xy;

    vec3 rgbA = (1.0/2.0) * (
        texture(tex, uv.xy + dir * (1.0/3.0 - 0.5), 0.0).xyz +
        texture(tex, uv.xy + dir * (2.0/3.0 - 0.5), 0.0).xyz);
    vec3 rgbB = rgbA * (1.0/2.0) + (1.0/4.0) * (
        texture(tex, uv.xy + dir * (0.0/3.0 - 0.5), 0.0).xyz +
        texture(tex, uv.xy + dir * (3.0/3.0 - 0.5), 0.0).xyz);
    
    float lumaB = dot(rgbB, luma);

    if((lumaB < lumaMin) || (lumaB > lumaMax)) return rgbA;
    
    return rgbB; 
}

//AgX Settings
const float MIDDLE_GREY = 0.18f;
const float SLOPE = 2.3f;
const float TOE_POWER = 1.9f;
const float SHOULDER_POWER = 3.1f;
const float COMPRESSION = 0.15;

//"Look" Settings
//Try 1.2 for a more saturated look. There's nothing wrong with intentionally skewing to develop a look you like,
//because intention is the entire point. That's why we should separate grading from compression, rather
//than combining them and forcing an artist 
const float SATURATION = 1.; 

//Demo Settings
const float EXPOSURE = -1.0;
const float MIN_EV = -10.0f;
const float MAX_EV = 6.5f;
const float AGX_LERP = 1.0;


mat3 InverseMat(mat3 m) 
{
    float d = m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) -
              m[0].y * (m[1].x * m[2].z - m[1].z * m[2].x) +
              m[0].z * (m[1].x * m[2].y - m[1].y * m[2].x);
              
    float id = 1.0f / d;
    
    mat3 c = mat3(1,0,0,0,1,0,0,0,1);
    
    c[0].x = id * (m[1].y * m[2].z - m[2].y * m[1].z);
    c[0].y = id * (m[0].z * m[2].y - m[0].y * m[2].z);
    c[0].z = id * (m[0].y * m[1].z - m[0].z * m[1].y);
    c[1].x = id * (m[1].z * m[2].x - m[1].x * m[2].z);
    c[1].y = id * (m[0].x * m[2].z - m[0].z * m[2].x);
    c[1].z = id * (m[1].x * m[0].z - m[0].x * m[1].z);
    c[2].x = id * (m[1].x * m[2].y - m[2].x * m[1].y);
    c[2].y = id * (m[2].x * m[0].y - m[0].x * m[2].y);
    c[2].z = id * (m[0].x * m[1].y - m[1].x * m[0].y);
    
    return c;
}

vec3 xyYToXYZ(vec3 xyY)
{
    if(xyY.y == 0.0f)
    {
        return vec3(0, 0, 0);
    }

    float Y = xyY.z;
    float X = (xyY.x * Y) / xyY.y;
    float Z = ((1.0f - xyY.x - xyY.y) * Y) / xyY.y;

    return vec3(X, Y, Z);
}

vec3 Unproject(vec2 xy)
{
    return xyYToXYZ(vec3(xy.x, xy.y, 1));				
}

mat3 PrimariesToMatrix(vec2 xy_red, vec2 xy_green, vec2 xy_blue, vec2 xy_white)
{
    vec3 XYZ_red = Unproject(xy_red);
    vec3 XYZ_green = Unproject(xy_green);
    vec3 XYZ_blue = Unproject(xy_blue);

    vec3 XYZ_white = Unproject(xy_white);

    mat3 temp = mat3(XYZ_red.x,	XYZ_green.x, XYZ_blue.x,
                     1.0f, 1.0f, 1.0f,
                     XYZ_red.z,	XYZ_green.z, XYZ_blue.z);

    mat3 inverse = InverseMat(temp);
    vec3 scale =  XYZ_white * inverse;

    return mat3(scale.x * XYZ_red.x, scale.y * XYZ_green.x,	scale.z * XYZ_blue.x,
                scale.x * XYZ_red.y, scale.y * XYZ_green.y,	scale.z * XYZ_blue.y,
                scale.x * XYZ_red.z, scale.y * XYZ_green.z,	scale.z * XYZ_blue.z);
}

mat3 ComputeCompressionMatrix(vec2 xyR, vec2 xyG, vec2 xyB, vec2 xyW, float compression)
{
    float scale_factor = 1.0f / (1.0f - compression);
    vec2 R = ((xyR - xyW) * scale_factor) + xyW;
    vec2 G = ((xyG - xyW) * scale_factor) + xyW;
    vec2 B = ((xyB - xyW) * scale_factor) + xyW;
    vec2 W = xyW;

    return PrimariesToMatrix(R, G, B, W);
}


vec3 OpenDomainToNormalizedLog2(vec3 openDomain, float minimum_ev, float maximum_ev)
{
    float total_exposure = maximum_ev - minimum_ev;

    vec3 output_log = clamp(log2(openDomain / MIDDLE_GREY), minimum_ev, maximum_ev);

    return (output_log - minimum_ev) / total_exposure;
}


float AgXScale(float x_pivot, float y_pivot, float slope_pivot, float power)
{
    return pow(pow((slope_pivot * x_pivot), -power) * (pow((slope_pivot * (x_pivot / y_pivot)), power) - 1.0), -1.0 / power);
}

float AgXHyperbolic(float x, float power)
{
    return x / pow(1.0 + pow(x, power), 1.0f / power);
}

float AgXTerm(float x, float x_pivot, float slope_pivot, float scale)
{
    return (slope_pivot * (x - x_pivot)) / scale;
}

float AgXCurve(float x, float x_pivot, float y_pivot, float slope_pivot, float toe_power, float shoulder_power, float scale)
{
    if(scale < 0.0f)
    {
        return scale * AgXHyperbolic(AgXTerm(x, x_pivot, slope_pivot, scale), toe_power) + y_pivot;
    }
    else
    {
        return scale * AgXHyperbolic(AgXTerm(x,x_pivot, slope_pivot,scale), shoulder_power) + y_pivot;
    }
}

float AgXFullCurve(float x, float x_pivot, float y_pivot, float slope_pivot, float toe_power, float shoulder_power)
{
    float scale_x_pivot = x >= x_pivot ? 1.0f - x_pivot : x_pivot;
    float scale_y_pivot = x >= x_pivot ? 1.0f - y_pivot : y_pivot;

    float toe_scale = AgXScale(scale_x_pivot, scale_y_pivot, slope_pivot, toe_power);
    float shoulder_scale = AgXScale(scale_x_pivot, scale_y_pivot, slope_pivot, shoulder_power);				

    float scale = x >= x_pivot ? shoulder_scale : -toe_scale;

    return AgXCurve(x, x_pivot, y_pivot, slope_pivot, toe_power, shoulder_power, scale);
}



void main()
{
    //*texture(albedo, texCoord).xyz;
     float RW = texture(weightS, texCoord).z;
    vec3 col = texture(outgoingrS, texCoord).xyz*clamp(RW, 0., 200.);
//weigth;
//uniform sampler2D outgoingr;
RW = texture(weigth, texCoord).z;
    col = texture(outgoingr, texCoord).xyz*clamp(RW, 0., 200.);

  //  vec3 col = texture(Acc, texCoord).xyz;
 //col = upscaleIndirect(wh, texCoord).xyz;
//col = texture(TAA, texCoord).xyz;

//float con;
//float sharpness = 0.4;
// FsrRcasCon(con,sharpness);
//col = FsrRcasF(texCoord*wh*2., con);

  //vec3 color = sharpn(wh*1.5);
   vec2 fragCoord = texCoord*wh;
    uint r = uint(uint(fragCoord.x) * uint(1973) + uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) | uint(1);

 vec4 p22 = vec4((texCoord) * 2.0 - 1.0, 0.0, 1.0);
    vec3 dir =
	  (invproj * p22).xyz / (invproj * p22).w;
    dir = normalize(mat3(invview) * dir);
    vec3 wi = dir;

float depth = texture(normal, texCoord).w;
float depthCenter = texture(normal, vec2(0.5)).w;
float f = 1.0 - exp(-pow((depth - depthCenter)*0.5,1.));
vec3 color = col;
//vec3 color = blur2(texCoord, clamp(f*0., 0., 20.), wh);
/*
    float maxRGB = max(color.r, max(color.g, color.b));
    float minRGB = min(color.r, min(color.g, color.b));
    
    //color = exp(15.0*L)*(color-minRGB)/(maxRGB-minRGB);
    
    vec3 workingColor = max(color, 0.0f) * pow(2.0f, EXPOSURE);
    
    mat3 sRGB_to_XYZ = PrimariesToMatrix(vec2(0.64,0.33),
                                         vec2(0.3,0.6), 
                                         vec2(0.15,0.06), 
                                         vec2(0.3127, 0.3290));

    mat3 adjusted_to_XYZ = ComputeCompressionMatrix(vec2(0.64,0.33),
                                                    vec2(0.3,0.6), 
                                                    vec2(0.15,0.06), 
                                                    vec2(0.3127, 0.3290), COMPRESSION);

    								
    mat3 XYZ_to_adjusted = InverseMat(adjusted_to_XYZ);

    vec3 xyz = workingColor * sRGB_to_XYZ;
    vec3 adjustedRGB = xyz * XYZ_to_adjusted;

    float x_pivot = abs(MIN_EV) / (MAX_EV - MIN_EV);
    float y_pivot = 0.5f;

    vec3 logV = OpenDomainToNormalizedLog2(adjustedRGB, MIN_EV, MAX_EV);

    float outputR = AgXFullCurve(logV.r, x_pivot, y_pivot, SLOPE, TOE_POWER, SHOULDER_POWER);
    float outputG = AgXFullCurve(logV.g, x_pivot, y_pivot, SLOPE, TOE_POWER, SHOULDER_POWER);
    float outputB = AgXFullCurve(logV.b, x_pivot, y_pivot, SLOPE, TOE_POWER, SHOULDER_POWER);

    workingColor = clamp(vec3(outputR, outputG, outputB), 0.0, 1.0);

    vec3 luminanceWeight = vec3(0.2126729f,  0.7151522f,  0.0721750f);
    vec3 desaturation = vec3(dot(workingColor, luminanceWeight));
    workingColor = mix(desaturation, workingColor, SATURATION);
    workingColor = clamp(workingColor, 0.f, 1.f);

    // Lerp between raw and image
    workingColor = mix(color, workingColor, AGX_LERP);	

col = workingColor;*/

col = color;
vec2 iResolution = wh*1.5;


//col = texture(co, texCoord).xyz;
//col = blur2(texCoord, clamp(f*2., 0., 10.), wh);
//vec3 bloom = dblur(texCoord, 1.5)*0.2;

// vec2 rcpFrame = 1./iResolution.xy;
// vec2 uv2 = texCoord;
        
    
    
// vec4 uv = vec4( uv2, uv2 - (rcpFrame * (0.5 + FXAA_SUBPIX_SHIFT)));
// col = FxaaPixelShader( uv, TAA, 1./iResolution.xy );


//col = 1.0-max(1.0-col,0.)*max(1.0-bloom,0.);

 // vec3 col = texture(den1, texCoord).xyz;
  
    //col *= ccc;
//texture(albedo, texCoord).w > 0.5 || 
//if( texture(position, texCoord).w > 0.5){
//col = texture(albedo, texCoord).xyz;
//}
//col = texture(TAA, texCoord).xyz/texture(TAA, texCoord).w;
    //weigth;
    //col = clamp(col, 0., 1.);
//col += 1./(1.+exp(-2.*(depth*0.05-2.)))*2.*vec3(0.01, 0.1, 0.2);
//col = texture(var1, texCoord).xxx;
//col = imageLoad(rcpos, ivec3((texture(position, texCoord).xyz+40.-floor(viewPos))*.9)).xyz;
//col = imageLoad(rcpos, ivec3(0,0,0)).xyz;

//col = vec3(0.);//
//if(texture(position, texCoord).w > 0.5){
//col = texture(albedo, texCoord).xyz;
 //}
vec3 camera = viewPos;
vec3 pos = texture(position, texCoord).xyz;

if(texture(albedo, texCoord).w > 0.5){
   pos = camera + wi*100.; 
}

	
vec3 direct = (pos-camera)/20.;
vec3 accumfog = vec3(0.);
float trans = 1.;
for(int i = 0; i < 20; i++){
    vec3 connect = vec3(0.);
    for(int k = 0; k < 27; k++){
        vec3 offset = vec3(float(k%3)-1., float((k/3)%3)-1., float(k/9)-1.);
        vec3 currp = (camera+40.)+offset*5.;
        connect += imageLoad(rcpos, ivec3(currp)).xyz;
    }
    accumfog += connect/27.;
        camera += direct*rndf(r);
        trans *= 1.0-lum(connect/9.);

}
//col *= trans;
//col += accumfog*0.05;
//if(texture(albedo, texCoord).w > 0.5 ){
 //   col = texture(albedo, texCoord).xyz;
//}
//col = texture(Acc, texCoord).xyz;
    col = max(col, 0.);
    //col = col*col*1.8;
    col *= 2.;
  //  col = pow(col, vec3(1.2, 1.5, 1.9));
    //   col += texture(colorfog, texCoord).xyz*0.8;

    col = tonemap_uchimura2(col);
   col = pow(col, vec3(1./2.2));
//col = sRGB(col);
    outputColor = vec4(col, 1.);
}