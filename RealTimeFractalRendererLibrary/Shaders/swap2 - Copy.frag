#version 330

layout(location = 0) out vec4 ups;

layout(location = 1) out vec4 prevS;

in vec2 texCoord;

uniform sampler2D col;
uniform sampler2D prevTAA;
uniform sampler2D position;
uniform sampler2D prevnorm;
uniform sampler2D norm;


uniform mat4 prevview;
uniform mat4 prevproj;

uniform vec2 wh;
uniform float time;
const ivec2 I_ZERO = ivec2(0);
const ivec2 I_ONE  = ivec2(1);


// Trig Constants
const float c_pi = 2.0 * acos(0.0);
const float c_tau = 2.0 * c_pi;

// 1D low discrepancy noise
const float c_phi = (1.0 + sqrt(5.0)) / 2.0;

// 2D low discrepancy noise
const float c_rho_1 = (
    2.0 * cosh(acosh(3.0 * sqrt(3.0) / 2.0) / 3.0) / sqrt(3.0)
);

const float c_rho_2 = c_rho_1 * c_rho_1; 

const vec2 c_rho = vec2(c_rho_1, c_rho_2);
const vec2 c_inv_rho = vec2(1.0) / c_rho;

vec4 RGBToYCoCg( vec4 RGB )
{
	float Y = dot(RGB.rgb, vec3(  1, 2,  1 )) * 0.25;
	float Co= dot(RGB.rgb, vec3(  2, 0, -2 )) * 0.25 + ( 0.5 * 256.0/255.0 );
	float Cg= dot(RGB.rgb, vec3( -1, 2, -1 )) * 0.25 + ( 0.5 * 256.0/255.0 );
	return vec4(Y, Co, Cg, RGB.a);
}

vec4 YCoCgToRGB( vec4 YCoCg )
{
    float Y= YCoCg.x;
	float Co= YCoCg.y - ( 0.5 * 256.0 / 255.0 );
	float Cg= YCoCg.z - ( 0.5 * 256.0 / 255.0 );
	float R= Y + Co-Cg;
	float G= Y + Cg;
	float B= Y - Co-Cg;
	return vec4(R,G,B, YCoCg.a);
}

struct ColorRange {
    vec4 _min;
    vec4 _max;
    vec4 _avg;
    vec4 _std;
};

const float c_pos_infty = +pow(2.0, 64.0);
const float c_neg_infty = -pow(2.0, 64.0);
const float c_min_std = 1.0 / 16.0;
const float c_min_var = c_min_std * c_min_std;

ColorRange sampleRange(sampler2D channel, vec2 uv, vec2 texel) {
    vec4 minColor = vec4(c_pos_infty);
    vec4 maxColor = vec4(c_neg_infty);
    
    float total = 0.0;
    vec4 L1Color = vec4(0.0);
    vec4 L2Color = vec4(0.0);
    
    for (int y = -1; y <= 1; y++) 
    for (int x = -1; x <= 1; x++) {
        vec4 s = RGBToYCoCg(
            texture(channel, uv + texel * vec2(x, y))
        );
        
        minColor = min(minColor, s);
        maxColor = max(maxColor, s);
        
        float w = 2.0 / float(1 + abs(x) + abs(y));
        total += w;
        L1Color += w * s;
        L2Color += w * s * s;
    }
    
    L1Color /= total;
    L2Color /= total;
    
    return ColorRange(
        minColor, maxColor, L1Color, 
        sqrt(c_min_var + L2Color - L1Color * L1Color)
    );
}

vec4 clampToRange(vec4 color, ColorRange range) {
    return YCoCgToRGB(
        clamp(
            RGBToYCoCg(color), 
            range._min, 
            range._max
        )
    );
}

float varianceRangeToRange(ColorRange lhs, ColorRange rhs) {
    vec4 std = 1.0 / ((1.0 / lhs._std) + (1.0 / rhs._std));
    vec4 diff = lhs._avg - rhs._avg;
    vec4 variance = (c_min_var + diff * diff) / (std * std);
    return length(variance / 4.0);
}


// Calculate offset for TAA Frames
vec2 offset(int frame) {
    // https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    vec2 x = float(frame) * c_inv_rho;
    return fract(x) - 0.5;
}
const float c_upscale_amount = 8.0;



void main() {
 	ivec2 iU = ivec2(texCoord*wh);
   	//Copy Original Texel
        vec2 texel = vec2(1.0) / wh.xy; 
  vec2 uv = texCoord;

  vec3 View = texture2D(position, uv).xyz;


  vec4 Projected = vec4(View.xyz, 1.); // - vec4(off, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;

vec4 nPrev = texture2D(prevnorm, ProjectedCoordinates);

vec4 norm2 = texture2D(norm, uv);


float ddt = max(dot(nPrev.xyz, norm2.xyz), 0.);

//ProjectedCoordinates += offset(int(time)-1)/wh.xy;
    vec4 next = texture(col, texCoord);
    vec4 prev = texture(prevTAA, ProjectedCoordinates.xy);
    
    ColorRange nextRange = sampleRange(col, uv, texel);
    ColorRange prevRange = sampleRange(prevTAA, ProjectedCoordinates, texel * c_upscale_amount);
    
    vec4 clamped = clampToRange(prev, nextRange);
    float prevDist = distance(prev, clamped);
    float nextDist = distance(next, clamped);
    float ratio = prevDist / (nextDist + prevDist + 0.015625);
    prev = mix(prev, next, ratio);
    
    vec2 offset = 2.0 * c_upscale_amount * (fract(uv / texel) - 0.5);
    float w = exp(-(1.0 - ratio) * dot(offset, offset));
    w *= c_upscale_amount * c_upscale_amount ;
    
   
    float var = varianceRangeToRange(nextRange, prevRange);
    float rc = 1.0 - exp(-16.0  * var * w);


    float ms = exp(-abs(nPrev.w - norm2.w)/2.);

    ups = max(mix(prev, next, rc*(1.-ddt)),0.);
prevS = max(mix(prev, next, rc*(1.-ddt)),0.);
}