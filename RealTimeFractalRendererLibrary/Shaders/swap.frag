#version 330


layout (location = 0) out vec4 tempwe;
layout (location = 1) out vec4 templ;
layout (location = 2) out vec4 temppos;
layout (location = 3) out vec4 spatw;
layout (location = 4) out vec4 spatl;
layout (location = 5) out vec4 prevN;
layout (location = 6) out vec4 prevAcc;
layout (location = 7) out vec4 prevTAA;
//layout (location = 8) out vec4 prevPosition;

in vec2 texCoord;

/*
_swapShader.SetInt("tempWeights", 0);
            _swapShader.SetInt("tempOutL", 1);
            _swapShader.SetInt("tempPos", 2);
            _swapShader.SetInt("spatWe", 3);
            _swapShader.SetInt("spatLO", 4);
*/

uniform sampler2D tempWeights;
uniform sampler2D tempOutL;
uniform sampler2D tempPos;
uniform sampler2D spatWe;
uniform sampler2D spatLO;
uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D Acc;
uniform sampler2D TAA;

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


void main()
{
   
/*
uniform sampler2D tempWeights;
uniform sampler2D tempOutL;
uniform sampler2D tempPos;
uniform sampler2D spatWe;
uniform sampler2D spatLO;
*/


  vec3 cameraOffset = viewPos - lastViewPos;
  vec3 View = texture(position, texCoord*1.5).xyz;
  vec4 Projected = vec4(View.xyz, 1.) + vec4(cameraOffset, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;

vec4 a = texture(tempWeights, texCoord*1.5);
vec4 a1 = texture(tempOutL, texCoord*1.5);
vec4 a2 = texture(tempPos, texCoord*1.5);
vec4 a3 = texture(spatWe, texCoord*1.5);
vec4 a4 = texture(spatLO, texCoord*1.5);
/*
if(ProjectedCoordinates.x > 1. || ProjectedCoordinates.x < 0. || ProjectedCoordinates.y > 1. || ProjectedCoordinates.y < 0. || length(cameraOffset) > 0.01){
    a = vec4(1.);
    a1 = vec4(0.);
    a2 = vec4(0.);
    a3 = vec4(1.);
    a4 = vec4(0.);
}*/
prevN = texture(normal, texCoord*1.5);
tempwe = a;
templ = a1;
temppos = a2;
spatw = a3;
spatl = a4;
prevAcc = texture(Acc, texCoord*1.5);
prevTAA = texture(TAA, texCoord*1.5);
//prevPosition = texture(position, texCoord*1.5);
}