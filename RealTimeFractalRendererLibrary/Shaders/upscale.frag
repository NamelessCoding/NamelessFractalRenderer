#version 450 compatibility
// colortexFogPrev

layout(location = 0) out vec4 upp;

// prevSecondPosition
in vec2 texCoord;

uniform sampler2D TAA;

uniform vec2 wh;
uniform float scale;
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

const ivec2 I_ZERO = ivec2(0);
const ivec2 I_ONE  = ivec2(1);



void main() {
  vec2 iResolution = wh;

  	ivec2 iU = ivec2(texCoord*iResolution);
   	//Copy Original Texel
    upp = all(equal( iU&1 , I_ZERO)) ? texelFetch(TAA,iU>>1,0) : upp-upp;
}