#version 330

layout(location = 0) out vec4 prevN;
layout(location = 1) out vec4 prevAcc;

in vec2 texCoord;

uniform sampler2D normal;
uniform sampler2D Acc;
uniform sampler2D TAA;

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

void main() {

  prevN = texture2D(normal, texCoord);
  prevAcc = texture2D(Acc, texCoord);
}