#version 330

layout(location = 0) out vec4 prevN;
layout(location = 1) out vec4 prevAcc;
layout(location = 2) out vec4 prevTAA;
layout(location = 3) out vec4 prevup;

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
uniform sampler2D up;

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
uniform vec3 ldir;

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

  /*
  uniform sampler2D tempWeights;
  uniform sampler2D tempOutL;
  uniform sampler2D tempPos;
  uniform sampler2D spatWe;
  uniform sampler2D spatLO;
  */

  /*
  if(ProjectedCoordinates.x > 1. || ProjectedCoordinates.x < 0. ||
  ProjectedCoordinates.y > 1. || ProjectedCoordinates.y < 0. ||
  length(cameraOffset) > 0.01){ a = vec4(1.); a1 = vec4(0.); a2 = vec4(0.); a3 =
  vec4(1.); a4 = vec4(0.);
  }*/
  vec2 iResolution = wh;

  uint seedCam = uint(max(time + 1, 0));
  vec2 smallOffset =
      ((vec2(rndf(seedCam), rndf(seedCam))) * 2. - 1.) / iResolution;

  prevN = texture2D(normal, texCoord);
  prevAcc = texture2D(Acc, texCoord);
  prevTAA = texture2D(TAA, texCoord);
  prevup = texture2D(up, texCoord);
  // prevPosition = texture2D(position, texCoord*1.5);
}