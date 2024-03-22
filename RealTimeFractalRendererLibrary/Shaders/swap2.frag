#version 330

layout(location = 0) out vec4 tempwe;
layout(location = 1) out vec4 templ;
layout(location = 2) out vec4 temppos;
layout(location = 3) out vec4 spatw;
layout(location = 4) out vec4 spatl;

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

  vec4 a = texture2D(tempWeights, texCoord * 1.);
  vec4 a1 = texture2D(tempOutL, texCoord * 1.);
  vec4 a2 = texture2D(tempPos, texCoord * 1.);
  vec4 a3 = texture2D(spatWe, texCoord * 1.);
  vec4 a4 = texture2D(spatLO, texCoord * 1.);
  /*
  if(ProjectedCoordinates.x > 1. || ProjectedCoordinates.x < 0. ||
  ProjectedCoordinates.y > 1. || ProjectedCoordinates.y < 0. ||
  length(cameraOffset) > 0.01){ a = vec4(1.); a1 = vec4(0.); a2 = vec4(0.); a3 =
  vec4(1.); a4 = vec4(0.);
  }*/
  tempwe = a;
  templ = a1;
  temppos = a2;
  spatw = a3;
  spatl = a4;

  // prevPosition = texture2D(position, texCoord*1.5);
}