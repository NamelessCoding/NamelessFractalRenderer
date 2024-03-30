#version 330

layout(location = 0) out vec4 tempwe;
layout(location = 1) out vec4 templ;
layout(location = 2) out vec4 temppos;
layout(location = 3) out vec4 spatw;
layout(location = 4) out vec4 spatl;

in vec2 texCoord;

uniform sampler2D tempWeights;
uniform sampler2D tempOutL;
uniform sampler2D tempPos;
uniform sampler2D spatWe;
uniform sampler2D spatLO;

void main() {

  vec4 a = texture2D(tempWeights, texCoord * 1.);
  vec4 a1 = texture2D(tempOutL, texCoord * 1.);
  vec4 a2 = texture2D(tempPos, texCoord * 1.);
  vec4 a3 = texture2D(spatWe, texCoord * 1.);
  vec4 a4 = texture2D(spatLO, texCoord * 1.);

  tempwe = a;
  templ = a1;
  temppos = a2;
  spatw = a3;
  spatl = a4;
}