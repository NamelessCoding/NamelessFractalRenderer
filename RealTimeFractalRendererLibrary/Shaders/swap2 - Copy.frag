#version 330

layout(location = 0) out float DEPTH;
layout(location = 1) out vec2 motionVec;

layout(location = 2) out vec4 TAA;

in vec2 texCoord;

uniform sampler2D pos;
uniform sampler2D taa;

uniform mat4 prevview;
uniform mat4 prevproj;
uniform mat4 view;
uniform mat4 proj;
float linearize_depth(float d,float zNear,float zFar)
{
    return zNear * zFar / (zFar + d * (zNear - zFar));
}
uniform vec2 wh;
void main() {

  vec4 a = texture2D(pos, texCoord * 1.);
  vec4 a1 = texture2D(taa, texCoord * 1.);


  //tempwe = a;
 //templ = a1;
  //temppos = a2;

  vec4 ProjectedC = vec4(a.xyz, 1.); // - vec4(off, 0.);
  ProjectedC = view * ProjectedC;
  ProjectedC = proj * ProjectedC;
  ProjectedC /= ProjectedC.w;

  vec4 Projected = vec4(a.xyz, 1.); // - vec4(off, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  // Projected.xy -= smallOffset;

  vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;

motionVec = vec2((ProjectedCoordinates)-(texCoord))*wh;

  DEPTH = linearize_depth(ProjectedC.z, 0.01, 100.);

TAA = a1;
}