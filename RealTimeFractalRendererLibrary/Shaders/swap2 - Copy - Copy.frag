#version 330

layout(location = 0) out vec4 ups;


in vec2 texCoord;

uniform sampler2D col;


uniform vec2 wh;
const ivec2 I_ZERO = ivec2(0);
const ivec2 I_ONE  = ivec2(1);


vec3 ArrCol[16];
vec3 GetD1() {
    return
          abs(ArrCol[0] - ArrCol[5] ) + abs(ArrCol[1] - ArrCol[6] ) + abs(ArrCol[2]  - ArrCol[7])
        + abs(ArrCol[4] - ArrCol[9] ) + abs(ArrCol[5] - ArrCol[10]) + abs(ArrCol[6]  - ArrCol[11])
        + abs(ArrCol[8] - ArrCol[13]) + abs(ArrCol[9] - ArrCol[14]) + abs(ArrCol[10] - ArrCol[15]);
}
vec3 GetD2() {
    return
          abs(ArrCol[1] - ArrCol[4] ) + abs(ArrCol[2]  - ArrCol[5])  + abs(ArrCol[3]  - ArrCol[6])
        + abs(ArrCol[5] - ArrCol[8] ) + abs(ArrCol[6]  - ArrCol[9])  + abs(ArrCol[7]  - ArrCol[10])
        + abs(ArrCol[9] - ArrCol[12]) + abs(ArrCol[10] - ArrCol[13]) + abs(ArrCol[11] - ArrCol[14]);
}
void ArrayPixel(sampler2D _Tex,ivec2 iU) {
    iU -= 3;
    for (int y = 0;y < 4;y++) {
        for (int x = 0;x < 4;x++) {
            int id = y * 4 + x;
            ArrCol[id] = texelFetch(_Tex,(iU+ivec2(x,y)*2),0).rgb*255.;
        }
    }
}
vec3 DCCI_StepI(ivec2 iU){
	vec3 OutColor = vec3(0.);

    if(all(equal( iU&1 , I_ONE))){
		ArrayPixel(col,iU);
        vec3 d1 = GetD1(), d2 = GetD2();
        
        vec3 DownRightPixel = -ArrCol[3] + 9.*ArrCol[6] + 9.*ArrCol[9] - ArrCol[12];
        vec3 UpRightPixel = -ArrCol[15] + 9.*ArrCol[10] + 9.*ArrCol[5] - ArrCol[0];
        
        vec3 w1 = 1. / (1. + d1*d1*d1*d1*d1);
        vec3 w2 = 1. / (1. + d2*d2*d2*d2*d2);
        vec3 weight1 = w1 / (w1 + w2);
        vec3 weight2 = w2 / (w1 + w2);
        vec3 BlendPixel = DownRightPixel * weight1 + UpRightPixel * weight2;
        
        vec3 A = step(0.,((1. + d1) / (1. + d2)) - 1.15);
        vec3 B = step(0.,((1. + d2) / (1. + d1)) - 1.15);
		vec3 C = (1.-A)*(1.-B);
        
        OutColor = DownRightPixel*A + UpRightPixel*B + C*BlendPixel;
        
        OutColor /= 16.;
        OutColor /= 255.;
    }
    else{
    	OutColor = texelFetch(col,iU,0).rgb;
    }
    return OutColor;
}
void main() {

   	ivec2 iU = ivec2(texCoord*wh);

    ups = vec4(DCCI_StepI(iU),1.);


}