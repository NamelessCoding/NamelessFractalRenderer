#version 430


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


float box(vec3 p, vec3 c){
	vec3 a = abs(p)-c;
	return max(a.x, max(a.y, a.z));

}

vec3 pal(float t, vec3 a){
return 0.5 + 0.5*cos(2.*3.14159*t + a);
}

float escape = 0.;


#define SceneRadius 100.0
#define DetailLevel 4.
#define StepFactor 1.

#define Pi 3.14159265359

float seed;

mat3 rotationMatrix(vec3 rotEuler){
    float c = cos(rotEuler.x), s = sin(rotEuler.x);
    mat3 rx = mat3(1, 0, 0, 0, c, -s, 0, s, c);
    c = cos(rotEuler.y), s = sin(rotEuler.y);
    mat3 ry = mat3(c, 0, -s, 0, 1, 0, s, 0, c);
    c = cos(rotEuler.z), s = sin(rotEuler.z);
    mat3 rz = mat3(c, -s, 0, s, c, 0, 0, 0, 1);
    
    return rz * rx * ry;
}
void fold(inout vec3 z, vec3 o, vec3 n) {
    z -= 2.*n*min(dot(z-o, n), 0.);
}

float sdfIFS(vec3 z, inout float ls){
    
	float scale = 2.;
    int Iterations = 20;
    mat3 rot = rotationMatrix(vec3(.5)*Pi);
    ls = 0.;
    vec3 n1 = normalize(vec3(1., 1., -1.));
    vec3 n2 = normalize(vec3(1., -1., -1.));
    //vec3 n3 = normalize(vec3(1., -1., 1.));
    escape = 0.;
    vec3 ot = vec3(1.);
    for(int i = 0; i < Iterations; i++) {
        fold(z, vec3(-.0), n1);
        fold(z, vec3(-.25), n2);
        //fold(z, vec3(-.25), n3);
        n1 *= rot;
        z = z*scale - sign(z)*(scale-1.0);
        ot = min(abs(z), ot);
        escape += exp(-0.2*dot(z,z));
    }
   // surf = Surface(true, 1.0, .1, vec3(0.), vec3(.8));
    //if(ot.r >= .75) surf = Surface(false, 1., .0, ot.ggb*30.*vec3(12., 2., .5), vec3(0.1));
        if(ot.r >= .35) ls = ot.g*3.;

    return length(z) * pow(scale, float(-Iterations));
}


float sdf(in vec3 pos, inout float ls) {
    float sSc = length(pos)-SceneRadius;
    float s = sdfIFS(pos, ls);
    return abs(sSc) > abs(s) || s > 0. ? s : sSc;
    
}

float sdf2(vec3 pos, inout float ls){
    //Surface surf;
    return sdf(pos,ls);
}
	
  float de23(vec3 p0, inout float ls){
    vec4 p = vec4(p0, 1.);
ls = 0.;
    p.xyz=abs(p.xyz);
    if(p.x < p.z)p.xz = p.zx;
    if(p.z < p.y)p.zy = p.yz;
    if(p.y < p.x)p.yx = p.xy;
    for(int i = 0; i < 8; i++){
      if(p.x < p.z)p.xz = p.zx;
      if(p.z < p.y)p.zy = p.yz;
      if(p.y > p.x)p.yx = p.xy;
      p.xyz = abs(p.xyz);
      p*=(1.8/clamp(dot(p.xyz,p.xyz),.0,1.));
      p.xyz-=vec3(3.6,1.9,0.5);
      if(dot(p.xyz,p.xyz)<0.1){
        ls += 0.1;
      }
    }
    float m = 1.5;
    p.xyz-=clamp(p.xyz,-m,m);
    return length(p.xyz)/p.w;
  }
 float de33( vec3 p, inout float ls ){
    p = p.xzy;
    vec3 cSize = vec3(1., 1., 1.3);
    float scale = 1.;
    ls = 0.;
    for( int i=0; i < 12; i++ ){
      p = 2.0*clamp(p, -cSize, cSize) - p;
      float r2 = dot(p,p+sin(p.z*.3));
      float k = max((2.)/(r2), .027);
      p *= k;  scale *= k;
      if(dot(p,p)<0.1)ls+=0.1;
    }
    float l = length(p.xy);
    float rxy = l - 4.0;
    float n = l * p.z;
    rxy = max(rxy, -(n) / 4.);
    return (rxy) / abs(scale);
  }
vec3 fold2(vec3 p0){
vec3 p = p0;
//if(abs(p.x) > 1.)p.x = 1.0-p.x;
//if(abs(p.y) > 1.)p.y = 1.0-p.y;
//if(abs(p.z) > 1.)p.z = 1.0-p.z;
if(length(p) > 2.)return p;
p = mod(p,2.)-1.;

return p;
}


float DE33(vec3 p0, inout float ls){
    vec4 p = vec4(p0, 1.);
    escape = 0.;
    ls = 0.;
    for(int i = 0; i < 12; i++){
        //p.xyz = clamp(p.xyz, vec3(-2.3), vec3(2.3))-p.xyz;
        //p.xyz += sin(float(i+1));
        if(p.x > p.z)p.xz = p.zx;
        if(p.z > p.y)p.zy = p.yz;
        if(p.y > p.x)p.yx = p.xy;
        p = abs(p);
        //p.xyz = fold(p.xyz);
                p.xyz = fold2(p.xyz);


       // p.xyz = fract(p.xyz*0.5 - 1.)*2.-1.0;
        p.xyz = mod(p.xyz-1., 2.)-1.;
                p*=(1.0/clamp(dot(p.xyz,p.xyz),0.1,1.));

        //p.xyz-=vec3(0.1,0.4,0.2);
        //p*=1.2;
        escape += exp(-0.2*dot(p.xyz,p.xyz));
        if(dot(p.xyz,p.xyz)<0.1){
            ls += 0.1;
        }
    }
    p/=p.w;
    return length(p.xz)*0.25;
}


vec3 fold3(vec3 p0){
vec3 p = p0;
//if(abs(p.x) > 1.)p.x = 1.0-p.x;
//if(abs(p.y) > 1.)p.y = 1.0-p.y;
//if(abs(p.z) > 1.)p.z = 1.0-p.z;
if(length(p) > 2.)return p;
p = mod(p,2.)-1.;

return p;
}

//float escape;
float DE32(vec3 p0, inout float ls){
    ls = 0.;
    vec4 p = vec4(p0, 1.);
    escape = 0.;
    for(int i = 0; i < 12; i++){
        //p.xyz = clamp(p.xyz, vec3(-2.3), vec3(2.3))-p.xyz;
        //p.xyz += sin(float(i+1));
        if(p.x > p.z)p.xz = p.zx;
        if(p.z > p.y)p.zy = p.yz;
        p = abs(p);
        p.xyz = fold3(p.xyz);

        //p.xyz = fract(p.xyz*0.5 - 1.)*2.-1.0;
        p.xyz = mod(p.xyz-1., 2.)-1.;
        p*=(1.05/dot(p.xyz,p.xyz));
        //p*=1.2;
        escape += exp(-0.2*dot(p.xyz,p.xyz));

    }
    p/=p.w;
    return length(p.xz)*0.25;
}

void boxFold(inout vec4 p, vec3 s) {
	p.xyz = clamp(p.xyz, -s, s) * 2. - p.xyz;
}

void sphereFold(inout vec4 p, float mr, float fr) {
	p *= fr / clamp(dot(p.xyz, p.xyz), mr, fr);
}

/*
 float Scale = 1.29842932;
 vec3 Offset = vec3(-3.2189972,-0.9234828,-4.5382586);
 vec3 BoxScale = vec3(4.4101877,10.,0.1742627);
 vec3 BoxFold = vec3(2.,0.33821572,0.40479362);
 vec3 Rotation = vec3(90.,0.,90.);
 int Iterations = 24;
 float MinRadius = 22.807018;
 float FixedRadius = 30.3693575;


vec4 orbitTrap = vec4(9999.);
float DEf(vec3 pos, inout float ls) {
	vec3 c = cos(Rotation*TWO_PI/360.);
	vec3 s = sin(Rotation*TWO_PI/360.);
	mat3 rx = mat3(1, 0, 0, 0, c.x, s.x, 0, -s.x, c.x);
	mat3 ry = mat3(c.y, 0, s.y, 0, 1, 0, -s.y, 0, c.y);
	mat3 rz = mat3(c.z, s.z, 0, -s.z, c.z, 0, 0, 0, 1);
	mat3 rot = rx*ry*rz;
orbitTrap = vec4(9999.);
	vec4 p = vec4(pos, 1);
ls = 0.;
	float trap = SceneRadius*2.;
	
	for (int i = 0; i < Iterations; i++) {
		orbitTrap = min(orbitTrap, vec4(abs(p.xyz), length(p.xyz)/p.w));
		sphereFold(p, MinRadius, FixedRadius);
		p.xyz += Offset;
		boxFold(p, BoxFold);
		p.xyz *= rot;
		//p.xyz = p.zxy * vec3(1, -1, -1);
		p *= Scale;
        if(dot(p.xyz,p.xyz)<3.9){
            ls += 0.1;
        }
		trap = min(boxDE(p, BoxScale), trap);
	}
	return trap;
}*/
/*
 float Scale; slider[1,1.5,5]
 vec3 Offset; slider[(-10,-10,-10),(1,1,1),(10,10,10)]
 vec3 BoxScale; slider[(0,0,0),(1,1,1),(10,10,10)]
 vec3 BoxFold; slider[(0,0,0),(.1,.1,.1),(2,2,2)]
 vec3 Rotation; slider[(0,0,0),(0,0,0),(360,360,360)]
 int Iterations; slider[1,16,64]
 float MinRadius; slider[0,1,50]
 float FixedRadius; slider[0,10,50]
*/
float boxDE(vec4 p, vec3 s) {
    vec3 a = abs(p.xyz) - s;
    return (min(max(max(a.x, a.y), a.z), 0.) + length(max(a, 0.))) / p.w;
}

float TWO_PI = 2.*3.14159;
float Scale = 1.67517732;
vec3 Offset = vec3(4.1917026,1.2732476,-5.9370528);
vec3 BoxScale = vec3(0.4949054,0.2474527,10.);
vec3 BoxFold = vec3(0.6300578,0.81213874,1.04046244);
vec3 Rotation = vec3(90.,270.,81.509436);
int Iterations = 16;
float MinRadius = 5.2785925;
float FixedRadius = 28.1250005;

float DE3333(vec3 pos, inout float ls) {
    ls = 0.;
	vec3 c = cos(Rotation.xxy*TWO_PI/360.);
	vec3 s = sin(Rotation.xxy*TWO_PI/360.);
	mat3 rx = mat3(1, 0, 0, 0, c.x, s.x, 0, -s.x, c.x);
	mat3 ry = mat3(c.y, 0, s.y, 0, 1, 0, -s.y, 0, c.y);
	mat3 rz = mat3(c.z, s.z, 0, -s.z, c.z, 0, 0, 0, 1);
	mat3 rot = rx*ry*rz;

	vec4 p = vec4(pos, 1);

	float trap = SceneRadius*2.;
	
	for (int i = 0; i < Iterations; i++) {
		//orbitTrap = min(orbitTrap, vec4(abs(p.xyz), length(p.xyz)/p.w));
		sphereFold(p, MinRadius, FixedRadius);
		p.xyz += Offset;
		boxFold(p, BoxFold);
		p.xyz *= rot;
		p *= Scale;
		trap = min(boxDE(p, BoxScale), trap);
                if(dot(p.xyz, p.xyz)<0.2){ls+=0.1;}

	}
	return trap;
}

float sdBox( vec3 p, vec3 b )
{
  vec3  di = abs(p) - b;
  float mc = max(di.x,max(di.y,di.z));
  return min(mc,length(max(di,0.0)));
}
 float l  = 0.;
  float rough = 1.;
  vec3 cccc = vec3(1.);
float fractal(vec3 p){
vec3 w = p;
    vec3 q = p;

    q.xz = mod( q.xz+1.0, 2.0 ) -1.0;
    
    float d = sdBox(q,vec3(1.0));
    float s = 1.0;
    for( int m=0; m<6; m++ )
    {
        float h = float(m)/6.0;

        p =  q - 0.5*sin( abs(p.y) + float(m)*3.0+vec3(0.0,3.0,1.0));

        vec3 a = mod( p*s, 2.0 )-1.0;
        s *= 3.0;
        vec3 r = abs(1.0 - 3.0*abs(a));

        float da = max(r.x,r.y);
        float db = max(r.y,r.z);
        float dc = max(r.z,r.x);
        float c = (min(da,min(db,dc))-1.0)/s;

        d = max( c, d );
   }

   vec2 res = vec2(d,1.0);
    
   {
   d = length(w-vec3(0.22,0.35,0.4)) - 0.09;
   if( d<res.x ) res=vec2(d,2.0);
   }
   
   {
   d = w.y + 0.22;
   if( d<res.x ) res=vec2(d,3.0);
   }
    
    if( res.y<1.5 )
        {
        cccc= vec3(0.38)*vec3(1.2,0.8,0.6);
        
        }
        else if( res.y<2.5 )
        {
        //surfColor = vec3(0.37);
        cccc = vec3(1.);
        rough = 0.001;
        }
        else //if( tm.y<2.5 )
        {
        cccc = vec3(0.38)*vec3(1.2,0.8,0.6);
        
        }


   return res.x;
}
float escape2;
float fractal_de15(vec3 p){
    p=abs(p)-1.2;
    if(p.x<p.z)p.xz=p.zx;
    if(p.y<p.z)p.yz=p.zy;
    if(p.x<p.y)p.xy=p.yx;
escape2 = 0.;
    float s=1.;
    for(int i=0;i<6;i++)
    {
      p=abs(p);
      float r=2./clamp(dot(p,p),.1,1.);
      s*=r;
      p*=r;
      p-=vec3(.6,.6,3.5);
              escape2+=exp(-0.2*dot(p,p));

    }
    float a=1.5;
    p-=clamp(p,-a,a);
    return length(p)/s;
}

float fractal_de46(vec3 p){
    float s = 2.;
    float e = 0.;
    escape = 0.;
    for(int j=0;++j<7;){
        p.xz=abs(p.xz)-2.3,
        p.z>p.x?p=p.zyx:p,
        p.z=1.5-abs(p.z-1.3+sin(p.z)*.2),
        p.y>p.x?p=p.yxz:p,
        p.x=3.-abs(p.x-5.+sin(p.x*3.)*.2),
        p.y>p.x?p=p.yxz:p,
        p.y=.9-abs(p.y-.4),
        e=12.*clamp(.3/min(dot(p,p),1.),.0,1.)+
        2.*clamp(.1/min(dot(p,p),1.),.0,1.),
        p=e*p-vec3(7,1,1),
        s*=e;
        escape+=exp(-0.2*dot(p,p));
        }
    return length(p)/s;
}
 float desss(vec3 p){
    vec4 o=vec4(p,1);
    vec4 q=o;
    for(float i=0.;i<9.;i++){
      o.xyz=clamp(o.xyz,-1.,1.)*2.-o.xyz;
      o=o*clamp(max(.25/dot(o.xyz,o.xyz),.25),0.,1.)*vec4(11.2)+q;
    }
    return (length(o.xyz)-1.)/o.w-5e-4;
  }

vec3 ldir = normalize(vec3(0., 0.9, 0.4));


float jb(vec3 p){
    float s=3., e;
    s*=e=3./min(dot(p,p),50.);
    p=abs(p)*e;
    escape = 0.;
    for(int i=0;i++<12;){
        p=vec3(2,4,2)-abs(p-vec3(4,4,2)),
            s*=e=8./min(dot(p,p),9.),
            p=abs(p)*e;
            escape += exp(-0.2*dot(p,p));
            }
    return min(length(p.xz)-.1,p.y)/s;
}
/*
float map(vec3 p){
    float ls = 0.;
        p = p.xzy;

float c = 32.1;
    vec3 l33 = vec3(3., 1., 3.);
//   p = p-c*clamp(round(p/c),-l33,l33);

l = 0.;
rough = .1;
cccc = vec3(1.);

    float aa3 = fractal_de15((vec3(-15.0,-2.7,3.0)-p)/10.)*10.;
    float aa = fractal_de46((vec3(-5.0,25.0,13.0)-p)/10.)*10.;
    float sphere = length(p)-1.9;
    //float final = min(min(min(min(min(min(b2, a),b),c1),c2),c3),aa);
    float final = min(min(aa,sphere),aa3);
   // if(final == b2)l = 3.;
    //if(final == c1)c = vec3(0.9,0.1,0.1);
    //if(final == c2)c = vec3(0.1,0.9,0.1);
    if(final == aa){rough = 1.0;cccc=pal(escape*2.,vec3(0.8,0.4,0.4));}
    if(final == aa3){rough = 0.1;cccc=pal(escape2,vec3(0.4,0.4,0.8));}
    if(final == sphere){rough = 0.001;}

     return final;

     //return fractal_de46(p/5.)*5.;
}*/

float map(vec3 p){
	l = 0.;
    rough = 1.;
	cccc = vec3(1.);
	//return box(mod(p, 12.)-6., vec3(1.));
    float a = jb(p/6.)*6.;

    float b = box(p, vec3(.7, 15., .7));
    float c = length(vec3(5., 0., 0.)-p)-5.;
    float final =min(a,min(b,c));
    if(final == b){
        l = 2.;
        cccc = vec3(0.9, 0.6, 0.2);
    }
    if(final == a){
      //  l=escape*0.006;
      rough = 0.1;
      	cccc = pal(escape, vec3(0.9,0.6,0.2));

    }
    if(final == c){
        cccc = vec3(1.0);
        rough = 0.01;
        //rough = ;
    }
return final;

}



#define eps 0.01
#define size vec3(2.0,2.0,2.0)
#define roundr 0.25
#define global_k 1.25
#define Lipschitz_GridSize 0.1

vec3 calcGradient( in vec3 pos )
{
    const vec3 v1 = vec3(1.0,0.0,0.0);
    const vec3 v2 = vec3(0.0,1.0,0.0);
    const vec3 v3 = vec3(0.0,0.0,1.0);
	return (vec3(map(pos + v1*eps),map(pos + v2*eps),map(pos + v3*eps))
           -vec3(map(pos - v1*eps),map(pos - v2*eps),map(pos - v3*eps)))/(2.0*eps);
}
vec3 calcGradientCheap( in vec3 pos ,in float original)
{
    const vec3 v1 = vec3(1.0,0.0,0.0);
    const vec3 v2 = vec3(0.0,1.0,0.0);
    const vec3 v3 = vec3(0.0,0.0,1.0);
	return (vec3(map(pos + v1*eps),map(pos + v2*eps),map(pos + v3*eps))
           -vec3(original))/(eps);
}
//2 samples
vec2 sceneK_2(vec3 start,vec3 end,vec3 dir)
{
    float dist = 0.0;
    dist = map(start);
    float dist2 = map(end);

    float lam = abs(dist - dist2)/length(start-end);
    
    
    return vec2(lam,dist);
}
vec2 sceneK_13(vec3 start,vec3 end,vec3 dir)
{
    float dist = 0.0;
    dist = map(start);
    
    vec3 gs = calcGradient(start);
    vec3 ge = calcGradient(end);
    //vec3 gs = calcGradientCheap(start,dist);
    //vec3 ge = calcGradientCheap(end,dist);
    float fds = abs(dot(gs,dir));
    float fde = abs(dot(ge,dir));
    float lam = max(fds,fde);
    
    
    return vec2(lam,dist);
}
float calcGradientDir( in vec3 pos ,vec3 dir, in float original)
{
	return abs(map(pos + dir*eps) - original)/(eps);
}
float calcGradientDirCentered( in vec3 pos ,vec3 dir){
    return abs(map(pos + dir*eps) - map(pos - dir*eps))/(eps*2.0);
}

vec2 sceneK_4(vec3 start,vec3 end,vec3 dir)
{
    float dist = 0.0;
    dist = map(start);
    
    float fds = calcGradientDir(start,dir,dist);
    float fde = calcGradientDirCentered(end,dir);

    float lam = max(fds,fde);
    
    
    return vec2(lam,dist);
}





vec3 norm(vec3 p){
	return normalize(
		vec3(
			map(vec3(p.x+0.02, p.yz))-map(vec3(p.x-0.02, p.yz)),
			map(vec3(p.x, p.y+0.02, p.z))-map(vec3(p.x, p.y-0.02, p.z)),
			map(vec3(p.x, p.y, p.z+0.02))-map(vec3(p.x, p.y, p.z-0.02))
		)
	);

}

/*
float t = 0.01;
    for( int i=0; i<512; i++ )
    {
	    float precis = 0.001 * t;
        
	    float h = map( ro+rd*t, s );
        if( h<precis||t>maxd ) break;
        t += h;
*/
float k = 0.;
bool trace(inout vec3 p, vec3 d){
    vec3 cam = p;
    float t = 0.01;
    k = 0.;
	for(int i = 0; i < 280; i++){
        //p = cam + d*t;

		float dist = map(p);
       // float precis = 0.001 * t;

		if(dist < 0.01){
			return true;
		}
        if(length(cam-p) > 60.){
            return false;
        }
        if(dist < 0.1){
            k += 1.0;
        }
        //t += dist;
        p += d*dist;
	}
	return false;
}


bool trace2( inout vec3 ro, in vec3 rd )
{
    float mint = 0.0;
    float maxt = 160.0;
	float t = mint;
	float c = 2.0;
	float ts = (maxt - mint);
    ts = min(ts,Lipschitz_GridSize);
	for(int i=0;i<212;i++)
	{
        vec3 pt = ro+rd*t;
		vec3 pts = ro+rd*(t + ts);
        vec2 data = sceneK_2(pt,pts,rd);
		float dist = data.y;
		if (dist < 0.01)
        {
            ro = pt;
			return true;
        }
		float k = data.x;
		float tk = abs(dist) / max(k,0.01);
		tk = max(abs(dist)/global_k,min(tk, ts));
		ts = tk;
		if(tk >= 0.0)
		{
			t += max(tk, eps);
		}
		ts = tk * c;
        ts = min(ts,Lipschitz_GridSize);
        if(t > maxt)
        {
            return false;
        }
	}
	return false;
}


vec3 F(float costheta, float y, vec3 A, vec3 B, vec3 C, vec3 D, vec3 E){
return (1.+A*exp(B/costheta))*(1.0+C*exp(D*y) + E*cos(y)*cos(y));
}

vec3 skyp2(vec3 d, vec3 lig){//my code to begin with
vec3 zenith = vec3(0.,0.,1.);
float costheta = max(dot(d,zenith),0.);
float costhetas = max(dot(lig,zenith),0.);
float cosy = max(dot(lig,d),0.);
float y = acos(cosy);
//return vec3(0.1);
//if(costheta<0.01)return vec3(0.);
//simple cie sky
float T = 3.;
float X = (4./9. - T/120.)*(3.14159-2.*acos(costhetas));
float Yz = (4.0453*T - 4.9710)*tan(X)-0.2155*T+2.4192;

//vec3 template = vec3(*T+,*T+,*T+);
vec3 AYxy = vec3(0.1787*T-1.4630, -0.0193*T-0.2592,-0.0167*T-0.2608);
vec3 BYxy = vec3(-0.3554*T+0.4275,-0.0665*T+0.0008,-0.0950*T+0.0092);
vec3 CYxy = vec3(-0.0227*T+5.3251,-0.0004*T+0.2125,-0.0079*T+0.2102);
vec3 DYxy = vec3(0.1206*T-2.5771,-0.0641*T-0.8989,-0.0441*T-1.6537);
vec3 EYxy = vec3(-0.0670*T+0.3703,-0.0033*T+0.0452,-0.0109*T+0.0529);

float ts = acos(costhetas);
float ts2 = ts*ts;
float ts3 = ts*ts*ts;
vec3 xz0 = vec3(0.00166*ts3 -0.00375*ts2 + 0.00209*ts,
-0.02903*ts3 + 0.06377*ts2 - 0.03202*ts + 0.00394,
0.11693*ts3 - 0.21196*ts2 + 0.06052*ts + 0.25886
);

vec3 yz0 = vec3(0.00275*ts3 -0.00610*ts2 + 0.00317*ts,
-0.04214*ts3 + 0.08970*ts2 - 0.04153*ts + 0.00516,
0.15346*ts3 - 0.26756*ts2 + 0.06670*ts + 0.26688
);

float xz = xz0.x*T*T + xz0.y*T + xz0.z;
float yz = yz0.x*T*T + yz0.y*T + yz0.z;

vec3 Yxyz = vec3(Yz,xz,yz);
//test
//vec3 test1 = F(costheta, y, AYxy, BYxy, CYxy, DYxy, EYxy);
vec3 Ftop = F(costheta, y, AYxy, BYxy, CYxy, DYxy, EYxy);
vec3 Fbottom = F(1., ts, AYxy, BYxy, CYxy, DYxy, EYxy);

vec3 finalYxy = Yxyz*(Ftop/Fbottom);

vec3 XYZ = vec3(
(finalYxy.y*finalYxy.x)/finalYxy.z,
finalYxy.x,
((1.-finalYxy.y - finalYxy.z)*finalYxy.x)/finalYxy.z
);

vec3 rgb = vec3(
3.2404542*XYZ.x -1.5371385*XYZ.y -0.4985314*XYZ.z,
-0.9692660*XYZ.x + 1.8760108*XYZ.y + 0.0415560*XYZ.z,
0.0556434*XYZ.x - 0.2040259*XYZ.y + 1.0572252*XYZ.z
);

//return test1*0.1;
return rgb*0.034 + exp(-y*20.)*vec3(0.9,0.6,0.2);
}

vec3 angledircos(vec3 n, inout uint r){
    float r1 = rndf(r);
    float r2 = rndf(r);

    float x = cos(2.*3.14159*r1)*sqrt(1.-r2);
    float y = sin(2.*3.14159*r1)*sqrt(1.-r2);
    float z = sqrt(r2);

    vec3 W = (abs(n.x)>0.99)?vec3(0.,1.,0.):vec3(1.,0.,0.);
    vec3 N = n;
    vec3 T = normalize(cross(N,W));
    vec3 B = cross(T,N);
    return normalize(x*T + y*B + z*N);
}

float ggx_D(vec3 m, vec3 n, float a){
float top = a*a;
float bottom = 3.14159*pow((a*a-1.)*(max(dot(m,n),0.)*max(dot(m,n),0.))+1.,2.);
return top/bottom;
}

float ggx_pdf(vec3 m, vec3 n, float a){
float top = a*a*max(dot(m,n),0.);
float bottom = 3.14159*pow((a*a-1.)*max(dot(m,n),0.)*max(dot(m,n),0.)+1.,2.);
return top/bottom;
}

float ggx_pdf2(vec3 m, vec3 n, float a){
float ang = sin(acos(dot(m,n)));
float top = a*a*max(dot(m,n),0.)*ang;
float bottom = 3.14159*pow((a*a-1.)*max(dot(m,n),0.)*max(dot(m,n),0.)+1.,2.);
return top/bottom;
}

float ggx_G(vec3 h, vec3 n, vec3 wi, vec3 l, float a){
float g1 = (2.*max(dot(n,h),0.)*max(dot(n,-wi),0.))/max(dot(-wi,h),0.);
float g2 = (2.*max(dot(n,h),0.)*max(dot(n,l),0.))/max(dot(-wi,h),0.);
float G = min(1.,min(g1,g2));
return G;
}

float ggx_G2(vec3 h, vec3 n, vec3 wi, vec3 l, float a){
float top = 2.*max(dot(n,-wi),0.);
float bottom = max(dot(n,-wi),0.)+sqrt(a*a + (1.-a*a)*pow(max(dot(n,-wi),0.),2.));
return top/bottom;
}

vec3 ggx_F(vec3 Fo, float cost){
return Fo + (1.-Fo)*pow(1.-cost,5.);
}
//energy conserving
//vec3 ggx_F(vec3 Fo, vec3 v, vec3 n, vec3 l){
//return Fo + (1.-Fo)*pow(1.-cost,5.);
//}

vec3 ggx_S(vec3 n, inout uint r, float a){
        float r1 = rndf(r);
        float r2 = rndf(r);
    
        float theta = atan(a*sqrt(r1/(1.-r1)));
        //float theta = acos(sqrt((1.-r1)/(r1*(a*a-1.)+1.) ));
        float phi = 2.*3.14159*r2;
        
        float x = cos(phi)*sin(theta);
        float y = sin(phi)*sin(theta);
        float z = cos(theta); 
         
        vec3 W = (abs(n.x)>0.99)?vec3(0.,1.,0.):vec3(1.,0.,0.);
        vec3 N = n;
        vec3 T = normalize(cross(N,W));
        vec3 B = cross(T,N);
        return normalize(x*T + y*B + z*N);
}
bool traceVolume(inout vec3 p, vec3 d, inout uint r){
    vec3 cam = p;
    float t = 0.01;
	for(int i = 0; i < 280; i++){
        //p = cam + d*t;

		float dist = map(p);
       // float precis = 0.001 * t;
        if(rndf(r) < 0.05){
            d = ggx_S(d.xzy, r, 0.1).xzy;
        }
		if(dist < 0.01 && l > 0.01){
			return true;
		}
        if(length(cam-p) > 60.){
            return false;
        }
        //t += dist;
        p += d*dist;
	}
	return false;
}
