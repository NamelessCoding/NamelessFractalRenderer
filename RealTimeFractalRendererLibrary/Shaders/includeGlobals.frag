


bool renderSkyAndSun = true;
bool removeWater = true;
bool useRadCache = true;
//vec3 sunColor = mix(vec3(0.9, 0.8, 0.5), vec3(0.9,0.85, 0.8), max(ldir.y,0.))*pow(max(dot(normalize(ldir), vec3(0., 1., 0.)),0.),3.);
vec3 sunColor = vec3(0.9, 0.8, 0.5);
float sunStrength = 12.;
float skyStrength = 60.;



float epsilon = 0.01;



vec3 randomSpherePoint(vec2 rand) {
  float ang1 = (rand.x + 1.0) * 3.14159; // [-1..1) -> [0..2*PI)
  float u = rand.y; // [-1..1), cos and acos(2v-1) cancel each other out, so we arrive at [-1..1)
  float u2 = u * u;
  float sqrt1MinusU2 = sqrt(1.0 - u2);
  float x = sqrt1MinusU2 * cos(ang1);
  float y = sqrt1MinusU2 * sin(ang1);
  float z = u;
  return vec3(x, y, z);
}

vec3 HenyeyGreensteinSampleSphere(vec3 n, float g, inout uint r)
{
    float t = (1.0 - g * g) / (1.0 - g + 2.0 * g * rndf(r));
    float cosTheta = (1.0 + g * g - t) / (2.0 * g);
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    float phi = 2.0 * 3.14159 * rndf(r);
    
    vec3 xyz = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
        vec3 W = (abs(n.x)>0.99)?vec3(0.,1.,0.):vec3(1.,0.,0.);
        vec3 N = n;
        vec3 T = normalize(cross(N,W));
        vec3 B = cross(T,N);
        return normalize(xyz.x*T + xyz.y*B + xyz.z*N);
}

bool traceVolume(inout vec3 p, vec3 d, inout uint r, inout vec3 tt, inout vec3 secondPos, inout float isLight){
    vec3 cam = p;
    float t = 0.01;
    secondPos = cam;
    isLight = 1.;
    bool isFIrst = false;
	for(int i = 0; i < 280; i++){
        //p = cam + d*t;

		float dist = map(p);
       // float precis = 0.001 * t;
        if(rndf(r) < clamp(0.05*dist, 0.01, 0.95)){
        //  if(rndf(r) < exp(-0.2*dist)){
          // d = ggx_S(d.xzy, r, 0.1).xzy;
            //d = randomSpherePoint(vec2(rndf(r)*2.-1.0, rndf(r)*2.-1.0));
            vec3 wi = d;
            d = HenyeyGreensteinSampleSphere(d.xzy, 0., r).xzy;

            if(!isFIrst){
                secondPos = p;
                isLight = dot(wi, d);
            }
            isFIrst = true;
            tt *= exp(-0.001 * dist);
        }
		if(dist < 0.1 && l > 0.01){
			return true;
		}
        if(length(cam-p) > 160.){
            return false;
        }
        // if(i == 1){
        //     secondPos = p;

        // }
        //t += dist;
        p += d*dist;

	}
	return false;
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
    float k = epsilon*2.;
	return normalize(
		vec3(
			map(vec3(p.x+k , p.yz))-map(vec3(p.x-k , p.yz)),
			map(vec3(p.x, p.y+k , p.z))-map(vec3(p.x, p.y-k , p.z)),
			map(vec3(p.x, p.y, p.z+k ))-map(vec3(p.x, p.y, p.z-k ))
		)
	);

}


vec3 normWat(vec3 p){
    float k = epsilon*2.;
	return normalize(
		vec3(
			mapPl(vec3(p.x+k , p.yz))-mapPl(vec3(p.x-k , p.yz)),
			mapPl(vec3(p.x, p.y+k , p.z))-mapPl(vec3(p.x, p.y-k , p.z)),
			mapPl(vec3(p.x, p.y, p.z+k ))-mapPl(vec3(p.x, p.y, p.z-k ))
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
bool trace(inout vec3 p, vec3 d, inout vec3 watp, inout vec3 watn, bool renderWat, inout bool hitswat, bool stopatwat = false){
    vec3 cam = p;
    //p += d*epsilon*2.;
    bool hitWat = false;
    watp = cam;
    watn = vec3(0.);
    hitswat = false;

    vec3 keepWP = cam;
    for(int i = 0; i < 280; i++){
        //p = cam + d*t;

		float A = map(p);
        float B = mapPl(p);
        float dist = A;
        if(!hitWat && renderWat){
            dist = min(dist, B);
        }
       // float precis = 0.001 * t;

        if(dist < epsilon && abs(dist - A) < 0.001 && !hitWat){
            return true;
        }

		if(dist < epsilon && !hitWat && renderWat){
			watp = p;
            keepWP = watp;
            watn = normWat(p-d*epsilon);
            hitswat = true;
            hitWat = true;
            //if(dot(d, watn) < -0.1){
            //    d = normalize(refract(d, normalize(watn), 0.1));
            //}
            if(stopatwat){
                return true;
            }
          //  continue;
		}else if(dist < epsilon){
            return true;
        }

        if(length(cam-p) > 260. ){
            if(hitWat && renderWat){
                p = keepWP;
                return true;
            }
            return false;
        }

        //t += dist;
        p += d*dist;
	}
    if(hitWat && renderWat){
        p = keepWP;
        return true;
    }
	return false;
}

bool traceSun(inout vec3 p, vec3 d, float maxDist){
    vec3 cam = p;
    //p += d*epsilon*2.;
    p += d * 30.;
	for(int i = 0; i < 380; i++){
        //p = cam + d*t;

		float dist = map(p);
       // float precis = 0.001 * t;

		if(dist < epsilon  ){
			return true;
		}
        if(length(cam-p) > maxDist){
            return false;
        }

        //t += dist;
        p += d*dist;
	}
	return false;
}

bool traceShadow(inout vec3 p, vec3 d){
    vec3 cam = p;
    //p += d*epsilon*2.;
	for(int i = 0; i < 80; i++){
        //p = cam + d*t;

		float dist = map(p);
       // float precis = 0.001 * t;

		if(dist < epsilon*4.){
			return true;
		}
        if(length(cam-p) > 160.){
            return false;
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
