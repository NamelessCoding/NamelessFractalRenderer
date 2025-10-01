layout(binding = 3, Rgba32f) uniform image3D rcfog;

float RMS = 0.;

bool renderSkyAndSun = true;
bool removeWater = false;
bool useRadCache = true;

bool doGLass = false;

// vec3 sunColor = mix(vec3(0.9, 0.8, 0.5), vec3(0.9,0.85, 0.8),
// max(ldir.y,0.))*pow(max(dot(normalize(ldir), vec3(0., 1., 0.)),0.),3.);
vec3 sunColor = vec3(0.9, 0.8, 0.5);
float sunStrength = 2.25;
float skyStrength = 2.25;

float epsilon = 0.0021;
float sssStepSize = 0.5;


vec3 randomSpherePoint(vec2 rand) {
  float ang1 = (rand.x + 1.0) * 3.14159; // [-1..1) -> [0..2*PI)
  float u = rand.y; // [-1..1), cos and acos(2v-1) cancel each other out, so we
                    // arrive at [-1..1)
  float u2 = u * u;
  float sqrt1MinusU2 = sqrt(1.0 - u2);
  float x = sqrt1MinusU2 * cos(ang1);
  float y = sqrt1MinusU2 * sin(ang1);
  float z = u;
  return vec3(x, y, z);
}

vec3 HenyeyGreensteinSampleSphere(vec3 n, float g, inout uint r) {
  float t = (1.0 - g * g) / (1.0 - g + 2.0 * g * rndf(r));
  float cosTheta = (1.0 + g * g - t) / (2.0 * g);
  float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
  float phi = 2.0 * 3.14159 * rndf(r);

  vec3 xyz = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
  vec3 W = (abs(n.x) > 0.99) ? vec3(0., 1., 0.) : vec3(1., 0., 0.);
  vec3 N = n;
  vec3 T = normalize(cross(N, W));
  vec3 B = cross(T, N);
  return normalize(xyz.x * T + xyz.y * B + xyz.z * N);
}

bool traceVolume(inout vec3 p, vec3 d, inout uint r, inout vec3 tt,
                 inout vec3 secondPos, inout float isLight) {
  vec3 cam = p;
  float t = 0.01;
  secondPos = cam;
  isLight = 1.;
  bool isFIrst = false;
  for (int i = 0; i < 280; i++) {
    // p = cam + d*t;

    float dist = map(p,0.);
    // float precis = 0.001 * t;
    if (rndf(r) < clamp(0.05 * dist, 0.01, 0.95)) {
      //  if(rndf(r) < exp(-0.2*dist)){
      // d = ggx_S(d.xzy, r, 0.1).xzy;
      // d = randomSpherePoint(vec2(rndf(r)*2.-1.0, rndf(r)*2.-1.0));
      vec3 wi = d;
      d = HenyeyGreensteinSampleSphere(d.xzy, 0., r).xzy;

      if (!isFIrst) {
        secondPos = p;
        isLight = dot(wi, d);
      }
      isFIrst = true;
      tt *= exp(-0.001 * dist);
    }
    if (dist < 0.1 && l > 0.01) {
      return true;
    }
    if (length(cam - p) > 160.) {
      return false;
    }
    // if(i == 1){
    //     secondPos = p;

    // }
    // t += dist;
    p += d * dist;
  }
  return false;
}

#define eps 0.01
#define size vec3(2.0, 2.0, 2.0)
#define roundr 0.25
#define global_k 1.25
#define Lipschitz_GridSize 0.1

vec3 calcGradient(in vec3 pos) {
  const vec3 v1 = vec3(1.0, 0.0, 0.0);
  const vec3 v2 = vec3(0.0, 1.0, 0.0);
  const vec3 v3 = vec3(0.0, 0.0, 1.0);
  return (vec3(map(pos + v1 * eps,0.), map(pos + v2 * eps,0.), map(pos + v3 * eps,0.)) -
          vec3(map(pos - v1 * eps,0.), map(pos - v2 * eps,0.), map(pos - v3 * eps,0.))) /
         (2.0 * eps);
}
vec3 calcGradientCheap(in vec3 pos, in float original) {
  const vec3 v1 = vec3(1.0, 0.0, 0.0);
  const vec3 v2 = vec3(0.0, 1.0, 0.0);
  const vec3 v3 = vec3(0.0, 0.0, 1.0);
  return (vec3(map(pos + v1 * eps,0.), map(pos + v2 * eps,0.), map(pos + v3 * eps,0.)) -
          vec3(original)) /
         (eps);
}
// 2 samples
vec2 sceneK_2(vec3 start, vec3 end, vec3 dir) {
  float dist = 0.0;
  dist = map(start,0.);
  float dist2 = map(end,0.);

  float lam = abs(dist - dist2) / length(start - end);

  return vec2(lam, dist);
}
vec2 sceneK_13(vec3 start, vec3 end, vec3 dir) {
  float dist = 0.0;
  dist = map(start,0.);

  vec3 gs = calcGradient(start);
  vec3 ge = calcGradient(end);
  // vec3 gs = calcGradientCheap(start,dist);
  // vec3 ge = calcGradientCheap(end,dist);
  float fds = abs(dot(gs, dir));
  float fde = abs(dot(ge, dir));
  float lam = max(fds, fde);

  return vec2(lam, dist);
}
float calcGradientDir(in vec3 pos, vec3 dir, in float original) {
  return abs(map(pos + dir * eps,0.) - original) / (eps);
}
float calcGradientDirCentered(in vec3 pos, vec3 dir) {
  return abs(map(pos + dir * eps,0.) - map(pos - dir * eps,0.)) / (eps * 2.0);
}

vec2 sceneK_4(vec3 start, vec3 end, vec3 dir) {
  float dist = 0.0;
  dist = map(start,0.);

  float fds = calcGradientDir(start, dir, dist);
  float fde = calcGradientDirCentered(end, dir);

  float lam = max(fds, fde);

  return vec2(lam, dist);
}

vec3 norm(vec3 p, float times) {
  float k = epsilon * 2.;
  return normalize(
      vec3(map(vec3(p.x + k, p.yz),times) - map(vec3(p.x - k, p.yz),times),
           map(vec3(p.x, p.y + k, p.z),times) - map(vec3(p.x, p.y - k, p.z),times),
           map(vec3(p.x, p.y, p.z + k),times) - map(vec3(p.x, p.y, p.z - k),times)));
}

vec3 normWat(vec3 p, float times) {
  float k = epsilon * 2.;
  return normalize(
      vec3(mapPl(vec3(p.x + k, p.yz),times) - mapPl(vec3(p.x - k, p.yz), times),
           mapPl(vec3(p.x, p.y + k, p.z),times) - mapPl(vec3(p.x, p.y - k, p.z),times),
           mapPl(vec3(p.x, p.y, p.z + k),times) - mapPl(vec3(p.x, p.y, p.z - k), times)));
}

vec3 normGlass(vec3 p) {
  float k = epsilon * 2.;
  return normalize(
      vec3(glassMap(vec3(p.x + k, p.yz)) - glassMap(vec3(p.x - k, p.yz)),
           glassMap(vec3(p.x, p.y + k, p.z)) - glassMap(vec3(p.x, p.y - k, p.z)),
           glassMap(vec3(p.x, p.y, p.z + k)) - glassMap(vec3(p.x, p.y, p.z - k))));
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

vec2 boxIntersectionXX( in vec3 ro, in vec3 rd, vec3 boxSize )
{
    vec3 m = 1.0/rd;
    vec3 n = m*ro;
    vec3 k = abs(m)*boxSize;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );
    if( tN>tF || tF<0.0) return vec2(-1.0);
    return vec2( tN, tF );
}

vec3 someoneisreallyretarded(vec3 p){
  float k = epsilon*2.;
  return normalize(
    vec3(
        glassMap(vec3(p.x + k, p.yz))-glassMap(vec3(p.x - k, p.yz)),
        glassMap(vec3(p.x, p.y + k, p.z))-glassMap(vec3(p.x, p.y - k, p.z)),
        glassMap(vec3(p.xy, p.z + k))-glassMap(vec3(p.xy, p.z - k))
    )
  );
}


bool traceInit(inout vec3 p, vec3 d, inout vec3 watp, inout vec3 watn,
           bool renderWat, inout bool hitswat, inout float isk2,float times,
            inout vec3 glassP, inout vec3 glassN, inout bool hasHitGlass,
           vec3 viewPos = vec3(0.), bool stopatwat = false) {
  vec3 cam = p;
  // p += d*epsilon*2.;
  bool hitWat = false;
  watp = cam;
  watn = vec3(0.);
  hitswat = false;
  isk2 = 0.;
  float epsl = epsilon;
  vec3 keepWP = cam;


  bool hasHit = true;


  vec3 rayDir = d;
  vec3 rayPos = p;
 // ivec3 mapPos = ivec3(floor(rayPos + 0.));

  
  //vec3 prevPos = mapPos; 
  bool managedToHit = true;

  p = rayPos;
  
  float boostEps = 1.;
  
  if(managedToHit){
   // boostEps = 6.;
  }
  //float iterShit = 0.;

int iters = 180;
//if(!managedToHit){
 // iters = 60;
//}

  for (int i = 0; i < iters; i++) {
    // p = cam + d*t;

    float A = map(p,times);
    float dist = A;
    if (!hitWat && renderWat) {
      float B = mapPl(p, times);

      dist = min(dist, B);
    }

    if(!hasHitGlass && doGLass){
        float C = glassMap(p);

        dist = min(dist, C);
    } 

    // float precis = 0.001 * t;

    if (dist < epsl && abs(dist - A) < epsl && !hitWat && !hasHitGlass) {
      
      return true;
    }

     if (dist < epsl && !hasHitGlass && doGLass) {
      glassP = p;
      //keepWP = watp;
      glassN = someoneisreallyretarded(p - d * epsl);
      hasHitGlass = true;
     // hitWat = true;
      // if(dot(d, watn) < -0.1){
      //    d = normalize(refract(d, normalize(watn), 0.1));
      //}
      float indexofcantthinkcauseofsomeone = .87; 
      vec3 dir = refract(d, glassN, indexofcantthinkcauseofsomeone);
      vec3 pos = p+dir*0.1;
      for(int i = 0; i < 50; i++){
        if(glassMap(pos) > 0.01 && i > 4){
          break;
        }
        pos += dir*0.1;
      }
      //not gonna refract a second time for now
      p = pos;
      d = dir;
      //  continue;
    } else if (dist < epsl && abs(dist - A) < epsl) {
      return true;
    }

    if (dist < epsl && !hitWat && renderWat) {
      watp = p;
      keepWP = watp;
      watn = normWat(p - d * epsl, times);
      hitswat = true;
      hitWat = true;
      // if(dot(d, watn) < -0.1){
      //    d = normalize(refract(d, normalize(watn), 0.1));
      //}
      if (stopatwat) {
        return true;
      }
      //  continue;
    } else if (dist < epsl && abs(dist - A) < epsl) {
      return true;
    }

    if (length(cam - p) > 360.) {
      if (hitWat && renderWat) {
        isk2 = 1.;
        p = keepWP;
        return true;
      }
      return false;
    }

    // t += dist;
    epsl = epsilon * (1.+ boostEps*sqrt(length(cam-p))) ;
   // iterShit = exp(-A*1.1);
   //epsl = epsilon;
    //epsl = (epsilon * 20.) / (float(i+1));
    p += d * dist;
  }

  if (hitWat && renderWat) {
    isk2 = 1.;
    p = keepWP;
    return true;
  }
  /*if(managedToHit){
        float disss = length(prevPos - p)*0.5;
        ivec3 newP = ivec3((vec3(prevPos)*3. - floor(viewPos)*3. )+ 128.);
        //vec3 newDir = vec3(12., 0., 0.);
        vec3 norm = imageLoad(rcfog, ivec3(newP)).xyz;
        norm.z = 1. + disss;
        imageStore(rcfog, ivec3(newP), vec4(norm, imageLoad(rcfog, ivec3(newP)).w));
  }*/
  return false;
}

bool trace(inout vec3 p, vec3 d, inout vec3 watp, inout vec3 watn,
           bool renderWat, inout bool hitswat, inout float isk2,float times,
           vec3 viewPos = vec3(0.), bool stopatwat = false) {
  vec3 cam = p;
  // p += d*epsilon*2.;
  bool hitWat = false;
  watp = cam;
  watn = vec3(0.);
  hitswat = false;
  isk2 = 0.;
  float epsl = epsilon;
  vec3 keepWP = cam;


  bool hasHit = true;


  vec3 rayDir = d;
  vec3 rayPos = p;
 // ivec3 mapPos = ivec3(floor(rayPos + 0.));

  
  //vec3 prevPos = mapPos; 
  bool managedToHit = true;
  /*
  float prevDist = 0.;
  for(int i = 0; i < 60; i++){
    //- floor(viewPos)
    //- floor(viewPos)*2.
     ivec3 newP = ivec3((vec3(rayPos)*3. - floor(viewPos)*3. )+ 128.);
     vec3 normalCurrently = imageLoad(rcfog, newP).xyz;
     if(newP.x >= 256. || newP.x < 0. || newP.y >= 256. || newP.y < 0 || newP.z >= 256. || newP.z < 0){
        //return false;
        p = rayPos;
        break;
     }
     float distanc = imageLoad(rcfog, newP).w*0.79;
    float phi = normalCurrently.x;
    float theta = normalCurrently.y;
    vec3 normal = normalize(
      vec3(
        cos(phi)*sin(theta),
        cos(theta),
        sin(phi)*sin(theta)
      )
    );
     if(distanc < 0.3 ){
        distanc = 0.1;
       // if(dot(normal, rayDir) >= 0. && distanc > 0.2){
       //   distanc = 0.2;
       // }
     }
    // if(dot(normalCurrently, rayDir) >= 0. && distanc > 0.1 && distanc < 0.3){

      //  distanc *= 14.;
     //}
   // if(distanc < 1.){
    
//&& normalCurrently.z > 1.
       // if(dot(normal, rayDir) < 0.&& distanc > 0.1 && distanc < 0.3 ){
       ///     distanc *= 14.;
           // distanc = normalCurrently.z;
      //  }
    //}

    //if(dot(normal, rayDir) < 0.&& distanc > 0.1 && distanc < 0.3&& normalCurrently.z > 1. ){
            //distanc *= 14.;
        //    distanc = normalCurrently.z-1.;
       // }


     if(distanc < 0.15){
        //  p = rayPos + rayDir * I.x ;
          //float A = map(rayPos);
          //if(A < 0.05){
            //rayPos -= rayDir * prevDist;
            bool toexit = false;
            float A = 0.;
            for(int k = 0; k < 10; k++){
              A = map(rayPos)*1.;
              if(A < 0.05){
                toexit = true;
                break;
              }
              rayPos += rayDir*A;
            }
            
            //if(distanc > 0.1){
              
            //}
            distanc = A;
            if(toexit){
                 break;
            }else{
              rayPos += rayDir*0.15;
              managedToHit = false;
            }
           
         // }
          //else{
          //  rayPos += rayDir * 1.;
         // }
         
     }

     

     //if(abs(12. - directionSofar.x) < 0.01){
     //   distanc = 0.5;
     //}
      

   //  float simiral = max(dot(directionSofar, rayDir),0.);

      //if(directionSofar.x < 0.5){
      //  distanc = 10.1;
      //}

      //*exp(-length(worldCenter - rayPos))
      rayPos += rayDir * max(distanc, 0.);
      prevDist = distanc;

  }
  vec3 prevPos = rayPos;
  */
  p = rayPos;
  
  float boostEps = 1.;
  
  if(managedToHit){
   // boostEps = 6.;
  }
  //float iterShit = 0.;

int iters = 180;
//if(!managedToHit){
 // iters = 60;
//}

  for (int i = 0; i < iters; i++) {
    // p = cam + d*t;

    float A = map(p, times);
    float dist = A;
    if (!hitWat && renderWat) {
      float B = mapPl(p, times);

      dist = min(dist, B);
    }
    // float precis = 0.001 * t;

    if (dist < epsl && abs(dist - A) < epsl && !hitWat) {
      
      return true;
    }

    if (dist < epsl && !hitWat && renderWat) {
      watp = p;
      keepWP = watp;
      watn = normWat(p - d * epsl, times);
      hitswat = true;
      hitWat = true;
      // if(dot(d, watn) < -0.1){
      //    d = normalize(refract(d, normalize(watn), 0.1));
      //}
      if (stopatwat) {
        return true;
      }
      
    } else if (dist < epsl) {
      return true;
    }

    if (length(cam - p) > 360.) {
      if (hitWat && renderWat) {
        isk2 = 1.;
        p = keepWP;
        return true;
      }
      return false;
    }

    // t += dist;
    epsl = epsilon * (1.+ boostEps*sqrt(length(cam-p))) ;
   // iterShit = exp(-A*1.1);
   //epsl = epsilon;
    //epsl = (epsilon * 20.) / (float(i+1));
    p += d * dist;
  }

  if (hitWat && renderWat) {
    isk2 = 1.;
    p = keepWP;
    return true;
  }
  /*if(managedToHit){
        float disss = length(prevPos - p)*0.5;
        ivec3 newP = ivec3((vec3(prevPos)*3. - floor(viewPos)*3. )+ 128.);
        //vec3 newDir = vec3(12., 0., 0.);
        vec3 norm = imageLoad(rcfog, ivec3(newP)).xyz;
        norm.z = 1. + disss;
        imageStore(rcfog, ivec3(newP), vec4(norm, imageLoad(rcfog, ivec3(newP)).w));
  }*/
  return false;
}

// https://www.shadertoy.com/view/llKfD1
bool trace222(inout vec3 ro, vec3 rd, inout vec3 watp, inout vec3 watn,
              bool renderWat, inout bool hitswat,float times, bool stopatwat = false,
              float isk2 = 0.) {

  float w = 1.2;
  float dw = 1.8;
  float prevF = 0.0;
  float prevDt = 0.0;
  bool relaxed = true;
  float eps2 = epsilon;
  vec3 p = ro;

  float t = map(p,times);

  if (renderWat) {
    float B = mapPl(p,times);

    t = min(t, B);
  }

  float dt = 0.0;
  float stepCtr = 0.0;

  vec3 cam = p;
  // p += d*epsilon*2.;
  bool hitWat = false;
  watp = cam;
  watn = vec3(0.);
  hitswat = false;
  isk2 = 0.;

  for (int i = 0; i < 182; i++) {
    p = ro + rd * t;
    float A = map(p, times);
    float f = A;
    if (!hitWat && renderWat) {
      float B = mapPl(p,times);

      f = min(f, B);
    }

    dt = f * w;
    if (prevF + f < prevDt &&
        relaxed) // Reset to fallback position on excessive relaxation
    {
      relaxed = false;
      t += prevDt * (1.0 - w);
      p = ro + rd * t;
      float As = map(p,times);
      f = As;
      if (!hitWat && renderWat) {
        float B = mapPl(p,times);

        f = min(f, B);
      }
      dt = f * w;
    }

    if (t > 980.) {
      ro = p;
      if (hitWat && renderWat) {
        isk2 = 1.;
        // p = keepWP;
        return true;
      }
      return false;
    }

    if (f < eps2 && abs(f - A) < eps2 && !hitWat) {
      ro = p;
      return true;
    }

    if (f < eps2 && !hitWat && renderWat) {
      ro = p;
      watp = p;
      // keepWP = watp;
      watn = normWat(p - rd * eps2,times);
      hitswat = true;
      hitWat = true;
      // if(dot(d, watn) < -0.1){
      //    d = normalize(refract(d, normalize(watn), 0.1));
      //}
      if (stopatwat) {
        return true;
      }
    } else if (f < eps2) {
      ro = p;
      return true;
    } else {
      t += dt;
      prevF = f;
      prevDt = dt;
      if (relaxed) {
        w = mod(fract(w) * dw, 1.0) + 1.0;
      } // Escalate step scaling after each relaxation
      else {
        w = 1.2;
      }              // Otherwise re-set [w]
      eps2 *= 1.025; // Potential for further research here...
    }
    stepCtr += 1.0;
  }
  ro = p;
  return false;
}

bool traceSun2(inout vec3 ro, vec3 rd, float mxd) {

  float w = 1.2;
  float dw = 1.8;
  float prevF = 0.0;
  float prevDt = 0.0;
  bool relaxed = true;
  float eps2 = epsilon;
  vec3 p = ro;

  float t = map(p,0.);

  float dt = 0.0;
  float stepCtr = 0.0;

  for (int i = 0; i < 182; i++) {
    p = ro + rd * t;
    float A = map(p,0.);
    float f = A;

    dt = f * w;
    if (prevF + f < prevDt &&
        relaxed) // Reset to fallback position on excessive relaxation
    {
      relaxed = false;
      t += prevDt * (1.0 - w);
      p = ro + rd * t;
      float As = map(p,0.);
      f = As;

      dt = f * w;
    }

    if (f < eps2) {
      ro = p;
      return true;
    } else {
      t += dt;
      prevF = f;
      prevDt = dt;
      if (relaxed) {
        w = mod(fract(w) * dw, 1.0) + 1.0;
      } // Escalate step scaling after each relaxation
      else {
        w = 1.2;
      }              // Otherwise re-set [w]
      eps2 *= 1.025; // Potential for further research here...
    }
    stepCtr += 1.0;
  }
  ro = p;
  return false;
}

bool traceSun(inout vec3 p, vec3 d, float maxDist, float iTime) {
  vec3 cam = p;
  // p += d*epsilon*2.;
  p += d * 30.;
  float epsl = epsilon;

  for (int i = 0; i < 380; i++) {
    // p = cam + d*t;

    float dist = map(p, iTime);
    // float precis = 0.001 * t;

    if (dist < epsl) {
      return true;
    }
    if (length(cam - p) > maxDist) {
      return false;
    }

    // t += dist;
    p += d * dist;
    epsl += epsilon * 0.001;
  }
  return false;
}

bool traceShadow(inout vec3 p, vec3 d) {
  vec3 cam = p;
  // p += d*epsilon*2.;
  for (int i = 0; i < 80; i++) {
    // p = cam + d*t;

    float dist = map(p, 0.);
    // float precis = 0.001 * t;

    if (dist < epsilon * 4.) {
      return true;
    }
    if (length(cam - p) > 160.) {
      return false;
    }

    // t += dist;
    p += d * dist;
  }
  return false;
}

bool trace2(inout vec3 ro, in vec3 rd) {
  float mint = 0.0;
  float maxt = 160.0;
  float t = mint;
  float c = 2.0;
  float ts = (maxt - mint);
  ts = min(ts, Lipschitz_GridSize);
  for (int i = 0; i < 212; i++) {
    vec3 pt = ro + rd * t;
    vec3 pts = ro + rd * (t + ts);
    vec2 data = sceneK_2(pt, pts, rd);
    float dist = data.y;
    if (dist < 0.01) {
      ro = pt;
      return true;
    }
    float k = data.x;
    float tk = abs(dist) / max(k, 0.01);
    tk = max(abs(dist) / global_k, min(tk, ts));
    ts = tk;
    if (tk >= 0.0) {
      t += max(tk, eps);
    }
    ts = tk * c;
    ts = min(ts, Lipschitz_GridSize);
    if (t > maxt) {
      return false;
    }
  }
  return false;
}
