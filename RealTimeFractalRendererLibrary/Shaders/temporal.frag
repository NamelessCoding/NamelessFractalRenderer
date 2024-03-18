#version 430

layout (location = 0) out vec4 weigths;
layout (location = 1) out vec4 outrad;
layout (location = 2) out vec4 outpos;
layout (location = 3) out vec4 weigthsFog;
layout (location = 4) out vec4 outradFog;
layout (location = 5) out vec4 temporalPositionFog;

//temporalPositionFog
in vec2 texCoord;

uniform sampler2D color;
uniform sampler2D position;
uniform sampler2D normal;
uniform sampler2D albedo;
uniform sampler2D secondpos;
uniform sampler2D prevW;
uniform sampler2D prevL;
uniform sampler2D prevP;
uniform sampler2D prevN;
uniform sampler2D reflAlb;
uniform sampler2D prevPosition;
uniform sampler2D prevSecondPosition;
uniform sampler2D tempFog;
uniform sampler2D LOFog;
uniform sampler2D fog;
uniform sampler2D fogsecpos;
uniform sampler2D tempfogpos;
/*
_TemporalRestirShader.SetInt("tempFog", 12);
            _TemporalRestirShader.SetInt("LOFog", 13);
*/
/*
_TemporalRestirShader.SetInt("prevW", 5);
            _TemporalRestirShader.SetInt("prevL", 6);
            _TemporalRestirShader.SetInt("prevP", 7);
             */


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



float lum(vec3 c){
return sqrt( 0.299*c.x*c.x + 0.587*c.y*c.y + 0.114*c.z*c.z );
}






vec3 proj_point_in_plane(vec3 p, vec3 v0, vec3 n, out float d) {
 d = dot(n, p - v0);
 return p - (n * d);
}

vec3 find_reflection_incident_point(vec3 p0, vec3 p1, vec3 v0, vec3 n) {
 float d0 = 0;
 float d1 = 0;
 vec3 proj_p0 = proj_point_in_plane(p0, v0, n, d0);
 vec3 proj_p1 = proj_point_in_plane(p1, v0, n, d1);

 if(d1 < d0)
  return (proj_p0 - proj_p1) * d1/(d0+d1) + proj_p1;
 else
  return (proj_p1 - proj_p0) * d0/(d0+d1) + proj_p0;
}

vec2 find_previous_reflection_position(vec2 projected){
    vec3 v0 = texture2D(position, texCoord).xyz;
    vec3 n0 = texture2D(normal, texCoord).xyz;
    vec3 p0 = texture2D(secondpos, texCoord).xyz;
    vec2 motion_vector = projected;
    vec3 v1 = texture2D(prevPosition, projected).xyz;

    vec3 View = p0;
    vec4 Projected = vec4(View.xyz, 1.);// + vec4(cameraOffset, 0.);
    Projected = prevview * Projected;
    Projected = prevproj * Projected;
    Projected /= Projected.w;
    vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;
    vec3 p1 = texture2D(prevSecondPosition, ProjectedCoordinates).xyz;
    vec3 view_n = texture2D(prevN, ProjectedCoordinates).xyz;
    

    vec3 view_p0 = vec3(0.,0.,0.);
    vec3 view_v0 = v1;
    vec3 view_p1 = p1;

    vec3 view_intersection = 
  find_reflection_incident_point(view_p0, view_p1, view_v0, view_n);


    //vec3 ss_intersection = view_to_ss(view_intersection, 1);
    vec4 Projected2 = vec4(view_intersection.xyz, 1.);// + vec4(cameraOffset, 0.);
    Projected2 = prevview * Projected2;
    Projected2 = prevproj * Projected2;
    Projected2 /= Projected2.w;
    vec2 ProjectedCoordinates2 = Projected2.xy * 0.5 + 0.5;

    return ProjectedCoordinates2.xy;

}


vec2 motionVectorSpecular(vec3 pos, vec2 motionpos, vec2 texcord, vec3 prevpos){
    vec2 iResolution = wh;

    float stepSize = 4.;
    vec2 bestPos = texcord;
    vec2 centerPos = texcord;
    float bestDist = length(pos-prevpos);
    //vec3 pm = texture2D(color, texCoord - motionpos).xyz;
    vec2 m = texcord - motionpos;
    vec3 prevposm = texture2D(prevSecondPosition, m).xyz;
    if(length(pos - prevposm) < bestDist){
        bestDist = length(pos - prevposm);
        bestPos = m;
        centerPos = m;
    }


    for(int i = 0; i < 20; i++){
        vec2 q = (centerPos*iResolution + (vec2(1., 0.)*stepSize))/iResolution;
        vec3 prevposq = texture2D(prevSecondPosition, q).xyz;
        if(length(pos-prevposq) < bestDist){
            bestDist = length(pos-prevposq);
            bestPos = q;
        }

        q = (centerPos*iResolution + (vec2(-1., 0.)*stepSize))/iResolution;
        prevposq = texture2D(prevSecondPosition, q).xyz;
        if(length(pos-prevposq) < bestDist){
            bestDist = length(pos-prevposq);
            bestPos = q;
        }

        q = (centerPos*iResolution + (vec2(0., 1.)*stepSize))/iResolution;
         prevposq = texture2D(prevSecondPosition, q).xyz;
        if(length(pos-prevposq) < bestDist){
            bestDist = length(pos-prevposq);
            bestPos = q;
        }

       q = (centerPos*iResolution + (vec2(0., -1.)*stepSize))/iResolution;
         prevposq = texture2D(prevSecondPosition, q).xyz;
        if(length(pos-prevposq) < bestDist){
            bestDist = length(pos-prevposq);
            bestPos = q;
        }

       q = (centerPos*iResolution + (vec2(0., 0.)*stepSize))/iResolution;
         prevposq = texture2D(prevSecondPosition, q).xyz;
        if(length(pos-prevposq) < bestDist){
            bestDist = length(pos-prevposq);
            bestPos = q;
        }

        if(length(bestPos*iResolution - centerPos*iResolution) < 0.001){
            if(stepSize == 1.){
                break;
            }
            stepSize *= 0.5;
        }
        centerPos = bestPos;
    }


    for(int i = 0; i < 9; i++){
        //if(i == 4){continue;}
        vec2 offset = vec2(float(i%3)-1., float(i/3)-1.);
        vec3 prevposq = texture2D(prevSecondPosition, (centerPos*iResolution + offset)/iResolution).xyz;
        if(length(pos - prevposq) < bestDist){
            bestDist = length(pos - prevposq);
            bestPos = (centerPos*iResolution + offset)/iResolution;
        }
    }
    return bestPos;
}

/*


*/
vec3 ClipAABB(vec3 q,vec3 aabb_min, vec3 aabb_max){
	vec3 p_clip = 0.5 * (aabb_max + aabb_min);
	vec3 e_clip = 0.5 * (aabb_max - aabb_min) + 0.00000001;

	vec3 v_clip = q - vec3(p_clip);
	vec3 v_unit = v_clip.xyz / e_clip;
	vec3 a_unit = abs(v_unit);
	float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

	if (ma_unit > 1.0)
		return vec3(p_clip) + v_clip / ma_unit;
	else
		return q;
}

#define NORMAL_DISTANCE 0.1f
#define PLANE_DISTANCE 5.0f
bool plane_distance_disocclusion_check(vec3 current_pos, vec3 history_pos, vec3 current_normal)
{
    vec3  to_current    = current_pos - history_pos;
    float dist_to_plane = abs(dot(to_current, current_normal));

    return dist_to_plane > PLANE_DISTANCE;
}

// ------------------------------------------------------------------------
bool normals_disocclusion_check(vec3 current_normal, vec3 history_normal)
{
    if (pow(abs(dot(current_normal, history_normal)), 2) > NORMAL_DISTANCE)
        return false;
    else
        return true;
}

bool out_of_frame_disocclusion_check(vec2 coord, vec2 image_dim)
{
    // check whether reprojected pixel is inside of the screen
    if (any(lessThan(coord, ivec2(0, 0))) || any(greaterThan(coord, image_dim - ivec2(1, 1))))
        return true;
    else
        return false;
}

bool is_reprojection_valid(vec2 coord, vec3 current_pos, vec3 history_pos, vec3 current_normal, vec3 history_normal, float current_mesh_id, float history_mesh_id, vec2 image_dim)
{
    // check if the history sample is within the frame
    if (out_of_frame_disocclusion_check(coord, image_dim)) return false;

    // check if the history belongs to the same surface
    //if (mesh_id_disocclusion_check(current_mesh_id, history_mesh_id)) return false;

    // check if history sample is on the same plane
    if (plane_distance_disocclusion_check(current_pos, history_pos, current_normal)) return false;

    // check normals for compatibility
    if (normals_disocclusion_check(current_normal, history_normal)) return false;

    return true;
}



void main()
{
    vec2 fragCoord = texCoord*wh;
    uint r = uint(uint(fragCoord.x) * uint(1973) + uint(fragCoord.y) * uint(9277) + uint(time) * uint(26699)) | uint(1);

    vec3 InitSampleLO = texture2D(color, texCoord).xyz;
    vec4 pst = texture2D(secondpos, texCoord).xyzw;

    vec3 cameraOffset = viewPos - lastViewPos;
    vec3 View = texture2D(position, texCoord).xyz;
    vec4 Projected = vec4(View.xyz, 1.);// + vec4(cameraOffset, 0.);
    Projected = prevview * Projected;
    Projected = prevproj * Projected;
    Projected /= Projected.w;
    vec2 ProjectedCoordinates = Projected.xy * 0.5 + 0.5;

            float roughness = texture2D(color, texCoord).w;
bool isthesky = false;

    bool outsc = false;
 vec4 currN = texture2D(normal, texCoord);
    vec4 prevN = texture2D(prevN, ProjectedCoordinates);
    bool disc = false;
    if (dot(normalize(currN.xyz),normalize(prevN.xyz)) < 0.) {
        disc = true;
    }
    if (abs(currN.w - prevN.w) > 2.)
    {
        disc = true;
    }
    float isSpec = texture2D(reflAlb, texCoord).w;
	    if(ProjectedCoordinates.x > 1. || ProjectedCoordinates.x < 0. || ProjectedCoordinates.y > 1. || ProjectedCoordinates.y < 0. || disc    ){
			ProjectedCoordinates = texCoord;
			outsc = true;
		}	

/*
if(roughness < 0.5){
    /*
 View = texture2D(secondpos , texCoord).xyz;
   Projected = vec4(View.xyz, 1.);// + vec4(cameraOffset, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
  vec2 ProjectedCoordinates2 = Projected.xy * 0.5 + 0.5;

    vec2 iResolution = wh;
    vec3 currSecondPos = texture2D(position, texCoord).xyz;
    float bestDist = 9999.;
    vec2 bestPos = texCoord;
    for(int i = 0; i < 29; i++){
        float radius = (30.0)*rndf(r);
        float angle = rndf(r)*2.0*3.14159;
        float x =  (cos(angle) * radius);
        float y =  (sin(angle) * radius);
        vec2 offset = vec2(x,y);
        vec2 currCoords = (texCoord*iResolution + offset)/iResolution;
        vec3 nextSP = texture2D(prevPosition, currCoords).xyz;

        if(length(currSecondPos - nextSP) < bestDist){
            bestDist = length(currSecondPos- nextSP);
            bestPos = currCoords;
        }
    }
vec2 centerPos = bestPos;
     ProjectedCoordinates = centerPos;
    //if(texture2D(secondpos, texCoord).w < 0.5){
        //ProjectedCoordinates = bestPos;
    //}
    
    View = texture2D(reflAlb , texCoord).xyz;
   Projected = vec4(View.xyz, 1.);// + vec4(cameraOffset, 0.);
  Projected = prevview * Projected;
  Projected = prevproj * Projected;
  Projected /= Projected.w;
   ProjectedCoordinates = Projected.xy * 0.5 + 0.5;


ProjectedCoordinates = motionVectorSpecular(texture2D(secondpos, texCoord).xyz, texCoord-ProjectedCoordinates , texCoord, texture2D(prevSecondPosition, texCoord).xyz);


if(texture2D(secondpos, texCoord).w > 0.5 || true){

    //  vec3 LO = InitSampleLO;
    //  weigths = vec4(1., 1., 1., 1.0);

	//  outrad = vec4(clamp(LO,0.,100.) , 1.0);
    //  return;
     isthesky = true;
    //ProjectedCoordinates = texCoord;
}
if(texCoord.x > 0.5){
//  vec3 LO = InitSampleLO;
//      weigths = vec4(1., 1., 1., 1.0);

// 	 outrad = vec4(clamp(LO,0.,100.) , 1.0);
//      return;
}

}*/
    /*
    prevW;
uniform sampler2D prevL;
uniform sampler2D prevP;
    */
    vec3 weigthR = texture2D(prevW, ProjectedCoordinates).xyz;
    float RM = weigthR.x;
    float Rw = weigthR.y;
    float RW = weigthR.z;
    vec3 LO = texture2D(prevL, ProjectedCoordinates).xyz;


    if(outsc  ){
        RM = 0.;
        RW = 0.;
        Rw = 0.;
        LO = InitSampleLO;
    }

  //  vec3 test = LO*0.9 + InitSampleLO*0.1;
    vec4 _pst = texture2D(prevP, ProjectedCoordinates);
    vec3 pporigin = texture2D(position, texCoord).xyz;
    float cosT = max(dot(normalize(_pst.xyz - pporigin), currN.xyz),0.);

    float w = lum(InitSampleLO);  
    vec3 prevCol = LO;
    /*if(isthesky){
        RM = 0.;
        Rw = 0.;
        RW = 0.;
       // LO = InitSampleLO;
    }*/



    
            Rw /= max(RM,0.001);    
   	    	RM = min(RM, 20.); 
   		    Rw *= max(RM,0.001);
   
    Rw = Rw + w;
	RM = RM + 1.0;
	///bool changed = false;
	if(rndf(r) < w/max(Rw,0.001)){
		LO = InitSampleLO;
		//isK = globalIllum.w;
        _pst = pst;
        //_nst = nst;
		//changed = true;
	}
            //R.Update(S, w, rndf(r));  
     
     if(RM * lum(LO) < 0.001){
         RW = 0.;
    }else{
        RW = Rw / max(RM * lum(LO), 0.001);
    }
   


   
    weigths = vec4(RM, Rw, RW, 1.0);
	outrad = vec4(clamp(LO,0.,1000.) , 1.0);
    outpos = _pst;








    //weigths;
    //outrad;
    //outpos;
}