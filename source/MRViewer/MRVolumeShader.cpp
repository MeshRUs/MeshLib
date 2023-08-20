#include "MRVolumeShader.h"
#include "MRGladGlfw.h"

namespace MR
{

std::string getTrivialVertexShader()
{
    return MR_GLSL_VERSION_LINE R"(
  precision highp float;
  precision highp int;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform sampler3D volume;
  uniform vec3 voxelSize;
  uniform vec3 minCorner;
  in vec3 position;

  void main()
  {
    vec3 dims = vec3( textureSize( volume, 0 ) );
    gl_Position = proj * view * model * vec4( voxelSize * dims * position + voxelSize * minCorner, 1.0 );
  }
)";
}

std::string getVolumeFragmentShader()
{
    return MR_GLSL_VERSION_LINE R"(
  precision highp float;
  precision highp int;

  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 normal_matrix;

  uniform int shadingMode; // 0-none,1-dense grad,2-alpha grad
  uniform float specExp;   // (in from base) lighting parameter
  uniform vec3 ligthPosEye;   // (in from base) light position transformed by view only (not proj)                            
  uniform float ambientStrength;    // (in from base) non-directional lighting
  uniform float specularStrength;   // (in from base) reflection intensity

  uniform vec4 viewport;
  uniform vec3 voxelSize;
  uniform vec3 minCorner;
  uniform float step;

  uniform sampler3D volume;
  uniform sampler2D denseMap;

  uniform bool useClippingPlane;     // (in from base) clip primitive by plane if true
  uniform vec4 clippingPlane;        // (in from base) clipping plane

  out vec4 outColor;

  vec3 normalEye( in vec3 textCoord, in vec3 dimStepVoxel, in bool alphaGrad )
  {
    float minXVal = texture( volume, textCoord - vec3(dimStepVoxel.x,0,0)).r;
    float maxXVal = texture( volume, textCoord + vec3(dimStepVoxel.x,0,0)).r;
    float minYVal = texture( volume, textCoord - vec3(0,dimStepVoxel.y,0)).r;
    float maxYVal = texture( volume, textCoord + vec3(0,dimStepVoxel.y,0)).r;
    float minZVal = texture( volume, textCoord - vec3(0,0,dimStepVoxel.z)).r;
    float maxZVal = texture( volume, textCoord + vec3(0,0,dimStepVoxel.z)).r;
    if ( alphaGrad )
    {
        minXVal = texture( denseMap, vec2( minXVal, 0.5 ) ).a;
        maxXVal = texture( denseMap, vec2( maxXVal, 0.5 ) ).a;
        minYVal = texture( denseMap, vec2( minYVal, 0.5 ) ).a;
        maxYVal = texture( denseMap, vec2( maxYVal, 0.5 ) ).a;
        minZVal = texture( denseMap, vec2( minZVal, 0.5 ) ).a;
        maxZVal = texture( denseMap, vec2( maxZVal, 0.5 ) ).a;        
    }
    vec3 grad = -vec3( maxXVal - minXVal, maxYVal - minYVal, maxZVal - minZVal );
    if ( dot( grad, grad ) < 1.0e-12 )
        return vec3(0,0,0);
    grad = vec3( normal_matrix *vec4( normalize(grad),0.0 ) );
    grad = normalize( grad );
    return grad;
  }

  void shadeColor( in vec3 positionEye, in vec3 normalEye, inout vec4 color )
  {
    if ( dot( normalEye,normalEye ) == 0.0 )
      return;
    vec3 normEyeCpy = normalEye;
    vec3 direction_to_light_eye = normalize (ligthPosEye - positionEye);

    float dot_prod = dot (direction_to_light_eye, normEyeCpy);
    if (dot_prod < 0.0)
    {
      //dot_prod = 0.0;//-dot_prod;
      dot_prod = -dot_prod;
      normEyeCpy = -normEyeCpy;
    }
    vec3 reflection_eye = reflect (-direction_to_light_eye, normEyeCpy);
    vec3 surface_to_viewer_eye = normalize (-positionEye);
    float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
    if ( dot_prod_specular < 0.0 )
      dot_prod_specular = 0.0;
    float specular_factor = pow (dot_prod_specular, specExp);

    vec3 ligthColor = vec3(1.0,1.0,1.0);

    vec3 ambient = ambientStrength * ligthColor;
    vec3 diffuse = dot_prod * ligthColor;
    vec3 specular = specular_factor * specularStrength * ligthColor;
    
    color.rgb = ( ambient + diffuse + specular ) * color.rgb;
  }

  void main()
  {
    mat4 eyeM = view * model;
    mat4 fullM = proj * eyeM;
    mat4 inverseFullM = inverse( fullM );

    vec3 clipNear = vec3(0.0,0.0,0.0);
    clipNear.xy =  (2.0 * (gl_FragCoord.xy - viewport.xy)) / (viewport.zw) - vec2(1.0,1.0);
    clipNear.z = (2.0 * gl_FragCoord.z - gl_DepthRange.near - gl_DepthRange.far) / gl_DepthRange.diff;
    vec3 clipFar = clipNear;
    clipFar.z = 1.0;

    if ( clipNear.x < -1.0 || clipNear.x > 1.0 || clipNear.y < -1.0 || clipNear.y > 1.0 )
        discard;

    vec4 rayStartW = inverseFullM * vec4( clipNear, 1.0 );
    vec4 rayEndW = inverseFullM * vec4( clipFar, 1.0 );

    vec3 rayStart = vec3( rayStartW.xyz ) / rayStartW.w;
    vec3 rayEnd = vec3( rayEndW.xyz ) / rayEndW.w;;
    vec3 normRayDir = normalize( rayEnd - rayStart );

    vec3 dims = vec3( textureSize( volume, 0 ) );
    vec3 onesVec = vec3(1,1,1);
    if (voxelSize.x > voxelSize.y && voxelSize.x > voxelSize.y)
        onesVec.yz = vec2(voxelSize.x/voxelSize.y,voxelSize.x/voxelSize.z);
    else if (voxelSize.y > voxelSize.z)
        onesVec.xz = vec2(voxelSize.y/voxelSize.x,voxelSize.y/voxelSize.z);
    else
        onesVec.xy = vec2(voxelSize.z/voxelSize.x,voxelSize.z/voxelSize.y);
    vec3 dimStepVoxel = onesVec/dims;
    vec3 minPoint = voxelSize * minCorner;
    vec3 maxPoint = minPoint + vec3( dims.x * voxelSize.x, dims.y * voxelSize.y, dims.z * voxelSize.z );
    
    bool firstFound = false;
    vec3 firstOpaque = vec3(0.0,0.0,0.0);

    vec3 textCoord = vec3(0.0,0.0,0.0);
    outColor = vec4(0.0,0.0,0.0,0.0);
    vec3 rayStep = step * normRayDir;
    rayStart = rayStart - rayStep * 0.5;
    vec3 diagonal = maxPoint - minPoint;
    while ( outColor.a < 1.0 )
    {
        rayStart = rayStart + rayStep;

        textCoord = ( rayStart - minPoint ) / diagonal;
        if ( any( lessThan( textCoord, vec3(0.0,0.0,0.0) ) ) || any( greaterThan( textCoord, vec3(1.0,1.0,1.0) ) ) )
            break;
        
        if (useClippingPlane && dot( vec3( model*vec4(rayStart,1.0)),vec3(clippingPlane))>clippingPlane.w)
            continue;

        float density = texture( volume, textCoord ).r;        
        vec4 color = texture( denseMap, vec2( density, 0.5 ) );

        float alpha = outColor.a + color.a * ( 1.0 - outColor.a );
        if ( alpha == 0.0 )
            continue;
        
        if ( shadingMode != 0 )
        {
            vec3 normEye = normalEye(textCoord, dimStepVoxel, shadingMode == 2 );
            if ( dot(normEye,normEye) == 0 )
                continue;

            shadeColor( vec3( eyeM * vec4( rayStart, 1.0 ) ), normEye, color );
        }

        outColor.rgb = mix( color.a * color.rgb, outColor.rgb, outColor.a ) / alpha;
        outColor.a = alpha;

        if ( outColor.a > 0.98 )
            outColor.a = 1.0;

        if ( !firstFound )
        {
            firstFound = true;
            firstOpaque = rayStart;
        }
    }

    if ( outColor.a == 0.0 )
        discard;

    vec4 projCoord = fullM * vec4( firstOpaque, 1.0 );
    gl_FragDepth = projCoord.z / projCoord.w * 0.5 + 0.5;
  }
)";
}

std::string getVolumePickerFragmentShader()
{
    return MR_GLSL_VERSION_LINE R"(
  precision highp float;
  precision highp int;

  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;

  uniform vec4 viewport;
  uniform vec3 voxelSize;
  uniform vec3 minCorner;

  uniform int shadingMode; // 0-none,1-dense grad,2-alpha grad

  uniform sampler3D volume;
  uniform sampler2D denseMap;

  uniform bool useClippingPlane;     // (in from base) clip primitive by plane if true
  uniform vec4 clippingPlane;        // (in from base) clipping plane

  uniform uint uniGeomId;
  out highp uvec4 outColor;

  bool normalEye( in vec3 textCoord, in vec3 dimStepVoxel, in bool alphaGrad )
  {
    float minXVal = texture( volume, textCoord - vec3(dimStepVoxel.x,0,0)).r;
    float maxXVal = texture( volume, textCoord + vec3(dimStepVoxel.x,0,0)).r;
    float minYVal = texture( volume, textCoord - vec3(0,dimStepVoxel.y,0)).r;
    float maxYVal = texture( volume, textCoord + vec3(0,dimStepVoxel.y,0)).r;
    float minZVal = texture( volume, textCoord - vec3(0,0,dimStepVoxel.z)).r;
    float maxZVal = texture( volume, textCoord + vec3(0,0,dimStepVoxel.z)).r;
    if ( alphaGrad )
    {
        minXVal = texture( denseMap, vec2( minXVal, 0.5 ) ).a;
        maxXVal = texture( denseMap, vec2( maxXVal, 0.5 ) ).a;
        minYVal = texture( denseMap, vec2( minYVal, 0.5 ) ).a;
        maxYVal = texture( denseMap, vec2( maxYVal, 0.5 ) ).a;
        minZVal = texture( denseMap, vec2( minZVal, 0.5 ) ).a;
        maxZVal = texture( denseMap, vec2( maxZVal, 0.5 ) ).a;        
    }
    vec3 grad = -vec3( maxXVal - minXVal, maxYVal - minYVal, maxZVal - minZVal );
    return dot( grad, grad ) >= 1.0e-12;
  }


  void main()
  {
    mat4 fullM = proj * view * model;
    mat4 inverseFullM = inverse( fullM );

    vec3 clipNear = vec3(0.0,0.0,0.0);
    clipNear.xy =  (2.0 * (gl_FragCoord.xy - viewport.xy)) / (viewport.zw) - vec2(1.0,1.0);
    clipNear.z = (2.0 * gl_FragCoord.z - gl_DepthRange.near - gl_DepthRange.far) / gl_DepthRange.diff;
    vec3 clipFar = clipNear;
    clipFar.z = 1.0;

    vec4 rayStartW = inverseFullM * vec4( clipNear, 1.0 );
    vec4 rayEndW = inverseFullM * vec4( clipFar, 1.0 );

    vec3 rayStart = vec3( rayStartW.xyz ) / rayStartW.w;
    vec3 rayEnd = vec3( rayEndW.xyz ) / rayEndW.w;;
    vec3 normRayDir = normalize( rayEnd - rayStart );

    vec3 dims = vec3( textureSize( volume, 0 ) );
    vec3 onesVec = vec3(1,1,1);
    if (voxelSize.x > voxelSize.y && voxelSize.x > voxelSize.y)
        onesVec.yz = vec2(voxelSize.x/voxelSize.y,voxelSize.x/voxelSize.z);
    else if (voxelSize.y > voxelSize.z)
        onesVec.xz = vec2(voxelSize.y/voxelSize.x,voxelSize.y/voxelSize.z);
    else
        onesVec.xy = vec2(voxelSize.z/voxelSize.x,voxelSize.z/voxelSize.y);
    vec3 dimStepVoxel = onesVec/dims;

    vec3 minPoint = voxelSize * minCorner;
    vec3 maxPoint = minPoint + vec3( dims.x * voxelSize.x, dims.y * voxelSize.y, dims.z * voxelSize.z );
    
    bool firstFound = false;
    vec3 textCoord = vec3(0.0,0.0,0.0);
    vec3 rayStep = 0.5 * length( voxelSize ) * normRayDir;
    rayStart = rayStart - rayStep * 0.5;
    vec3 diagonal = maxPoint - minPoint;
    while ( !firstFound )
    {
        rayStart = rayStart + rayStep;

        textCoord = ( rayStart - minPoint ) / diagonal;
        if ( any( lessThan( textCoord, vec3(0.0,0.0,0.0) ) ) || any( greaterThan( textCoord, vec3(1.0,1.0,1.0) ) ) )
            break;
        
        if (useClippingPlane && dot( vec3( model*vec4(rayStart,1.0)),vec3(clippingPlane))>clippingPlane.w)
            continue;

        float density = texture( volume, textCoord ).r;
        if ( texture( denseMap, vec2( density, 0.5 ) ).a == 0.0 )
            continue;

        if ( shadingMode != 0 && !normalEye(textCoord, dimStepVoxel, shadingMode == 2) )
            continue;

        firstFound = true;
    }

    if ( !firstFound )
        discard;
    vec4 projCoord = fullM * vec4( rayStart, 1.0 );
    float depth = projCoord.z / projCoord.w * 0.5 + 0.5;
    gl_FragDepth = depth;

    outColor.r = uint(0); // find VoxelId by world pos
    outColor.g = uniGeomId;

    outColor.a = uint(depth * 4294967295.0);
  }
)";
}

}