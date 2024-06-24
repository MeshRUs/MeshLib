#include "MRMeshShader.h"
#include "MRShaderBlocks.h"
#include "MRGladGlfw.h"

namespace MR
{

std::string getMeshVerticesShader()
{
    return MR_GLSL_VERSION_LINE R"(
            precision highp float;
            precision highp int;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 normal_matrix;

  in vec3 position;       // (in from base) vert position
  in vec3 normal;         // (in from base) vert normal
  in vec4 K;              // (in from base) vert color
  in vec2 texcoord;       // (in from base) vert uv coordinate

  out vec2 texcoordi;    // (out to fragment shader) vert uv coordinate
  out vec3 world_pos;    // (out to fragment shader) vert transformed position
  out vec4 Ki;           // (out to fragment shader) vert color 
  out vec3 position_eye; // (out to fragment shader) vert position transformed by model and view (not proj)
  out vec3 normal_eye;   // (out to fragment shader) vert normal transformed by model and view (not proj)
  out float primitiveIdf0;
  out float primitiveIdf1;

  void main()
  {
    world_pos = vec3(model*vec4 (position, 1.0));
    position_eye = vec3 (view * vec4 (world_pos, 1.0));
    normal_eye = vec3 (normal_matrix * vec4 (normal, 0.0));
    normal_eye = normalize(normal_eye);
    gl_Position = proj * vec4 (position_eye, 1.0); //proj * view * vec4(position, 1.0);"
    Ki = K;
    texcoordi = texcoord;
    uint primId = uint(gl_VertexID) / 3u;
    primitiveIdf1 = float( uint( primId >> 20u ) ) + 0.5;
    primitiveIdf0 = float( primId % uint( 1u << 20u ) ) + 0.5;
  }
)";
}

std::string getMeshFragmentShader( bool gl4, bool alphaSort )
{
    return
        getFragmentShaderHeaderBlock( gl4, alphaSort ) +
        getMeshFragmentShaderArgumetsBlock() +
        getShaderMainBeginBlock() +
        getFragmentShaderClippingBlock() +
        getFragmentShaderOnlyOddBlock( gl4 && !alphaSort ) +
        getMeshFragmentShaderColoringBlock() +
        getFragmentShaderEndBlock( alphaSort );
}

std::string getMeshFragmentShaderArgumetsBlock()
{
    return R"(
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 normal_matrix;

  uniform highp usampler2D selection;      // (in from base) selection BitSet
  uniform sampler2D faceNormals;     // (in from base) normals per face
  uniform sampler2D faceColors;      // (in from base) face color
  uniform bool perFaceColoring;      // (in from base) use faces colormap is true
  uniform bool perVertColoring;      // (in from base) linear interpolate colors if true
  uniform bool enableShading;        // (in from base) use normals or not
  uniform bool flatShading;          // (in from base) linear interpolate normals if false
  uniform bool onlyOddFragments;     // (in from base) discard every second fragment
  uniform bool showSelFaces;    // (in from base) use selection or not
 
  uniform vec4 mainColor;            // (in from base) main color
  uniform vec4 selectionColor;       // (in from base) selection color
  uniform vec4 backColor;            // (in from base) back face color
  uniform vec4 selBackColor;   // (in from base) selection back face color
  uniform bool useClippingPlane;     // (in from base) clip primitive by plane if true
  uniform vec4 clippingPlane;        // (in from base) clipping plane
  uniform bool invertNormals;        // (in from base) invert normals if true
  uniform bool mirrored;
  uniform sampler2D tex;             // (in from base) texture
  uniform float specExp;   // (in from base) lighting parameter 
  uniform bool useTexture;           // (in from base) enable texture
  uniform vec3 ligthPosEye;   // (in from base) light position transformed by view only (not proj)
                                     
  uniform float ambientStrength;    // (in from base) non-directional lighting
  uniform float specularStrength;   // (in from base) reflection intensity
  uniform float globalAlpha;        // (in from base) global transparency multiplier
                                     
  in vec3 position_eye;              // (in from vertex shader) vert position transformed by model and view (not proj)
  in vec3 normal_eye;                // (in from vertex shader) vert normal transformed by model and view (not proj)
  in vec4 Ki;                        // (in from vertex shader) vert color
  in vec2 texcoordi;                 // (in from vertex shader) vert uv coordinate
  in vec3 world_pos;                 // (in from vertex shader) vert transformed position
  
  in float primitiveIdf0;
  in float primitiveIdf1;
                                     
  out vec4 outColor;                 // (out to render) fragment color
)";
}

std::string getMeshFragmentShaderColoringBlock()
{
    return 
#ifdef __EMSCRIPTEN__
        R"(
    uint primitiveId = ( uint(primitiveIdf1) << 20u ) + uint(primitiveIdf0);)"
#else
        R"(
    uint primitiveId = uint(gl_PrimitiveID);)"
#endif
        R"(
    vec3 normEyeCpy = normal_eye;
    if ( flatShading )
    {
      ivec2 texSize = textureSize( faceNormals, 0 );
      vec3 norm = vec3( texelFetch( faceNormals, ivec2( primitiveId % uint(texSize.x), primitiveId / uint(texSize.x) ), 0 ) );
      normEyeCpy = normalize(vec3 (normal_matrix * vec4 (norm, 0.0)));
    }
    
    vec3 vector_to_light_eye = ligthPosEye - position_eye;
    vec3 direction_to_light_eye = normalize (vector_to_light_eye);
    float dot_prod = dot (direction_to_light_eye, normalize(normEyeCpy));

    vec4 colorCpy;
    bool selected = false;
    if ( showSelFaces )
    {
      ivec2 texSize = textureSize( selection, 0 );
      uint index = primitiveId / 32u;
      uint block = texelFetch( selection, ivec2( index % uint(texSize.x), index / uint(texSize.x) ), 0 ).r;
      selected = bool( block & uint( 1 << (primitiveId % 32u) ) );
    }

    bool invNorms = invertNormals;
    if ( mirrored )
    {
        dot_prod = -dot_prod;
        invNorms = !invNorms;
    }
    if ( gl_FrontFacing == invNorms )
        if ( !selected )
            colorCpy = backColor;
        else
            colorCpy = selBackColor;
    else
        if ( selected )
            colorCpy = selectionColor;
        else
        if ( perFaceColoring )
        {
            ivec2 texSize = textureSize( faceColors, 0 );
            colorCpy = texelFetch(faceColors, ivec2( primitiveId % uint(texSize.x), primitiveId / uint(texSize.x) ), 0 );
        }
        else
        if ( perVertColoring )
            colorCpy = Ki;
        else
            colorCpy = mainColor;

    if ( useTexture && !selected )
    {
      vec4 textColor = texture(tex, texcoordi);
      float destA = colorCpy.a;
      colorCpy.a = textColor.a + destA * ( 1.0 - textColor.a );
      if ( colorCpy.a == 0.0 )
        colorCpy.rgb = vec3(0);
      else
        colorCpy.rgb = mix(colorCpy.rgb*destA,textColor.rgb,textColor.a)/colorCpy.a;
    }  

    if (gl_FrontFacing == false) // don't use !gl_FrontFacing for some rare mac issue
      dot_prod = -dot_prod;

    if (dot_prod < 0.0)
      dot_prod = 0.0;

    vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normEyeCpy));
    vec3 surface_to_viewer_eye = normalize (-position_eye);
    float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
    if (dot_prod_specular < 0.0)
      dot_prod_specular = 0.0;

    float specular_factor = pow (dot_prod_specular, specExp);

    vec3 ligthColor = vec3(1.0,1.0,1.0);
    vec3 color = vec3(colorCpy);

    vec3 ambient = ambientStrength * ligthColor;
    vec3 diffuse = dot_prod * ligthColor;
    vec3 specular = specular_factor * specularStrength * ligthColor;
    
    if ( enableShading )
      color = ( ambient + diffuse + specular ) * color;

    outColor = vec4(color,colorCpy.a * globalAlpha);

    if (outColor.a == 0.0)
      discard;
)";
}

}
