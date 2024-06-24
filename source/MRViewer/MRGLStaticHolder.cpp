#include "MRGLStaticHolder.h"
#include "MRCreateShader.h"
#include "MRGLMacro.h"
#include "MRGladGlfw.h"
#include "MRVolumeShader.h"
#include "MRMeshShader.h"
#include "MRLinesShader.h"
#include "MRShaderBlocks.h"

namespace
{
std::string getShaderName( MR::GLStaticHolder::ShaderType type )
{
    const std::array<std::string, size_t( MR::GLStaticHolder::Count )> names =
    {
        "Mesh shader",
        "Picker shader",
        "Mesh desktop picker shader",
        "Alpha-sort mesh shader",

        "Points shader",

        "Lines shader",
        "Lines joint shader",
        "Lines picker shader",
        "Lines joint picker shader",

        "Alpha-sort lines shader",

        "Labels shader",

        "Viewport lines shader",
        "Viewport points shader",
        "Viewport points shader (no offset)",
        "Immediate tri shader",
        "Viewport border shader",
        "Alpha-sort overlay shader",
        "Shadow overlay shader",
        "Simple overlay shader",

        "Volume shader",
        "Volume picker shader"
    };
    return names[type];
}
}

namespace MR
{

unsigned GLStaticHolder::getShaderId( ShaderType type )
{
    auto& instance = GLStaticHolder::instance_();
    auto& id = instance.shadersIds_[type];
    if ( id == 0 )
    {
        instance.createShader_( type );
    }

    return id;
}

void GLStaticHolder::freeShader( ShaderType type )
{
    auto& instance = GLStaticHolder::instance_();
    if ( instance.shadersIds_[type] == 0 )
        return;

    destroyShader( instance.shadersIds_[type] );
    instance.shadersIds_[type] = 0;
}

void GLStaticHolder::freeAllShaders()
{
    for ( int i = 0; i < ShaderType::Count; ++i )
        freeShader( ShaderType( i ) );
}

GLStaticHolder::GLStaticHolder()
{
    logger_ = Logger::instance().getSpdLogger();
    for ( int i = 0; i < ShaderType::Count; ++i )
        shadersIds_[i] = 0;
}

GLStaticHolder::~GLStaticHolder()
{
    for ( int i = 0; i < ShaderType::Count; ++i )
        if ( shadersIds_[i] != 0 )
            logger_->warn( "{} is not freed", getShaderName( ShaderType( i ) ) );
}

GLStaticHolder& GLStaticHolder::instance_()
{
    static GLStaticHolder instance;
    return instance;
}

void GLStaticHolder::createShader_( ShaderType type )
{
    std::string vertexShader;
    std::string fragmentShader;
    if ( type == Volume || type == VolumePicker )
    {
        vertexShader = getTrivialVertexShader();
        if ( type == Volume )
            fragmentShader = getVolumeFragmentShader();
        else
            fragmentShader = getVolumePickerFragmentShader();
    }
    else if ( type == Mesh || type == TransparentMesh )
    {
        vertexShader = getMeshVerticesShader();

        int major, minor;
        auto window = glfwGetCurrentContext();
        major = glfwGetWindowAttrib( window, GLFW_CONTEXT_VERSION_MAJOR );
        minor = glfwGetWindowAttrib( window, GLFW_CONTEXT_VERSION_MINOR );

        bool gl4 = major >= 4 && ( major != 4 || minor >= 3 );
        bool alphaSort = type == TransparentMesh;
        
        fragmentShader = getMeshFragmentShader( gl4, alphaSort );
    }
    else if ( type == Lines || type == LinesJoint || type == TransparentLines )
    {
        if ( type == Lines || type == TransparentLines )
        {
            vertexShader = getLinesVertexShader();
            fragmentShader = getLinesFragmentShader( type == TransparentLines );
        }
        else
        {
            vertexShader = getLinesJointVertexShader();
            fragmentShader = getLinesJointFragmentShader();
        }
    }
    else if ( type == LinesPicker || type == LinesJointPicker )
    {
        if ( type == LinesPicker )
            vertexShader = getLinesPickerVertexShader();
        else
            vertexShader = getLinesJointPickerVertexShader();
        fragmentShader = getPickerFragmentShader( type == LinesJointPicker );
    }
    else if ( type == Picker || type == MeshDesktopPicker )
    {
        vertexShader =
            MR_GLSL_VERSION_LINE R"(
            precision highp float;
            precision highp int;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform uint primBucketSize;
  uniform float pointSize;

  in vec3 position;
  
  out vec3 world_pos;
  out float primitiveIdf0;
  out float primitiveIdf1;

  void main()
  {
    world_pos = vec3(model*vec4 (position, 1.0));
    gl_Position = proj * view * vec4 (world_pos, 1.0); //proj * view * vec4(position, 1.0);"
    uint primId = uint(gl_VertexID) / primBucketSize;
    primitiveIdf1 = float( uint( primId >> 20u ) ) + 0.5;
    primitiveIdf0 = float( primId % uint( 1u << 20u ) ) + 0.5;
    gl_PointSize = pointSize;
  }
)";

        fragmentShader = getPickerFragmentShader( false, type == Picker );
    }
    else if ( type == DrawPoints )
    {
        vertexShader =
            MR_GLSL_VERSION_LINE R"(
            precision highp float;
            precision highp int;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 normal_matrix;
  uniform float pointSize;
  uniform uint primBucketSize;

  in vec3 position;       // (in from base) vert position
  in vec3 normal;         // (in from base) vert normal
  in vec4 K;              // (in from base) vert color

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
    gl_PointSize = pointSize;
    uint primId = uint(gl_VertexID) / primBucketSize;
    primitiveIdf1 = float( uint( primId >> 20u ) ) + 0.5;
    primitiveIdf0 = float( primId % uint( 1u << 20u ) ) + 0.5;
  }
)";
        fragmentShader =
            MR_GLSL_VERSION_LINE R"(
                precision highp float;
            precision highp int;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 normal_matrix;

  uniform highp usampler2D selection;      // (in from base) selection BitSet
  uniform bool showSelVerts;    // (in from base) use selection or not
  uniform vec4 selectionColor;       // (in from base) selection color
  uniform vec4 selBackColor;   // (in from base) selection back face color

  uniform bool perVertColoring;      // (in from base) linear interpolate colors if true
  uniform bool hasNormals;           // (in from base) dont use normals if they are not
 
  uniform vec4 mainColor;            // (in from base) color if colormap is off
  uniform vec4 backColor;            // (in from base) back face color
  uniform bool useClippingPlane;     // (in from base) clip primitive by plane if true
  uniform vec4 clippingPlane;        // (in from base) clipping plane
  uniform bool invertNormals;        // (in from base) invert normals if true

  uniform float specExp;   // (in from base) lighting parameter
  uniform vec3 ligthPosEye;   // (in from base) light position transformed by view only (not proj)
                                     
  uniform float ambientStrength;    // (in from base) non-directional lighting
  uniform float specularStrength;   // (in from base) reflection intensity
  uniform float globalAlpha;        // (in from base) global transparency multiplier
                                     
  in vec3 position_eye;              // (in from vertex shader) vert position transformed by model and view (not proj)
  in vec3 normal_eye;                // (in from vertex shader) vert normal transformed by model and view (not proj)
  in vec4 Ki;                        // (in from vertex shader) vert color
  in vec3 world_pos;                 // (in from vertex shader) vert transformed position

  in float primitiveIdf0;
  in float primitiveIdf1;
                                     
  out vec4 outColor;                 // (out to render) fragment color

  void main()
  {
    if (length(gl_PointCoord - vec2(0.5)) > 0.5)
      discard;
    if (useClippingPlane && dot(world_pos,vec3(clippingPlane))>clippingPlane.w)
      discard;

    vec3 normEyeCpy = normal_eye;
    
    vec3 vector_to_light_eye = ligthPosEye - position_eye;
    vec3 direction_to_light_eye = normalize (vector_to_light_eye);
    if (!hasNormals)
      normEyeCpy = direction_to_light_eye;

    float dot_prod = dot (direction_to_light_eye, normalize(normEyeCpy));
    
    uint primitiveId = ( uint(primitiveIdf1) << 20u ) + uint(primitiveIdf0);
    vec4 colorCpy;
    bool selected = false;
    if ( showSelVerts )
    {
      ivec2 texSize = textureSize( selection, 0 );
      uint index = primitiveId / 32u;
      uint block = texelFetch( selection, ivec2( index % uint(texSize.x), index / uint(texSize.x) ), 0 ).r;
      selected = bool( block & uint( 1 << (primitiveId % 32u) ) );
    }

    bool frontFacing = dot_prod >= 0.0;
    if ( frontFacing == invertNormals )
    {
        if ( !selected )
            colorCpy = backColor;
        else
            colorCpy = selBackColor;
    }
    else
        if ( selected )
            colorCpy = selectionColor;
        else
        if ( perVertColoring )
            colorCpy = Ki;
        else
            colorCpy = mainColor;

    if (!frontFacing)
      dot_prod = -dot_prod;
    if ( dot_prod < 0.0 )
      dot_prod = 0.0;

    vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normEyeCpy));
    vec3 surface_to_viewer_eye = normalize (-position_eye);
    float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
    if ( dot_prod_specular < 0.0 )
      dot_prod_specular = 0.0;
    float specular_factor = pow (dot_prod_specular, specExp);

    vec3 ligthColor = vec3(1.0,1.0,1.0);
    vec3 color = vec3(colorCpy);

    vec3 ambient = ambientStrength * ligthColor;
    vec3 diffuse = dot_prod * ligthColor;
    vec3 specular = specular_factor * specularStrength * ligthColor;
    
    vec3 res = ( ambient + diffuse + specular ) * color;
    outColor = vec4(res,colorCpy.a * globalAlpha);
    if (outColor.a == 0.0)
      discard;
  }
)";
    }
    else if ( type == Labels )
    {
        vertexShader =
            MR_GLSL_VERSION_LINE R"(
            precision highp float;
            precision highp int;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform vec3 basePos;
  uniform vec2 modifier;
  uniform vec2 shift;

  in vec3 position;       // (in from base) vert position

  void main()
  {
    vec4 projBasePos = proj * ( view * model * vec4( basePos, 1.0 ) );
    vec4 coord = projBasePos + projBasePos.w * vec4( modifier.x * (position.x - shift.x), modifier.y * (position.y - shift.y), 0.0, 0.0);
    gl_Position = coord / coord.w;
  }
)";
        fragmentShader =
            MR_GLSL_VERSION_LINE R"(
            precision highp float;
            precision highp int;

  uniform vec4 mainColor;            // (in from base) main color
  uniform float globalAlpha;        // (in from base) global transparency multiplier
                                     
  out vec4 outColor;                 // (out to render) fragment color

  void main()
  {
    outColor = mainColor;
    outColor.a = outColor.a * globalAlpha;
    if (outColor.a == 0.0)
      discard;
  }
)";
    }
    else
    {
        if ( type == AdditionalLines || type == AdditionalPoints || type == AdditionalPointsNoOffset )
        {
            vertexShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
            precision highp int;
  uniform mat4 view;
  uniform mat4 proj;
  uniform float pointSize;

  in vec3 position;
  in vec4 color;
  out vec4 color_frag;

  void main()
  {
    gl_Position = proj * view * vec4 (position, 1.0);
    color_frag = color;
    gl_PointSize = pointSize;
  }
)";
        }
        else if ( type == AdditionalQuad )
        {
            vertexShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
            precision highp int;
  uniform mat4 model;
  uniform mat4 view;
  uniform mat4 proj;
  uniform mat4 normal_matrix;

  in vec3 position;
  in vec3 normal;         // (in from base) vert normal
  in vec4 color;

  out vec4 color_frag;
  out vec3 position_eye; // (out to fragment shader) vert position transformed by model and view (not proj)
  out vec3 normal_eye;   // (out to fragment shader) vert normal transformed by model and view (not proj)

  void main()
  {
    position_eye = vec3 (view * (model * vec4 (position, 1.0)));
    normal_eye = normalize(vec3 (normal_matrix * vec4 (normal, 0.0)));
    gl_Position = proj * vec4 (position_eye, 1.0); //proj * view * vec4(position, 1.0);"
    color_frag = color;
  }
)";
        }
        else
        {
            vertexShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
            precision highp int;
  uniform vec4 user_color;
  in vec3 position;
  out vec4 color_frag;

  void main()
  {
    gl_Position = vec4 (position, 1.0);
    color_frag = user_color;
  }
)";
        }

        if ( type == AdditionalLines || type == ViewportBorder )
        {
            fragmentShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
            precision highp int;
  uniform float offset;
  in vec4 color_frag;
  out vec4 outColor;
  void main()
  {
    outColor = color_frag;
    if (outColor.a == 0.0)
      discard;
    gl_FragDepth = gl_FragCoord.z + offset;
  }
)";
        }
        else if ( type == AdditionalPoints )
        {
            fragmentShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
            precision highp int;
  uniform float offset;

  in vec4 color_frag;
  out vec4 outColor;
  void main()
  {
    if (length(gl_PointCoord - vec2(0.5)) > 0.5)
      discard;
    outColor = color_frag;
    if (outColor.a == 0.0)
      discard;
    gl_FragDepth = gl_FragCoord.z + offset;
  }
)";
        }
        else if ( type == AdditionalPointsNoOffset )
        {
            fragmentShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
            precision highp int;
  in vec4 color_frag;
  out vec4 outColor;
  void main()
  {
    if (length(gl_PointCoord - vec2(0.5)) > 0.5)
      discard;
    outColor = color_frag;
    if (outColor.a == 0.0)
      discard;
  }
)";
        }
        else if ( type == AdditionalQuad )
        {
            fragmentShader =
                MR_GLSL_VERSION_LINE R"(
                precision highp float;
            precision highp int;
  uniform mat4 view;
  uniform mat4 proj;
  uniform vec3 ligthPosEye;   // (in from base) light position transformed by view only (not proj)
                                     
  float specExp = 35.0f;
  float ambientStrength = 0.1;
  float specularStrength = 0.5;
                                     
  in vec3 position_eye;              // (in from vertex shader) vert position transformed by model and view (not proj)
  in vec3 normal_eye;                // (in from vertex shader) vert normal transformed by model and view (not proj)
  in vec4 color_frag;                        // (in from vertex shader) vert color
                                     
  out vec4 outColor;                 // (out to render) fragment color

  void main()
  {    
    vec3 vector_to_light_eye = ligthPosEye - position_eye;
    vec3 direction_to_light_eye = normalize (vector_to_light_eye);
    float dot_prod = abs(dot (direction_to_light_eye, normalize(normal_eye)));

    vec3 reflection_eye = reflect (-direction_to_light_eye, normalize(normal_eye));
    vec3 surface_to_viewer_eye = normalize (-position_eye);
    float dot_prod_specular = dot (reflection_eye, surface_to_viewer_eye);
    if ( dot_prod_specular < 0.0 )
      dot_prod_specular = 0.0;
    float specular_factor = pow (dot_prod_specular, specExp);

    vec3 ligthColor = vec3(1.0,1.0,1.0);
    vec3 color = vec3(color_frag);

    vec3 ambient = ambientStrength * ligthColor;
    vec3 diffuse = dot_prod * ligthColor;
    vec3 specular = specular_factor * specularStrength * ligthColor;
    
    vec3 res = ( ambient + diffuse + specular ) * color;
    outColor = vec4(res,color_frag.a);

    if (outColor.a == 0.0)
      discard;
  }
)";
        }
        else if ( type == TransparencyOverlayQuad )
        {
            fragmentShader =
                R"(
#version 430 core

#define MAX_FRAGMENTS 18

// made as macro to have it inline
#define swapNodesInd(a, b) \
{ \
	uint tmp = a; \
	a = b; \
	b = tmp; \
}

layout (early_fragment_tests) in;

out vec4 color;

struct Node 
{
    vec4    color;
    float   depth;
    uint    next;
};

layout (binding = 0, r32ui)      uniform uimage2D    heads;
layout (binding = 0, offset = 0) uniform atomic_uint numNodes;

layout (binding = 0, std430 ) buffer Lists
{
    Node nodes [];
};

void sortedInsert(inout uint fragsInds[MAX_FRAGMENTS], uint nodeInd, inout int count) 
{
	if (count == MAX_FRAGMENTS) 
	{
		if (nodes[nodeInd].depth < nodes[fragsInds[count - 1]].depth) 
		{
			swapNodesInd(nodeInd, fragsInds[count - 1]);
		}
	}
	else 
	{
		fragsInds[count] = nodeInd;
		count++;
	}

	uint i = count - 1;
	while (i > 0 && nodes[fragsInds[i]].depth < nodes[fragsInds[i - 1]].depth)
	{
		swapNodesInd(fragsInds[i], fragsInds[i - 1]);
		i--;
	}
}

// separate function to work-around weird behavior on AMD video cards
vec4 blendColor(uint fragsInds[MAX_FRAGMENTS], int count)
{
    vec4 color = vec4(0, 0, 0, 0);
    
    for ( int i = count - 1; i >= 0; i-- )
    {
        float destA = color.a;
        vec4 fragColor = nodes[fragsInds[i]].color;
        color.a = fragColor.a + destA * ( 1.0 - fragColor.a );
        if ( color.a == 0.0 )
          color.rgb = vec3(0);
        else
          color.rgb = mix(color.rgb*destA,fragColor.rgb,fragColor.a)/color.a;
    }

    return color;
}

void main(void)
{
    uint nodesCounter = atomicCounter( numNodes );
    if (nodesCounter == 0)
      discard;

    uint fragsInds [MAX_FRAGMENTS];
    // suppress 'used uninitialized' warning; init values are not used
    fragsInds[0] = 0;
    int count = 0;
    // get the index of the head of the list
    uint n = imageLoad ( heads, ivec2 ( gl_FragCoord.xy ) ).r;

    // sort linked list to array
    while ( n != 0xFFFFFFFF )
    {
        sortedInsert(fragsInds, n, count);
        n = nodes[n].next;
    }
    
    if ( count == 0 )
        discard;
    
    // traverse the array, and combine the colors using the alpha channel    
    color = blendColor(fragsInds, count);
    
    gl_FragDepth = nodes[fragsInds[0]].depth;
}
)";
        }
        else if ( type == ShadowOverlayQuad )
        {
        fragmentShader =
            MR_GLSL_VERSION_LINE R"(
                precision highp float;
            precision highp int;
  uniform sampler2D pixels;
  uniform vec4 color;
  uniform vec2 shift;
  uniform float blurRadius;
  uniform bool convX;
  out vec4 outColor;                 // (out to render) fragment color

  const float gaussWeights[] = float[7] (0.161046, 0.148645, 0.116919, 0.078381, 0.044771, 0.021742, 0.009019);
  
  float getValue( vec2 pos )
  {
    if ( pos.x < 0.0 || pos.x > 1.0 || pos.y < 0.0 || pos.y > 1.0 )
      return 0.0;    
    return texture(pixels,pos).a;
  }
  void main()
  { 
    gl_FragDepth = 0.9999;
    ivec2 texSize = textureSize( pixels, 0 );
    vec2 pos = gl_FragCoord.xy;
    if ( !convX )
      pos = pos - shift;
    pos = vec2( pos.x/float(texSize.x),pos.y/float(texSize.y) );
    vec2 posShift = vec2(0.0);
    if ( convX )
      posShift = vec2(blurRadius /(6.0* float(texSize.x)),0.0);
    else
      posShift = vec2(0.0,blurRadius /(6.0* float(texSize.y)));
    
    float convSum = gaussWeights[0]*getValue(pos);
    for ( int i=1; i<=6; ++i )
    {
      vec2 fullShift = float(i)*posShift;
      convSum = convSum + gaussWeights[i]*getValue(pos+fullShift);
      convSum = convSum + gaussWeights[i]*getValue(pos-fullShift);
    }
    outColor = vec4(color.rgb,convSum);
    if ( !convX )
      outColor.a = outColor.a*color.a;
    if (outColor.a == 0.0)
      discard;
  }
)";
        }
        else if ( type == SimpleOverlayQuad )
        {
        fragmentShader =
            MR_GLSL_VERSION_LINE R"(
                precision highp float;
            precision highp int;
  uniform sampler2D pixels;
  uniform vec2 viewportSize;
  uniform float depth;
  out vec4 outColor;                 // (out to render) fragment color

  void main()
  { 
    gl_FragDepth = depth;
    vec2 pos = gl_FragCoord.xy;
    pos = vec2( pos.x/float(viewportSize.x),pos.y/float(viewportSize.y) );
    outColor = texture(pixels, pos );
    if (outColor.a == 0.0)
      discard;
  }
)";
        }
    }

    DisabledWarnings warns = {};
    if ( type == TransparencyOverlayQuad )
        warns.push_back( { 7050,"used uninitialized" } );

    createShader( getShaderName( type ), vertexShader, fragmentShader, shadersIds_[type], warns );
}

RenderObjectBuffer &GLStaticHolder::getStaticGLBuffer()
{
    return instance_().glBuffer_;
}

}
