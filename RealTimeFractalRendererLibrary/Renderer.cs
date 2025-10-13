
using Common;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using OpenTK.Windowing.GraphicsLibraryFramework;
using StbImageSharp;
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace RealTimeFractalRendererLibrary
{
    public class Renderer
    {
      
      


        #region Fields

        // Buffers and handles
        private int fullscreenQuadVbo, fullscreenQuadVao, fullscreenQuadEbo;
        private int gBufferFramebuffer, mainFramebuffer, radianceCacheSsbo;
        private int temporalFramebuffer, spatialFramebuffer, previousFrameFramebuffer, previousTemporalFramebuffer, tempAccumulationFramebuffer, denoiseFramebufferA, denoiseFramebufferB, spatialPrepassFramebuffer;
        private int skyFramebuffer, sunFramebuffer, upscaleFramebuffer, taaFramebuffer;
        private int upscale1;
        private int upscale2;
        private int upscale3;

        private Shader ups1 = default!;
        private Shader ups2 = default!;
        private Shader ups3 = default!;


        private readonly TextureBinding upscaleTex1 = new(nameof(upscaleTex1));
        private readonly TextureBinding upscaleTex2 = new(nameof(upscaleTex2));
        private readonly TextureBinding upscaleTex3 = new(nameof(upscaleTex3));

        private List<TextureBinding> Textures2D = [];
        private List<TextureBinding> Textures3D = [];

        // Shaders
        private Shader initializationShader = default!;
        private Shader geometryPassShader = default!;
        private Shader finalCompositeShader = default!;
        private Shader temporalResamplingShader = default!;
        private Shader spatialPrepassShader = default!;
        private Shader spatialResamplingShader = default!;
        private Shader previousFrameCopyShader = default!;
        private Shader previousTemporalCopyShader = default!;
        private Shader tempAccumulationShader = default!;
        private Shader upscaleShader = default!;
        private Shader taaShader = default!;
        private Shader denoiseShader = default!;
        private Shader skyShader = default!;
        private Shader sunShader = default!;
        private Shader radianceCacheShader = default!;

        private Shader worleyNoiseShader = default!;
        private Shader cloudShader = default!;

        // Texture bindings
        private readonly TextureBinding cubemap = new(nameof(cubemap));

        private readonly TextureBinding colorTexture = new(nameof(colorTexture));
        private readonly TextureBinding holdInfoTexture = new(nameof(holdInfoTexture));
        private readonly TextureBinding reflectionTexture = new(nameof(reflectionTexture));
        private readonly TextureBinding positionTexture = new(nameof(positionTexture));
        private readonly TextureBinding normalTexture = new(nameof(normalTexture));
        private readonly TextureBinding albedoTexture = new(nameof(albedoTexture));
        private readonly TextureBinding fogPositionTexture = new(nameof(fogPositionTexture));
        private readonly TextureBinding secondaryPositionTexture = new(nameof(secondaryPositionTexture));
        private readonly TextureBinding secondaryAlbedoTexture = new(nameof(secondaryAlbedoTexture));
        private readonly TextureBinding sssTexture = new(nameof(sssTexture));
        private readonly TextureBinding sssDenoiseTexture = new(nameof(sssDenoiseTexture));
        private readonly TextureBinding waterPositionTexture = new(nameof(waterPositionTexture));
        private readonly TextureBinding waterNormalTexture = new(nameof(waterNormalTexture));
        private readonly TextureBinding fogColorTexture = new(nameof(fogColorTexture));
        private readonly TextureBinding fogColorTextureB = new(nameof(fogColorTextureB));
        private readonly TextureBinding temporalWeightsTexture = new(nameof(temporalWeightsTexture));
        private readonly TextureBinding temporalRadianceTexture = new(nameof(temporalRadianceTexture));
        private readonly TextureBinding temporalPositionTexture = new(nameof(temporalPositionTexture));
        private readonly TextureBinding spatialWeightsTexture = new(nameof(spatialWeightsTexture));
        private readonly TextureBinding spatialRadianceTexture = new(nameof(spatialRadianceTexture));
        private readonly TextureBinding prevTemporalWeightsTexture = new(nameof(prevTemporalWeightsTexture));
        private readonly TextureBinding prevTemporalRadianceTexture = new(nameof(prevTemporalRadianceTexture));
        private readonly TextureBinding prevTemporalPositionTexture = new(nameof(prevTemporalPositionTexture));
        private readonly TextureBinding prevSpatialWeightsTexture = new(nameof(prevSpatialWeightsTexture));
        private readonly TextureBinding prevSpatialRadianceTexture = new(nameof(prevSpatialRadianceTexture));
        private readonly TextureBinding prevNormalDepthTexture = new(nameof(prevNormalDepthTexture));
        private readonly TextureBinding prevAccumulationTexture = new(nameof(prevAccumulationTexture));
        private readonly TextureBinding prevTaaTexture = new(nameof(prevTaaTexture));
        private readonly TextureBinding tempAccumulationTexture = new(nameof(tempAccumulationTexture));
        private readonly TextureBinding upscaleTexture = new(nameof(upscaleTexture));
        private readonly TextureBinding upscaleTextureB = new(nameof(upscaleTextureB));
        private readonly TextureBinding taaTexture = new(nameof(taaTexture));
        private readonly TextureBinding denoiseTextureA = new(nameof(denoiseTextureA));
        private readonly TextureBinding denoiseTextureB = new(nameof(denoiseTextureB));
        private readonly TextureBinding varianceTextureA = new(nameof(varianceTextureA));
        private readonly TextureBinding varianceTextureB = new(nameof(varianceTextureB));
        private readonly TextureBinding radianceCachePositionTexture = new(nameof(radianceCachePositionTexture), TextureTarget.Texture3D);
        private readonly TextureBinding radianceCacheNormalTexture = new(nameof(radianceCacheNormalTexture), TextureTarget.Texture3D);
        private readonly TextureBinding radianceCacheRadianceTexture = new(nameof(radianceCacheRadianceTexture), TextureTarget.Texture3D);
        private readonly TextureBinding radianceCacheFogTexture = new(nameof(radianceCacheFogTexture), TextureTarget.Texture3D);
        private readonly TextureBinding skyTexture = new(nameof(skyTexture));
        private readonly TextureBinding cloudTextureX = new(nameof(cloudTextureX));
        private readonly TextureBinding sunTexture = new(nameof(sunTexture));
        private readonly TextureBinding worleyNoiseTexture = new(nameof(worleyNoiseTexture), TextureTarget.Texture3D);
        private readonly TextureBinding Cloud3DTex = new(nameof(Cloud3DTex), TextureTarget.Texture3D);
        private readonly TextureBinding Cloud3DTexPrev = new(nameof(Cloud3DTexPrev), TextureTarget.Texture3D);



        // Quad data
        private readonly float[] fullscreenQuadVertices = [
           1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
            1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f
       ];
        private readonly uint[] fullscreenQuadIndices = [0, 1, 3, 1, 2, 3];


        // Camera and rendering state
        private Camera camera = default!;
        public Camera Camera => camera;
        public int OutputFramebuffer { get; set; } = 0;
        private Vector3 previousCameraPosition = new(0.0f, 0.0f, 0.0f);
        private Vector2 renderResolution = new(1920.0f, 1080.0f);
        private Matrix4 previousViewMatrix, previousProjectionMatrix;
        private Stopwatch frameTimer = default!;
        private float frameIndex = 0.0f;
        public Vector3 LightDirection = new(0.0f, 0.1f, -0.9f);
        public float SceneBrightness { get; set; } = 1.0f;
        public bool EnableFinalBlur { get; set; } = false;
        public TextureBinding? DebugTextureOverride { get; set; } = null;
        public float CloudCoverage { get; set; } = 1.0f;


        // Constants
        private const bool UseNativeResolution = true;
        private const float DefaultScale = 1.0f;
        private static float renderScale = DefaultScale;
        private const int ShadowMapResolution = 2048;
        private float[] textureBorderColor = { 1.0f, 1.0f, 1.0f, 1.0f };

        private bool isRendererInitialized = false;



        #endregion

        #region Public API
        public bool EnableOpenGLDebugging = false;

        public void SetCloudCoverage(float value) => CloudCoverage = value;
        public void SetLightDirection(float x, float y, float z) => LightDirection = new Vector3(x, y, z);


        private void Configure2DTexture(
    TextureBinding textureS,
    bool scaleToRenderResolution = true,
    PixelInternalFormat internalFormat = PixelInternalFormat.Rgba32f,
    TextureMinFilter minFilter = TextureMinFilter.Nearest,
    TextureMagFilter magFilter = TextureMagFilter.Nearest,
    TextureWrapMode wrapMode = TextureWrapMode.MirroredRepeat,
    bool useCustomSize = false,
    int customWidth = 0,
    int customHeight = 0, bool dontadd = false)
        {
            GL.ActiveTexture(TextureUnit.Texture0 + textureS.Unit);
            GL.BindTexture(TextureTarget.Texture2D, textureS.Handle);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)minFilter);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)magFilter);
            
            

            if (useCustomSize)
            {
                GL.TexImage2D(
                   TextureTarget.Texture2D, 0,
                   internalFormat,
                   customWidth, customHeight, 0,
                   PixelFormat.Rgba,
                   PixelType.UnsignedByte, IntPtr.Zero);
            }
            else {
                GL.TexImage2D(
                        TextureTarget.Texture2D, 0,
                        internalFormat,
                        (int)(renderResolution.X), (int)(renderResolution.Y), 0,
                        PixelFormat.Rgba,
                        PixelType.UnsignedByte, IntPtr.Zero);
            }

            textureS.isCustomSize = useCustomSize;
            textureS.minFilter = minFilter;
            textureS.magFilter = magFilter;

            if (useCustomSize) {
                System.Numerics.Vector2 nameSize = new System.Numerics.Vector2(customWidth, customHeight);
                textureS.customSize = nameSize;
            }
            if (!dontadd)
            {
                Textures2D.Add(textureS);
            }


         //   GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, textureBorderColor);
        }

        private void Configure3DTexture(
    TextureBinding textureS,
    int width,
    int height,
    int depth,
    PixelInternalFormat internalFormat = PixelInternalFormat.Rgba16f,
    TextureMinFilter minFilter = TextureMinFilter.Nearest,
    TextureMagFilter magFilter = TextureMagFilter.Nearest,
    TextureWrapMode wrapModeXY = TextureWrapMode.Repeat,
    TextureWrapMode wrapModeZ = TextureWrapMode.Repeat)
        {
            GL.ActiveTexture(TextureUnit.Texture0+textureS.Unit);
            GL.BindTexture(TextureTarget.Texture3D, textureS.Handle);

            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureMinFilter, (int)minFilter);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureMagFilter, (int)magFilter);

            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapS, (int)wrapModeXY);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapT, (int)wrapModeXY);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapR, (int)wrapModeZ);

            GL.TexImage3D(
                TextureTarget.Texture3D, 0,
                internalFormat,
                width, height, depth, 0,
                PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);
        }
        /// <summary>
        /// Used to match sampler indices to color attachment indices.
        /// </summary>
        private static readonly DrawBuffersEnum[] drawBuffers = [
            DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3,
            DrawBuffersEnum.ColorAttachment4, DrawBuffersEnum.ColorAttachment5, DrawBuffersEnum.ColorAttachment6, DrawBuffersEnum.ColorAttachment7];

        /// <summary>
        /// Helper method that creates a new framebuffer object,
        /// sets up the specified textures as color attachments in increasing order and validates completeness.
        /// </summary>
        /// <param name="colorAttachments"></param>
        /// <returns>The handle of the created framebuffer object.</returns>
        private static int CreateFramebufferObject(params TextureBinding[] colorAttachments)
        {
            if (colorAttachments.Length > 8)
                throw new ArgumentException($"A maximum of 8 textures can be added as color attachment. Provided: {colorAttachments.Length}");

            var fboHandle = GL.GenFramebuffer();
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, fboHandle);
            for (int i = 0; i < colorAttachments.Length; i++)
                GL.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0 + i, colorAttachments[i].Target, colorAttachments[i].Handle, 0);
            GL.DrawBuffers(colorAttachments.Length, drawBuffers);

            var status = GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer);
            if (status is not FramebufferErrorCode.FramebufferComplete)
                throw new Exception($"Incomplete Framebuffer status: {status}");
            return fboHandle;
        }

        public void Load(Vector2i resolution)
        {
            
            GL.ClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            //SetSize(new Vector2i(960, 540), new Vector2i(1920, 1080));
      
            if (UseNativeResolution)
            {
                renderScale = 1.0f;
            }


        




            // Setup 2D textures
            Configure2DTexture(fogColorTextureB);
            Configure2DTexture(fogColorTexture);
            Configure2DTexture(waterNormalTexture);
            Configure2DTexture(waterPositionTexture);
            Configure2DTexture(skyTexture, true, PixelInternalFormat.Rgba16f, TextureMinFilter.LinearMipmapLinear, TextureMagFilter.Linear);
            Configure2DTexture(sunTexture, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear, TextureWrapMode.ClampToBorder, true, ShadowMapResolution, ShadowMapResolution);
            Configure3DTexture(worleyNoiseTexture, 1024, 1024, 512, PixelInternalFormat.R16f, TextureMinFilter.Linear, TextureMagFilter.Linear, TextureWrapMode.Repeat, TextureWrapMode.MirroredRepeat);
            Configure3DTexture(Cloud3DTex, 256, 256, 128, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear, TextureWrapMode.Repeat, TextureWrapMode.MirroredRepeat);
            Configure3DTexture(Cloud3DTexPrev, 512, 512, 256, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear, TextureWrapMode.Repeat, TextureWrapMode.MirroredRepeat);

            Configure3DTexture(radianceCachePositionTexture, 180, 180, 180, PixelInternalFormat.Rgba32f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            Configure3DTexture(radianceCacheFogTexture, 180, 180, 180, PixelInternalFormat.Rgba32f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            Configure3DTexture(radianceCacheNormalTexture, 180, 180, 180, PixelInternalFormat.Rgba32f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            Configure3DTexture(radianceCacheRadianceTexture, 180, 180, 180, PixelInternalFormat.Rgba16f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            Configure2DTexture(colorTexture, false);
           
            
            Configure2DTexture(cloudTextureX, false);

            Configure2DTexture(sssTexture, false);
            Configure2DTexture(sssDenoiseTexture, false, PixelInternalFormat.Rgba32f);
            Configure2DTexture(positionTexture, true, PixelInternalFormat.Rgba32f);
            Configure2DTexture(fogPositionTexture, false);
            Configure2DTexture(holdInfoTexture);
            Configure2DTexture(normalTexture, true, PixelInternalFormat.Rgba32f);
            Configure2DTexture(albedoTexture);
            Configure2DTexture(secondaryPositionTexture, true, PixelInternalFormat.Rgba32f);
            Configure2DTexture(reflectionTexture, false);
            Configure2DTexture(secondaryAlbedoTexture, false);
            Configure2DTexture(temporalWeightsTexture, false);
            Configure2DTexture(temporalRadianceTexture, false);
            Configure2DTexture(temporalPositionTexture, false, PixelInternalFormat.Rgba32f);
            Configure2DTexture(spatialWeightsTexture, false);
            Configure2DTexture(spatialRadianceTexture, false);
            Configure2DTexture(taaTexture, true, PixelInternalFormat.Rgba32f, TextureMinFilter.Nearest, TextureMagFilter.Nearest);
            Configure2DTexture(upscaleTexture, true, PixelInternalFormat.Rgba32f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.Repeat);
            Configure2DTexture(upscaleTextureB, true, PixelInternalFormat.Rgba16f, TextureMinFilter.LinearMipmapLinear, TextureMagFilter.Linear);
            Configure2DTexture(prevTemporalWeightsTexture, false);
            Configure2DTexture(prevTemporalRadianceTexture, false);
            Configure2DTexture(prevTemporalPositionTexture, false);
            Configure2DTexture(prevSpatialWeightsTexture, false);
            Configure2DTexture(prevSpatialRadianceTexture, false);
            Configure2DTexture(prevNormalDepthTexture, true, PixelInternalFormat.Rgba32f);
            Configure2DTexture(prevAccumulationTexture, true);
            Configure2DTexture(prevTaaTexture, false, PixelInternalFormat.Rgba16f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.ClampToBorder );
            Configure2DTexture(denoiseTextureA, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);
            Configure2DTexture(denoiseTextureB, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);
            Configure2DTexture(varianceTextureA, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);
            Configure2DTexture(varianceTextureB, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);
            Configure2DTexture(tempAccumulationTexture);
            Configure2DTexture(upscaleTex1, false, PixelInternalFormat.Rgba32f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.ClampToEdge);
            Configure2DTexture(upscaleTex2, false, PixelInternalFormat.Rgba32f, TextureMinFilter.Linear, TextureMagFilter.Linear, TextureWrapMode.ClampToEdge );
            Configure2DTexture(upscaleTex3, false, PixelInternalFormat.Rgba32f, TextureMinFilter.Linear, TextureMagFilter.Linear, TextureWrapMode.ClampToEdge );


            

            // stb_image loads from the top-left pixel, whereas OpenGL loads from the bottom-left, causing the texture to be flipped vertically.
            // This will correct that, making the texture display properly.
            StbImage.stbi_set_flip_vertically_on_load(1);

            // Load the image.
            ImageResult image = ImageResult.FromStream(File.OpenRead("Textures/cubemap.png"), ColorComponents.RedGreenBlueAlpha);
            GL.ActiveTexture(TextureUnit.Texture0 + cubemap.Unit);
            GL.BindTexture(TextureTarget.Texture2D, cubemap.Handle);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba32f, image.Width, image.Height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, image.Data);

            // Setup framebuffers
            skyFramebuffer = CreateFramebufferObject(skyTexture);
            sunFramebuffer = CreateFramebufferObject(sunTexture);
            gBufferFramebuffer = CreateFramebufferObject(positionTexture, normalTexture, albedoTexture, holdInfoTexture, fogColorTexture, waterPositionTexture, waterNormalTexture, cloudTextureX);
            mainFramebuffer = CreateFramebufferObject(colorTexture, secondaryPositionTexture, reflectionTexture, secondaryAlbedoTexture, fogColorTexture, sssTexture, sssDenoiseTexture);
            temporalFramebuffer = CreateFramebufferObject(temporalWeightsTexture, temporalRadianceTexture, temporalPositionTexture);
            spatialFramebuffer = CreateFramebufferObject(spatialWeightsTexture, spatialRadianceTexture, fogPositionTexture);
            taaFramebuffer = CreateFramebufferObject(taaTexture);
            upscaleFramebuffer = CreateFramebufferObject(upscaleTexture);
            previousFrameFramebuffer = CreateFramebufferObject(prevNormalDepthTexture, prevAccumulationTexture);
            previousTemporalFramebuffer = CreateFramebufferObject(prevTemporalWeightsTexture, prevTemporalRadianceTexture, prevTemporalPositionTexture, prevSpatialWeightsTexture, prevSpatialRadianceTexture);
            denoiseFramebufferA = CreateFramebufferObject(denoiseTextureB, varianceTextureB, fogColorTextureB, secondaryAlbedoTexture, sssDenoiseTexture);
            denoiseFramebufferB = CreateFramebufferObject(denoiseTextureA, varianceTextureA, fogColorTexture, secondaryAlbedoTexture, sssDenoiseTexture);
            tempAccumulationFramebuffer = CreateFramebufferObject(tempAccumulationTexture, denoiseTextureA, varianceTextureA);
            spatialPrepassFramebuffer = CreateFramebufferObject(prevTemporalWeightsTexture, prevTemporalRadianceTexture, prevTemporalPositionTexture);
            upscale1 = CreateFramebufferObject(upscaleTex1,prevTaaTexture);
            upscale2 = CreateFramebufferObject(upscaleTex2);
            upscale3 = CreateFramebufferObject(upscaleTex3);

            ups1 = new Shader("swap2.vert", "swap2 - Copy.frag");
            ups2 = new Shader("swap2.vert", "swap2 - Copy - Copy.frag");
            ups3 = new Shader("swap2.vert", "swap2 - Copy - Copy (2).frag");

            // Compile shaders
            skyShader = new Shader("sky.vert", "sky.frag");
            sunShader = new Shader("sun.vert", "sun.frag", true);
            geometryPassShader = new Shader("Test.vert", "Test.frag", true);
            initializationShader = new Shader("initial.vert", "initial.frag", true);
            finalCompositeShader = new Shader("final.vert", "final.frag", true);
            temporalResamplingShader = new Shader("temporal.vert", "temporal.frag");
            previousFrameCopyShader = new Shader("swap.vert", "swap.frag");
            previousTemporalCopyShader = new Shader("swap2.vert", "swap2.frag");
            spatialResamplingShader = new Shader("spatial.vert", "spatial.frag", true);
            tempAccumulationShader = new Shader("tempAccum.vert", "tempAccum.frag");
            denoiseShader = new Shader("denoise1.vert", "denoise1.frag");
            taaShader = new Shader("TAA.vert", "TAA.frag");
            radianceCacheShader = new Shader("RadianceCaching.comp", true);
  
            upscaleShader = new Shader("upscale.vert", "upscale.frag");
            worleyNoiseShader = new Shader("worley.comp", false);
            cloudShader = new Shader("cloudsShader.comp", true);

            spatialPrepassShader = new Shader("spatPrePass.vert", "spatPrePass.frag");

            // Setup fullscreen quad
            fullscreenQuadVbo = GL.GenBuffer();
            fullscreenQuadVao = GL.GenVertexArray();
            fullscreenQuadEbo = GL.GenBuffer();

            ups1.Use();
            // upscaleTex3
            ups1.SetInt("col", taaTexture.Handle);
            ups1.SetInt("prevTAA", prevTaaTexture.Handle);
            ups1.SetInt("position", positionTexture.Handle);
            ups1.SetInt("norm", normalTexture.Handle);
            ups1.SetInt("prevnorm", prevNormalDepthTexture.Handle);

            ups2.Use();
            ups2.SetInt("col", upscaleTexture.Handle);
            ups3.Use();
            ups3.SetInt("col", upscaleTex2.Handle);
            ups3.SetInt("orig", taaTexture.Handle);


            initializationShader.Use();
            initializationShader.SetInt("skytex", skyTexture.Handle);
            initializationShader.SetInt("suntex", sunTexture.Handle);

            skyShader.Use();
            skyShader.SetInt("skyprev", skyTexture.Handle);
            skyShader.SetInt("worl", worleyNoiseTexture.Handle);
            skyShader.SetInt("cubemap", cubemap.Handle);
            //skyShader.SetInt("suns", sunTexture.Handle);

            sunShader.Use();
            //sunShader.SetInt("sunprev", sunTexture.Handle);
            //sunShader.SetInt("worl", worleyNoiseTexture.Handle);

            radianceCacheShader.Use();
            radianceCacheShader.SetInt("skytex", skyTexture.Handle);
            radianceCacheShader.SetInt("suntex", sunTexture.Handle);

            geometryPassShader.Use();
            geometryPassShader.SetInt("position", positionTexture.Handle);
            geometryPassShader.SetInt("normal", normalTexture.Handle);
            geometryPassShader.SetInt("albedo", albedoTexture.Handle);
            geometryPassShader.SetInt("holdinfo", holdInfoTexture.Handle);
            //geometryPassShader.SetInt("colorfog", fogColorTexture.Handle);
            geometryPassShader.SetInt("skytex", skyTexture.Handle);
            geometryPassShader.SetInt("suntex", sunTexture.Handle);
            geometryPassShader.SetInt("watpos", waterPositionTexture.Handle);
            geometryPassShader.SetInt("watnorm", waterNormalTexture.Handle);
            geometryPassShader.SetInt("sssprev", sssTexture.Handle);
            geometryPassShader.SetInt("prevN", prevNormalDepthTexture.Handle);
            geometryPassShader.SetInt("den1", denoiseTextureB.Handle);

            temporalResamplingShader.Use();
            temporalResamplingShader.SetInt("color", colorTexture.Handle);
            temporalResamplingShader.SetInt("position", positionTexture.Handle);
            temporalResamplingShader.SetInt("normal", normalTexture.Handle);
            temporalResamplingShader.SetInt("secondpos", secondaryPositionTexture.Handle);
            temporalResamplingShader.SetInt("prevW", prevTemporalWeightsTexture.Handle);
            temporalResamplingShader.SetInt("prevL", prevTemporalRadianceTexture.Handle);
            temporalResamplingShader.SetInt("prevP", prevTemporalPositionTexture.Handle);
            temporalResamplingShader.SetInt("prevN", prevNormalDepthTexture.Handle);

            spatialPrepassShader.Use();
            spatialPrepassShader.SetInt("color", colorTexture.Handle);
            spatialPrepassShader.SetInt("position", positionTexture.Handle);
            spatialPrepassShader.SetInt("normal", normalTexture.Handle);
            spatialPrepassShader.SetInt("secondpos", secondaryPositionTexture.Handle);
            spatialPrepassShader.SetInt("prevW", temporalWeightsTexture.Handle);
            spatialPrepassShader.SetInt("prevL", temporalRadianceTexture.Handle);
            spatialPrepassShader.SetInt("prevP", temporalPositionTexture.Handle);
            spatialPrepassShader.SetInt("prevN", prevNormalDepthTexture.Handle);
            //spatialPrepassShader.SetInt("reflAlb", secondaryAlbedoTexture.Handle);

            spatialResamplingShader.Use();
            //spatialResamplingShader.SetInt("color", colorTexture.Handle);
            spatialResamplingShader.SetInt("position", positionTexture.Handle);
            spatialResamplingShader.SetInt("normal", normalTexture.Handle);
            spatialResamplingShader.SetInt("albedo", albedoTexture.Handle);
            //spatialResamplingShader.SetInt("secondpos", secondaryPositionTexture.Handle);
            spatialResamplingShader.SetInt("temppos", temporalPositionTexture.Handle);
            spatialResamplingShader.SetInt("prevW", prevSpatialWeightsTexture.Handle);
            spatialResamplingShader.SetInt("prevL", prevSpatialRadianceTexture.Handle);
            spatialResamplingShader.SetInt("tempW", temporalWeightsTexture.Handle);
            spatialResamplingShader.SetInt("tempL", temporalRadianceTexture.Handle);
            spatialResamplingShader.SetInt("prevN", prevNormalDepthTexture.Handle);
            spatialResamplingShader.SetInt("prevSPos", fogPositionTexture.Handle);
            spatialResamplingShader.SetInt("holdinfo", holdInfoTexture.Handle);

            tempAccumulationShader.Use();
            //tempAccumulationShader.SetInt("color", colorTexture.Handle);
            tempAccumulationShader.SetInt("position", positionTexture.Handle);
            tempAccumulationShader.SetInt("normal", normalTexture.Handle);
            tempAccumulationShader.SetInt("albedo", albedoTexture.Handle);
            tempAccumulationShader.SetInt("weightS", spatialWeightsTexture.Handle);
            tempAccumulationShader.SetInt("outgoingrS", spatialRadianceTexture.Handle);
            tempAccumulationShader.SetInt("prevN", prevNormalDepthTexture.Handle);
            tempAccumulationShader.SetInt("prevAcc", prevAccumulationTexture.Handle);
            tempAccumulationShader.SetInt("prevvar", varianceTextureB.Handle);
            tempAccumulationShader.SetInt("prevden", denoiseTextureB.Handle);

            denoiseShader.Use();
            denoiseShader.SetInt("color", temporalRadianceTexture.Handle);
            //denoiseShader.SetInt("position", positionTexture.Handle);
            denoiseShader.SetInt("normal", normalTexture.Handle);
            denoiseShader.SetInt("albedo", albedoTexture.Handle);
            denoiseShader.SetInt("secondpos", secondaryPositionTexture.Handle);
            denoiseShader.SetInt("reflN", reflectionTexture.Handle);
            denoiseShader.SetInt("reflectionAlb", secondaryAlbedoTexture.Handle);
            denoiseShader.SetInt("holdinfo", holdInfoTexture.Handle);
            denoiseShader.SetInt("watpos", waterPositionTexture.Handle);
            denoiseShader.SetInt("watnorm", waterNormalTexture.Handle);
            denoiseShader.SetInt("tempacc", tempAccumulationTexture.Handle);
            denoiseShader.SetInt("sstexden", sssDenoiseTexture.Handle);
            denoiseShader.SetInt("colorTex", colorTexture.Handle);
            // Ping-pong textures are set each frame, omitted here

            taaShader.Use();
            //taaShader.SetInt("color", colorTexture.Handle); //TODO: review. This was commented out.
            taaShader.SetInt("position", positionTexture.Handle);
            taaShader.SetInt("normal", normalTexture.Handle);
            taaShader.SetInt("albedo", albedoTexture.Handle);
            taaShader.SetInt("secondpos", secondaryPositionTexture.Handle);
            //taaShader.SetInt("Acc", tempAccumulationTexture.Handle);
            taaShader.SetInt("den1", denoiseTextureB.Handle); //den2 bound here
            taaShader.SetInt("reflAlb", secondaryAlbedoTexture.Handle);
            taaShader.SetInt("colorfog", fogColorTexture.Handle);
            //taaShader.SetInt("holdinfo", holdInfoTexture.Handle);
            taaShader.SetInt("suntex", sunTexture.Handle);
            taaShader.SetInt("skyTex", skyTexture.Handle);
            taaShader.SetInt("watpos", waterPositionTexture.Handle);
            taaShader.SetInt("watnorm", waterNormalTexture.Handle);
            taaShader.SetInt("worl", worleyNoiseTexture.Handle);
            taaShader.SetInt("sssT", sssDenoiseTexture.Handle);
            taaShader.SetInt("sssT2", sssTexture.Handle);
            taaShader.SetInt("prevN", prevNormalDepthTexture.Handle);
            taaShader.SetInt("holdinfo", holdInfoTexture.Handle);
            taaShader.SetInt("color", colorTexture.Handle);
            taaShader.SetInt("var", varianceTextureA.Handle);
            taaShader.SetInt("CLOOOOUDS", cloudTextureX.Handle);

            upscaleShader.Use();
            upscaleShader.SetInt("TAA", taaTexture.Handle);

            previousFrameCopyShader.Use();
            previousFrameCopyShader.SetInt("normal", normalTexture.Handle);
            previousFrameCopyShader.SetInt("Acc", tempAccumulationTexture.Handle);
            previousFrameCopyShader.SetInt("TAA", taaTexture.Handle);

            previousTemporalCopyShader.Use();
            previousTemporalCopyShader.SetInt("tempWeights", temporalWeightsTexture.Handle);
            previousTemporalCopyShader.SetInt("tempOutL", temporalRadianceTexture.Handle);
            previousTemporalCopyShader.SetInt("tempPos", temporalPositionTexture.Handle);
            previousTemporalCopyShader.SetInt("spatWe", spatialWeightsTexture.Handle);
            previousTemporalCopyShader.SetInt("spatLO", spatialRadianceTexture.Handle);

            finalCompositeShader.Use();
            finalCompositeShader.SetInt("color", prevSpatialWeightsTexture.Handle);
            finalCompositeShader.SetInt("normal", normalTexture.Handle);
            finalCompositeShader.SetInt("albedo", albedoTexture.Handle);
            finalCompositeShader.SetInt("TAA", taaTexture.Handle); //TODO: review. "TAA" sampler is bound to "upscale" here.
            finalCompositeShader.SetInt("fogLO", upscaleTex1.Handle); //TODO: review. "fogLO" sampler is bound to "TAA" here.
            finalCompositeShader.SetInt("watpos", waterPositionTexture.Handle);
            finalCompositeShader.SetInt("position", positionTexture.Handle);
            finalCompositeShader.SetInt("var1", varianceTextureB.Handle);
            finalCompositeShader.SetInt("clll", cloudTextureX.Handle);


            GL.BindVertexArray(fullscreenQuadVao);
            GL.BindBuffer(BufferTarget.ArrayBuffer, fullscreenQuadVbo);
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, fullscreenQuadEbo);
            GL.BufferData(BufferTarget.ArrayBuffer, fullscreenQuadVertices.Length * sizeof(float), fullscreenQuadVertices, BufferUsageHint.StaticDraw);
            GL.BufferData(BufferTarget.ElementArrayBuffer, fullscreenQuadIndices.Length * sizeof(uint), fullscreenQuadIndices, BufferUsageHint.StaticDraw);

            int vertexPositionLocation = GL.GetAttribLocation(geometryPassShader.Handle, "aPosition");
            GL.EnableVertexAttribArray(vertexPositionLocation);
            GL.VertexAttribPointer(vertexPositionLocation, 3, VertexAttribPointerType.Float, false, 5 * sizeof(float), 0);

            int texCoordLocation = GL.GetAttribLocation(geometryPassShader.Handle, "aTexCoord");
            GL.EnableVertexAttribArray(texCoordLocation);
            GL.VertexAttribPointer(texCoordLocation, 2, VertexAttribPointerType.Float, false, 5 * sizeof(float), 3 * sizeof(float));

            camera = new Camera(Vector3.UnitZ * 3, renderResolution.X / (float)renderResolution.Y);
            frameTimer = new Stopwatch();
            frameTimer.Start();

            // Setup compute textures and dispatch compute shaders as needed
            GL.BindImageTexture(7, worleyNoiseTexture.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.R16f);
            worleyNoiseShader.Use();
            GL.DispatchCompute(128, 128, 64);

            GL.BindImageTexture(0, radianceCachePositionTexture.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba32f);
            GL.BindImageTexture(1, radianceCacheNormalTexture.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba32f);
            GL.BindImageTexture(2, radianceCacheRadianceTexture.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            GL.BindImageTexture(3, radianceCacheFogTexture.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba32f);
            GL.BindImageTexture(4, sunTexture.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            GL.BindImageTexture(5, Cloud3DTex.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            GL.BindImageTexture(6, Cloud3DTexPrev.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            cloudShader.Use();
            cloudShader.SetInt("skytex", skyTexture.Handle);
            cloudShader.SetInt("suntex", sunTexture.Handle);


            isRendererInitialized = true;
        }

        public void RenderFrame()
        {
            if (!isRendererInitialized)
                throw new InvalidOperationException("Renderer must be initialized before rendering.");

            LightDirection = Vector3.Normalize(LightDirection);
            var (lightViewMatrix, lightProjectionMatrix, lightPosition) = CalculateLightMatrices();

            var viewMatrix = camera.GetViewMatrix();
            var projectionMatrix = camera.GetProjectionMatrix();

            RenderSkyPass();
            RenderSunPass(lightViewMatrix, lightProjectionMatrix, LightDirection, lightPosition);

            cloudShader.Use();
            cloudShader.SetVector3(LightDirection, "ldir");
            cloudShader.SetFloat(CloudCoverage, "cloudCoverage");
            cloudShader.SetFloat(frameIndex, "time");
            cloudShader.SetMatrix4(lightProjectionMatrix, "lightproj");
            cloudShader.SetMatrix4(lightViewMatrix, "lightview");
            cloudShader.SetVector3(camera.Position, "viewPos");

            GL.DispatchCompute(32, 32, 16);
            GL.MemoryBarrier(MemoryBarrierFlags.AllBarrierBits);



            RenderInitializationPass(viewMatrix, projectionMatrix, lightViewMatrix, lightProjectionMatrix);
            RenderRadianceCachePass(lightViewMatrix, lightProjectionMatrix, LightDirection);
            RenderGeometryPass(viewMatrix, projectionMatrix, lightViewMatrix, lightProjectionMatrix, LightDirection);
            RenderTemporalResamplingPass();
            RenderSpatPrePass();
            RenderSpatialResamplingPass();
            RenderTempAccumulationPass();
            RenderDenoisePass();
            RenderTaaPass(viewMatrix, projectionMatrix, lightViewMatrix, lightProjectionMatrix, LightDirection);
            RenderPreviousFrameCopyPass();
            RenderPreviousTemporalCopyPass();

            RenderUpscalePass();


          

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, upscale2);

            GL.Viewport(0, 0, (int)(renderResolution.X), (int)(renderResolution.Y));

            ups2.Use();
            ups2.SetVector2(new Vector2(renderResolution.X , renderResolution.Y ), "wh");
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);



            GL.BindFramebuffer(FramebufferTarget.Framebuffer, upscale3);

            GL.Viewport(0, 0, (int)(renderResolution.X), (int)(renderResolution.Y));

            ups3.Use();
            ups3.SetVector2(new Vector2(renderResolution.X , renderResolution.Y ), "wh");
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);


            GL.BindFramebuffer(FramebufferTarget.Framebuffer, upscale1);

            GL.Viewport(0, 0, (int)(renderResolution.X), (int)(renderResolution.Y));

            ups1.Use();
            ups1.SetFloat(frameIndex, "time");
            ups1.SetVector2(new Vector2(renderResolution.X, renderResolution.Y), "wh");
            ups1.SetMatrix4(previousViewMatrix, "prevview");
            ups1.SetMatrix4(previousProjectionMatrix, "prevproj");
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);



            RenderFinalCompositePass(viewMatrix, projectionMatrix);

            UpdatePreviousFrameData(viewMatrix, projectionMatrix);
            
            frameIndex += 1.0f;
        }


        public void UpdateResolution(Vector2i resolution)
        {
            camera.AspectRatio = resolution.X / (float)resolution.Y;
            GL.Viewport(0, 0, resolution.X, resolution.Y);
            renderResolution = resolution;

            for (int i = 0; i < Textures2D.Count; i++) { 
                TextureBinding tex = Textures2D[i];
                Configure2DTexture(tex, false, PixelInternalFormat.Rgba32f, tex.minFilter, tex.magFilter, 
                    TextureWrapMode.ClampToEdge, tex.isCustomSize, (int)tex.customSize.X, (int)tex.customSize.Y, true);
            }



        }

        #endregion

        #region Private Methods

        private (Matrix4 lightViewMatrix, Matrix4 lightProjectionMatrix, Vector3 lightPosition) CalculateLightMatrices()
        {
            Vector3 lightPosition = camera.Position + LightDirection * 270.0f;
            Matrix4 lightViewMatrix = Matrix4.LookAt(lightPosition, new Vector3(camera.Position.X, 0.0f, camera.Position.Z), Vector3.UnitY);
            Matrix4 lightProjectionMatrix = Matrix4.CreateOrthographic(400.0f, 400.0f, 0.0f, 127.5f);
            return (lightViewMatrix, lightProjectionMatrix, lightPosition);
        }

        private void RenderInitializationPass(Matrix4 viewMatrix, Matrix4 projectionMatrix, Matrix4 lightViewMatrix, Matrix4 lightProjectionMatrix)
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, gBufferFramebuffer);
            GL.Viewport(0, 0, (int)(renderResolution.X * renderScale), (int)(renderResolution.Y * renderScale));
            GL.Disable(EnableCap.DepthTest);
            initializationShader.Use();
            initializationShader.SetMatrix4(viewMatrix.Inverted(), "invview");
            initializationShader.SetMatrix4(projectionMatrix.Inverted(), "invproj");
            initializationShader.SetVector2(new Vector2(renderResolution.X * DefaultScale, renderResolution.Y * DefaultScale), "wh");
            initializationShader.SetFloat(frameIndex, "time");
            initializationShader.SetVector3(camera.Position, "viewPos");
            initializationShader.SetVector3(LightDirection, "ldir");
            initializationShader.SetFloat(CloudCoverage, "cloudCoverage");
            initializationShader.SetMatrix4(lightProjectionMatrix, "lightproj");
            initializationShader.SetMatrix4(lightViewMatrix, "lightview");
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
        }

        private void RenderSkyPass()
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, skyFramebuffer);
            GL.Viewport(0, 0, (int)(renderResolution.X * renderScale), (int)(renderResolution.Y * renderScale));
            GL.Disable(EnableCap.DepthTest);
            skyShader.Use();
            skyShader.SetVector2(new Vector2(renderResolution.X * DefaultScale, renderResolution.Y * DefaultScale), "wh");
            skyShader.SetFloat(frameIndex, "time");
            skyShader.SetVector3(LightDirection, "ldir");
            skyShader.SetFloat(CloudCoverage, "cloudCoverage");
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
            GL.GenerateTextureMipmap(skyTexture.Handle);
        }

        private void RenderSunPass(Matrix4 lightViewMatrix, Matrix4 lightProjectionMatrix, Vector3 lightDirection, Vector3 lightPosition)
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, sunFramebuffer);
            GL.Viewport(0, 0, ShadowMapResolution, ShadowMapResolution);
            GL.Disable(EnableCap.DepthTest);
            sunShader.Use();
            sunShader.SetFloat(frameIndex, "time");
            sunShader.SetVector3(lightDirection, "ldir");
            sunShader.SetVector3(lightPosition, "lpos");
            sunShader.SetMatrix4(lightProjectionMatrix, "lightproj");
            sunShader.SetMatrix4(lightViewMatrix, "lightview");
            sunShader.SetFloat(ShadowMapResolution, "scale");
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
        }

        private void RenderRadianceCachePass(Matrix4 lightViewMatrix, Matrix4 lightProjectionMatrix, Vector3 lightDirection)
        {
            radianceCacheShader.Use();
            radianceCacheShader.SetFloat(frameIndex, "time");
            radianceCacheShader.SetVector3(camera.Position, "viewPos");
            radianceCacheShader.SetVector3(previousCameraPosition, "lastViewPos");
            radianceCacheShader.SetVector3(lightDirection, "ldir");
            radianceCacheShader.SetMatrix4(lightProjectionMatrix, "lightproj");
            radianceCacheShader.SetMatrix4(lightViewMatrix, "lightview");
            GL.DispatchCompute(45, 45, 45);
        }

        private void RenderGeometryPass(Matrix4 viewMatrix, Matrix4 projectionMatrix, Matrix4 lightViewMatrix, Matrix4 lightProjectionMatrix, Vector3 lightDirection)
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, mainFramebuffer);
            GL.Viewport(0, 0, (int)renderResolution.X, (int)renderResolution.Y);
            GL.Disable(EnableCap.DepthTest);
            geometryPassShader.Use();
            geometryPassShader.SetMatrix4(viewMatrix.Inverted(), "invview");
            geometryPassShader.SetMatrix4(projectionMatrix.Inverted(), "invproj");
            geometryPassShader.SetVector2(new Vector2(renderResolution.X * DefaultScale, renderResolution.Y * DefaultScale), "wh");
            geometryPassShader.SetFloat(frameIndex, "time");
            geometryPassShader.SetVector3(camera.Position, "viewPos");
            geometryPassShader.SetVector3(lightDirection, "ldir");
            geometryPassShader.SetMatrix4(lightProjectionMatrix, "lightproj");
            geometryPassShader.SetMatrix4(lightViewMatrix, "lightview");
            geometryPassShader.SetMatrix4(previousViewMatrix, "prevview");
            geometryPassShader.SetMatrix4(previousProjectionMatrix, "prevproj");
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
        }


        private void RenderTemporalResamplingPass()
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, temporalFramebuffer);
            GL.Viewport(0, 0, (int)renderResolution.X, (int)renderResolution.Y);
            GL.Disable(EnableCap.DepthTest);
            temporalResamplingShader.Use();
            temporalResamplingShader.SetMatrix4(previousViewMatrix, "prevview");
            temporalResamplingShader.SetMatrix4(previousProjectionMatrix, "prevproj");
            temporalResamplingShader.SetVector2(new Vector2(renderResolution.X * DefaultScale, renderResolution.Y * DefaultScale), "wh");
            temporalResamplingShader.SetFloat(frameIndex, "time");
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
        }

        private void RenderSpatPrePass()
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, spatialPrepassFramebuffer);
            GL.Viewport(0, 0, (int)renderResolution.X, (int)renderResolution.Y);
            GL.Disable(EnableCap.DepthTest);
            spatialPrepassShader.Use();
            spatialPrepassShader.SetMatrix4(previousViewMatrix, "prevview");
            spatialPrepassShader.SetMatrix4(previousProjectionMatrix, "prevproj");
            spatialPrepassShader.SetVector2(new Vector2(renderResolution.X * DefaultScale, renderResolution.Y * DefaultScale), "wh");
            spatialPrepassShader.SetFloat(frameIndex, "time");
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
        }

        private void RenderSpatialResamplingPass()
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, spatialFramebuffer);
            GL.Viewport(0, 0, (int)renderResolution.X, (int)renderResolution.Y);
            GL.Disable(EnableCap.DepthTest);
            spatialResamplingShader.Use();
            spatialResamplingShader.SetMatrix4(previousViewMatrix, "prevview");
            spatialResamplingShader.SetMatrix4(previousProjectionMatrix, "prevproj");
            spatialResamplingShader.SetVector2(new Vector2(renderResolution.X * DefaultScale, renderResolution.Y * DefaultScale), "wh");
            spatialResamplingShader.SetFloat(frameIndex, "time");
            spatialResamplingShader.SetVector3(camera.Position, "viewPos");
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
        }

        private void RenderTempAccumulationPass()
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, tempAccumulationFramebuffer);
            GL.Viewport(0, 0, (int)(renderResolution.X * renderScale), (int)(renderResolution.Y * renderScale));
            GL.Disable(EnableCap.DepthTest);
            tempAccumulationShader.Use();
            tempAccumulationShader.SetMatrix4(previousViewMatrix, "prevview");
            tempAccumulationShader.SetMatrix4(previousProjectionMatrix, "prevproj");
            tempAccumulationShader.SetVector2(new Vector2(renderResolution.X * DefaultScale, renderResolution.Y * DefaultScale), "wh");
            tempAccumulationShader.SetFloat(frameIndex, "time");
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
            GL.MemoryBarrier(MemoryBarrierFlags.AllBarrierBits);
        }

        private void RenderDenoisePass()
        {
            GL.Viewport(0, 0, (int)(renderResolution.X * renderScale), (int)(renderResolution.Y * renderScale));
            GL.Disable(EnableCap.DepthTest);
            denoiseShader.Use();
            denoiseShader.SetVector2(new Vector2(renderResolution.X * DefaultScale, renderResolution.Y * DefaultScale), "wh");
            denoiseShader.SetFloat(frameIndex, "time");

            for (int i = 0; i < 7; i++)
            {
                if (i % 2 == 0)
                {
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, denoiseFramebufferA);
                    denoiseShader.SetInt("den1", denoiseTextureA.Unit);
                    denoiseShader.SetInt("var1", varianceTextureA.Unit);
                    denoiseShader.SetInt("colorfog", fogColorTexture.Unit);
                }
                else
                {
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, denoiseFramebufferB);
                    denoiseShader.SetInt("den1", denoiseTextureB.Unit);
                    denoiseShader.SetInt("var1", varianceTextureB.Unit);
                    denoiseShader.SetInt("colorfog", fogColorTextureB.Unit);
                }
                GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
                GL.MemoryBarrier(MemoryBarrierFlags.AllBarrierBits);
            }
        }

        private void RenderTaaPass(Matrix4 viewMatrix, Matrix4 projectionMatrix, Matrix4 lightViewMatrix, Matrix4 lightProjectionMatrix, Vector3 lightDirection)
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, taaFramebuffer);
            GL.Viewport(0, 0, (int)(renderResolution.X * renderScale), (int)(renderResolution.Y * renderScale));
            GL.Disable(EnableCap.DepthTest);
            taaShader.Use();
            taaShader.SetMatrix4(viewMatrix.Inverted(), "invview");
            taaShader.SetMatrix4(projectionMatrix.Inverted(), "invproj");
            taaShader.SetMatrix4(previousViewMatrix, "prevview");
            taaShader.SetMatrix4(previousProjectionMatrix, "prevproj");
            taaShader.SetVector2(new Vector2(renderResolution.X * DefaultScale, renderResolution.Y * DefaultScale), "wh");
            taaShader.SetFloat(frameIndex, "time");
            taaShader.SetVector3(camera.Position, "viewPos");
            taaShader.SetVector3(lightDirection, "ldir");
            taaShader.SetMatrix4(lightProjectionMatrix, "lightproj");
            taaShader.SetMatrix4(lightViewMatrix, "lightview");
            taaShader.SetMatrix4(viewMatrix, "view");
            taaShader.SetMatrix4(projectionMatrix, "proj");
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
            //GL.GenerateTextureMipmap(taaTexture.Handle);
        }

        private void RenderUpscalePass()
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, upscaleFramebuffer);
            GL.Viewport(0, 0, (int)(renderResolution.X), (int)(renderResolution.Y));
            GL.Disable(EnableCap.DepthTest);
            upscaleShader.Use();
            upscaleShader.SetVector2(new Vector2(renderResolution.X, renderResolution.Y), "wh");
            upscaleShader.SetFloat(DefaultScale, "scale");
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
            //GL.GenerateTextureMipmap(upscaleTexture.Handle);
        }

        private void RenderPreviousFrameCopyPass()
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, previousFrameFramebuffer);
            GL.Viewport(0, 0, (int)(renderResolution.X * renderScale), (int)(renderResolution.Y * renderScale));
            GL.Disable(EnableCap.DepthTest);
            previousFrameCopyShader.Use();
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
            GL.GenerateTextureMipmap(upscaleTextureB.Handle);
        }

        private void RenderPreviousTemporalCopyPass()
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, previousTemporalFramebuffer);
            GL.Viewport(0, 0, (int)renderResolution.X, (int)renderResolution.Y);
            GL.Disable(EnableCap.DepthTest);
            previousTemporalCopyShader.Use();
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
        }

        private void RenderFinalCompositePass(Matrix4 viewMatrix, Matrix4 projectionMatrix)
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, OutputFramebuffer);
            GL.Viewport(0, 0, (int)(renderResolution.X), (int)(renderResolution.Y));
            GL.Disable(EnableCap.DepthTest);
            finalCompositeShader.Use();
            finalCompositeShader.SetMatrix4(projectionMatrix, "projection");
            finalCompositeShader.SetVector3(camera.Position, "viewPos");
            finalCompositeShader.SetVector2(new Vector2(renderResolution.X * DefaultScale, renderResolution.Y * DefaultScale), "wh");
            finalCompositeShader.SetFloat(frameIndex, "time");
            finalCompositeShader.SetFloat(SceneBrightness, "brightness");
            finalCompositeShader.SetInt("enableBlur", EnableFinalBlur ? 1 : 0);
            finalCompositeShader.SetInt("enableTextureDebugging", DebugTextureOverride is not null ? 1 : 0);
            finalCompositeShader.SetInt("debuggedTexture", DebugTextureOverride?.Unit ?? 0);
            finalCompositeShader.SetMatrix4(previousViewMatrix, "prevview");
            finalCompositeShader.SetMatrix4(previousProjectionMatrix, "prevproj");
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
        }

        private void UpdatePreviousFrameData(Matrix4 viewMatrix, Matrix4 projectionMatrix)
        {
            previousViewMatrix = viewMatrix;
            previousProjectionMatrix = projectionMatrix;
            previousCameraPosition = camera.Position;
        }

        #endregion
    }
}