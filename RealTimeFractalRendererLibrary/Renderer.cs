
using Common;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using System;
using System.Diagnostics;
using System.Drawing;
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

        // Texture bindings
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
        private readonly TextureBinding sunTexture = new(nameof(sunTexture));
        private readonly TextureBinding worleyNoiseTexture = new(nameof(worleyNoiseTexture), TextureTarget.Texture3D);


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
    int textureHandle,
    bool scaleToRenderResolution = true,
    PixelInternalFormat internalFormat = PixelInternalFormat.Rgba32f,
    TextureMinFilter minFilter = TextureMinFilter.Nearest,
    TextureMagFilter magFilter = TextureMagFilter.Nearest,
    TextureWrapMode wrapMode = TextureWrapMode.MirroredRepeat,
    bool useCustomSize = false,
    int customWidth = 0,
    int customHeight = 0)
        {
            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, textureHandle);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)minFilter);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)magFilter);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)wrapMode);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)wrapMode);

            if (scaleToRenderResolution && !useCustomSize)
            {
                GL.TexImage2D(
                    TextureTarget.Texture2D, 0,
                    internalFormat,
                    (int)(renderResolution.X * renderScale), (int)(renderResolution.Y * renderScale), 0,
                    PixelFormat.Rgba,
                    PixelType.UnsignedByte, IntPtr.Zero);
            }
            else if (!useCustomSize)
            {
                GL.TexImage2D(
                    TextureTarget.Texture2D, 0,
                    internalFormat,
                    (int)(renderResolution.X), (int)(renderResolution.Y), 0,
                    PixelFormat.Rgba,
                    PixelType.UnsignedByte, IntPtr.Zero);
            }
            else
            {
                GL.TexImage2D(
                    TextureTarget.Texture2D, 0,
                    internalFormat,
                    customWidth, customHeight, 0,
                    PixelFormat.Rgba,
                    PixelType.UnsignedByte, IntPtr.Zero);
            }
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, textureBorderColor);
        }

        private void Configure3DTexture(
    int textureHandle,
    int width,
    int height,
    int depth,
    PixelInternalFormat internalFormat = PixelInternalFormat.Rgba16f,
    TextureMinFilter minFilter = TextureMinFilter.Nearest,
    TextureMagFilter magFilter = TextureMagFilter.Nearest,
    TextureWrapMode wrapModeXY = TextureWrapMode.Repeat,
    TextureWrapMode wrapModeZ = TextureWrapMode.Repeat)
        {
            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture3D, textureHandle);

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

            if (UseNativeResolution)
            {
                renderScale = 1.0f;
            }

            // Setup 2D textures
            Configure2DTexture(fogColorTextureB.Handle);
            Configure2DTexture(fogColorTexture.Handle);
            Configure2DTexture(waterNormalTexture.Handle);
            Configure2DTexture(waterPositionTexture.Handle);
            Configure2DTexture(skyTexture.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.LinearMipmapLinear, TextureMagFilter.Linear);
            Configure2DTexture(sunTexture.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear, TextureWrapMode.ClampToBorder, true, ShadowMapResolution, ShadowMapResolution);
            Configure3DTexture(worleyNoiseTexture.Handle, 512, 512, 512, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear, TextureWrapMode.Repeat, TextureWrapMode.MirroredRepeat);
            Configure3DTexture(radianceCachePositionTexture.Handle, 180, 180, 180, PixelInternalFormat.Rgba32f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            Configure3DTexture(radianceCacheFogTexture.Handle, 180, 180, 180, PixelInternalFormat.Rgba32f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            Configure3DTexture(radianceCacheNormalTexture.Handle, 180, 180, 180, PixelInternalFormat.Rgba32f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            Configure3DTexture(radianceCacheRadianceTexture.Handle, 180, 180, 180, PixelInternalFormat.Rgba16f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            Configure2DTexture(colorTexture.Handle, false);
            Configure2DTexture(sssTexture.Handle, false);
            Configure2DTexture(sssDenoiseTexture.Handle, false, PixelInternalFormat.Rgba32f);
            Configure2DTexture(positionTexture.Handle, true, PixelInternalFormat.Rgba32f);
            Configure2DTexture(fogPositionTexture.Handle, false);
            Configure2DTexture(holdInfoTexture.Handle);
            Configure2DTexture(normalTexture.Handle, true, PixelInternalFormat.Rgba32f);
            Configure2DTexture(albedoTexture.Handle);
            Configure2DTexture(secondaryPositionTexture.Handle, true, PixelInternalFormat.Rgba32f);
            Configure2DTexture(reflectionTexture.Handle, false);
            Configure2DTexture(secondaryAlbedoTexture.Handle, false);
            Configure2DTexture(temporalWeightsTexture.Handle, false);
            Configure2DTexture(temporalRadianceTexture.Handle, false);
            Configure2DTexture(temporalPositionTexture.Handle, false, PixelInternalFormat.Rgba32f);
            Configure2DTexture(spatialWeightsTexture.Handle, false);
            Configure2DTexture(spatialRadianceTexture.Handle, false);
            Configure2DTexture(taaTexture.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.LinearMipmapLinear, TextureMagFilter.Linear);
            Configure2DTexture(upscaleTexture.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.LinearMipmapLinear, TextureMagFilter.Linear);
            Configure2DTexture(upscaleTextureB.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.LinearMipmapLinear, TextureMagFilter.Linear);
            Configure2DTexture(prevTemporalWeightsTexture.Handle, false);
            Configure2DTexture(prevTemporalRadianceTexture.Handle, false);
            Configure2DTexture(prevTemporalPositionTexture.Handle, false);
            Configure2DTexture(prevSpatialWeightsTexture.Handle, false);
            Configure2DTexture(prevSpatialRadianceTexture.Handle, false);
            Configure2DTexture(prevNormalDepthTexture.Handle, true, PixelInternalFormat.Rgba32f);
            Configure2DTexture(prevAccumulationTexture.Handle, true);
            Configure2DTexture(prevTaaTexture.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.LinearMipmapLinear, TextureMagFilter.Linear);
            Configure2DTexture(denoiseTextureA.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);
            Configure2DTexture(denoiseTextureB.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);
            Configure2DTexture(varianceTextureA.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);
            Configure2DTexture(varianceTextureB.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);
            Configure2DTexture(tempAccumulationTexture.Handle);

            // Setup framebuffers
            skyFramebuffer = CreateFramebufferObject(skyTexture);
            sunFramebuffer = CreateFramebufferObject(sunTexture);
            gBufferFramebuffer = CreateFramebufferObject(positionTexture, normalTexture, albedoTexture, holdInfoTexture, fogColorTexture, waterPositionTexture, waterNormalTexture);
            mainFramebuffer = CreateFramebufferObject(colorTexture, secondaryPositionTexture, reflectionTexture, secondaryAlbedoTexture, fogColorTexture, sssTexture, sssDenoiseTexture);
            temporalFramebuffer = CreateFramebufferObject(temporalWeightsTexture, temporalRadianceTexture, temporalPositionTexture);
            spatialFramebuffer = CreateFramebufferObject(spatialWeightsTexture, spatialRadianceTexture, fogPositionTexture);
            taaFramebuffer = CreateFramebufferObject(taaTexture);
            upscaleFramebuffer = CreateFramebufferObject(upscaleTexture);
            previousFrameFramebuffer = CreateFramebufferObject(prevNormalDepthTexture, prevAccumulationTexture, prevTaaTexture);
            previousTemporalFramebuffer = CreateFramebufferObject(prevTemporalWeightsTexture, prevTemporalRadianceTexture, prevTemporalPositionTexture, prevSpatialWeightsTexture, prevSpatialRadianceTexture);
            denoiseFramebufferA = CreateFramebufferObject(denoiseTextureB, varianceTextureB, fogColorTextureB, secondaryAlbedoTexture, sssDenoiseTexture);
            denoiseFramebufferB = CreateFramebufferObject(denoiseTextureA, varianceTextureA, fogColorTexture, secondaryAlbedoTexture, sssDenoiseTexture);
            tempAccumulationFramebuffer = CreateFramebufferObject(tempAccumulationTexture, denoiseTextureA, varianceTextureA);
            spatialPrepassFramebuffer = CreateFramebufferObject(prevTemporalWeightsTexture, prevTemporalRadianceTexture, prevTemporalPositionTexture);

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
            spatialPrepassShader = new Shader("spatPrePass.vert", "spatPrePass.frag");

            // Setup fullscreen quad
            fullscreenQuadVbo = GL.GenBuffer();
            fullscreenQuadVao = GL.GenVertexArray();
            fullscreenQuadEbo = GL.GenBuffer();


            initializationShader.Use();
            initializationShader.SetInt("skytex", skyTexture.Unit);

            skyShader.Use();
            skyShader.SetInt("skyprev", skyTexture.Unit);
            skyShader.SetInt("worl", worleyNoiseTexture.Unit);
            //skyShader.SetInt("suns", sunTexture.Unit);

            sunShader.Use();
            //sunShader.SetInt("sunprev", sunTexture.Unit);
            //sunShader.SetInt("worl", worleyNoiseTexture.Unit);

            radianceCacheShader.Use();
            radianceCacheShader.SetInt("skytex", skyTexture.Unit);
            radianceCacheShader.SetInt("suntex", sunTexture.Unit);

            geometryPassShader.Use();
            geometryPassShader.SetInt("position", positionTexture.Unit);
            geometryPassShader.SetInt("normal", normalTexture.Unit);
            geometryPassShader.SetInt("albedo", albedoTexture.Unit);
            geometryPassShader.SetInt("holdinfo", holdInfoTexture.Unit);
            //geometryPassShader.SetInt("colorfog", fogColorTexture.Unit);
            geometryPassShader.SetInt("skytex", skyTexture.Unit);
            geometryPassShader.SetInt("suntex", sunTexture.Unit);
            geometryPassShader.SetInt("watpos", waterPositionTexture.Unit);
            geometryPassShader.SetInt("watnorm", waterNormalTexture.Unit);
            geometryPassShader.SetInt("sssprev", sssTexture.Unit);
            geometryPassShader.SetInt("prevN", prevNormalDepthTexture.Unit);
            geometryPassShader.SetInt("den1", denoiseTextureB.Unit);

            temporalResamplingShader.Use();
            temporalResamplingShader.SetInt("color", colorTexture.Unit);
            temporalResamplingShader.SetInt("position", positionTexture.Unit);
            temporalResamplingShader.SetInt("normal", normalTexture.Unit);
            temporalResamplingShader.SetInt("secondpos", secondaryPositionTexture.Unit);
            temporalResamplingShader.SetInt("prevW", prevTemporalWeightsTexture.Unit);
            temporalResamplingShader.SetInt("prevL", prevTemporalRadianceTexture.Unit);
            temporalResamplingShader.SetInt("prevP", prevTemporalPositionTexture.Unit);
            temporalResamplingShader.SetInt("prevN", prevNormalDepthTexture.Unit);

            spatialPrepassShader.Use();
            spatialPrepassShader.SetInt("color", colorTexture.Unit);
            spatialPrepassShader.SetInt("position", positionTexture.Unit);
            spatialPrepassShader.SetInt("normal", normalTexture.Unit);
            spatialPrepassShader.SetInt("secondpos", secondaryPositionTexture.Unit);
            spatialPrepassShader.SetInt("prevW", temporalWeightsTexture.Unit);
            spatialPrepassShader.SetInt("prevL", temporalRadianceTexture.Unit);
            spatialPrepassShader.SetInt("prevP", temporalPositionTexture.Unit);
            spatialPrepassShader.SetInt("prevN", prevNormalDepthTexture.Unit);
            //spatialPrepassShader.SetInt("reflAlb", secondaryAlbedoTexture.Unit);

            spatialResamplingShader.Use();
            //spatialResamplingShader.SetInt("color", colorTexture.Unit);
            spatialResamplingShader.SetInt("position", positionTexture.Unit);
            spatialResamplingShader.SetInt("normal", normalTexture.Unit);
            spatialResamplingShader.SetInt("albedo", albedoTexture.Unit);
            //spatialResamplingShader.SetInt("secondpos", secondaryPositionTexture.Unit);
            spatialResamplingShader.SetInt("temppos", temporalPositionTexture.Unit);
            spatialResamplingShader.SetInt("prevW", prevSpatialWeightsTexture.Unit);
            spatialResamplingShader.SetInt("prevL", prevSpatialRadianceTexture.Unit);
            spatialResamplingShader.SetInt("tempW", temporalWeightsTexture.Unit);
            spatialResamplingShader.SetInt("tempL", temporalRadianceTexture.Unit);
            spatialResamplingShader.SetInt("prevN", prevNormalDepthTexture.Unit);
            spatialResamplingShader.SetInt("prevSPos", fogPositionTexture.Unit);
            spatialResamplingShader.SetInt("holdinfo", holdInfoTexture.Unit);

            tempAccumulationShader.Use();
            //tempAccumulationShader.SetInt("color", colorTexture.Unit);
            tempAccumulationShader.SetInt("position", positionTexture.Unit);
            tempAccumulationShader.SetInt("normal", normalTexture.Unit);
            tempAccumulationShader.SetInt("albedo", albedoTexture.Unit);
            tempAccumulationShader.SetInt("weightS", spatialWeightsTexture.Unit);
            tempAccumulationShader.SetInt("outgoingrS", spatialRadianceTexture.Unit);
            tempAccumulationShader.SetInt("prevN", prevNormalDepthTexture.Unit);
            tempAccumulationShader.SetInt("prevAcc", prevAccumulationTexture.Unit);
            tempAccumulationShader.SetInt("prevvar", varianceTextureB.Unit);
            tempAccumulationShader.SetInt("prevden", denoiseTextureB.Unit);

            denoiseShader.Use();
            denoiseShader.SetInt("color", temporalRadianceTexture.Unit);
            //denoiseShader.SetInt("position", positionTexture.Unit);
            denoiseShader.SetInt("normal", normalTexture.Unit);
            denoiseShader.SetInt("albedo", albedoTexture.Unit);
            denoiseShader.SetInt("secondpos", secondaryPositionTexture.Unit);
            denoiseShader.SetInt("reflN", reflectionTexture.Unit);
            denoiseShader.SetInt("reflectionAlb", secondaryAlbedoTexture.Unit);
            denoiseShader.SetInt("holdinfo", holdInfoTexture.Unit);
            denoiseShader.SetInt("watpos", waterPositionTexture.Unit);
            denoiseShader.SetInt("watnorm", waterNormalTexture.Unit);
            denoiseShader.SetInt("tempacc", tempAccumulationTexture.Unit);
            denoiseShader.SetInt("sstexden", sssDenoiseTexture.Unit);
            denoiseShader.SetInt("colorTex", colorTexture.Unit);
            // Ping-pong textures are set each frame, omitted here

            taaShader.Use();
            //taaShader.SetInt("color", colorTexture.Unit); //TODO: review. This was commented out.
            taaShader.SetInt("position", positionTexture.Unit);
            taaShader.SetInt("normal", normalTexture.Unit);
            taaShader.SetInt("albedo", albedoTexture.Unit);
            taaShader.SetInt("secondpos", secondaryPositionTexture.Unit);
            //taaShader.SetInt("Acc", tempAccumulationTexture.Unit);
            taaShader.SetInt("den1", denoiseTextureB.Unit); //den2 bound here
            taaShader.SetInt("prevTAA", prevTaaTexture.Unit);
            taaShader.SetInt("reflAlb", secondaryAlbedoTexture.Unit);
            taaShader.SetInt("colorfog", fogColorTexture.Unit);
            //taaShader.SetInt("holdinfo", holdInfoTexture.Unit);
            taaShader.SetInt("suntex", sunTexture.Unit);
            taaShader.SetInt("skyTex", skyTexture.Unit);
            taaShader.SetInt("watpos", waterPositionTexture.Unit);
            taaShader.SetInt("watnorm", waterNormalTexture.Unit);
            taaShader.SetInt("worl", worleyNoiseTexture.Unit);
            taaShader.SetInt("sssT", sssDenoiseTexture.Unit);
            taaShader.SetInt("sssT2", sssTexture.Unit);
            taaShader.SetInt("prevN", prevNormalDepthTexture.Unit);
            taaShader.SetInt("holdinfo", holdInfoTexture.Unit);
            taaShader.SetInt("color", colorTexture.Unit);
            taaShader.SetInt("var", varianceTextureA.Unit);

            upscaleShader.Use();
            upscaleShader.SetInt("TAA", taaTexture.Unit);

            previousFrameCopyShader.Use();
            previousFrameCopyShader.SetInt("normal", normalTexture.Unit);
            previousFrameCopyShader.SetInt("Acc", tempAccumulationTexture.Unit);
            previousFrameCopyShader.SetInt("TAA", taaTexture.Unit);

            previousTemporalCopyShader.Use();
            previousTemporalCopyShader.SetInt("tempWeights", temporalWeightsTexture.Unit);
            previousTemporalCopyShader.SetInt("tempOutL", temporalRadianceTexture.Unit);
            previousTemporalCopyShader.SetInt("tempPos", temporalPositionTexture.Unit);
            previousTemporalCopyShader.SetInt("spatWe", spatialWeightsTexture.Unit);
            previousTemporalCopyShader.SetInt("spatLO", spatialRadianceTexture.Unit);

            finalCompositeShader.Use();
            finalCompositeShader.SetInt("color", prevSpatialWeightsTexture.Unit);
            finalCompositeShader.SetInt("normal", normalTexture.Unit);
            finalCompositeShader.SetInt("albedo", albedoTexture.Unit);
            finalCompositeShader.SetInt("TAA", upscaleTexture.Unit); //TODO: review. "TAA" sampler is bound to "upscale" here.
            finalCompositeShader.SetInt("fogLO", taaTexture.Unit); //TODO: review. "fogLO" sampler is bound to "TAA" here.
            finalCompositeShader.SetInt("watpos", waterPositionTexture.Unit);
            finalCompositeShader.SetInt("position", positionTexture.Unit);
            finalCompositeShader.SetInt("var1", varianceTextureB.Unit);



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

            camera = new Camera(Vector3.UnitZ * 3, resolution.X / (float)resolution.Y);
            frameTimer = new Stopwatch();
            frameTimer.Start();

            // Setup compute textures and dispatch compute shaders as needed
            GL.BindImageTexture(0, worleyNoiseTexture.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            worleyNoiseShader.Use();
            GL.DispatchCompute(64, 64, 64);

            GL.BindImageTexture(0, radianceCachePositionTexture.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba32f);
            GL.BindImageTexture(1, radianceCacheNormalTexture.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba32f);
            GL.BindImageTexture(2, radianceCacheRadianceTexture.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            GL.BindImageTexture(3, radianceCacheFogTexture.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba32f);
            GL.BindImageTexture(4, sunTexture.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            
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

            RenderInitializationPass(viewMatrix, projectionMatrix);
            RenderSkyPass();
            RenderSunPass(lightViewMatrix, lightProjectionMatrix, LightDirection, lightPosition);
            RenderRadianceCachePass(lightViewMatrix, lightProjectionMatrix, LightDirection);
            RenderGeometryPass(viewMatrix, projectionMatrix, lightViewMatrix, lightProjectionMatrix, LightDirection);
            RenderTemporalResamplingPass();
            RenderSpatPrePass();
            RenderSpatialResamplingPass();
            RenderTempAccumulationPass();
            RenderDenoisePass();
            RenderTaaPass(viewMatrix, projectionMatrix, lightViewMatrix, lightProjectionMatrix, LightDirection);
            RenderUpscalePass();
            RenderPreviousFrameCopyPass();
            RenderPreviousTemporalCopyPass();
            RenderFinalCompositePass(viewMatrix, projectionMatrix);

            UpdatePreviousFrameData(viewMatrix, projectionMatrix);
            frameIndex += 1.0f;
        }


        public void UpdateResolution(Vector2i resolution)
        {
            camera.AspectRatio = resolution.X / (float)resolution.Y;
            GL.Viewport(0, 0, resolution.X, resolution.Y);
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

        private void RenderInitializationPass(Matrix4 viewMatrix, Matrix4 projectionMatrix)
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
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
            GL.MemoryBarrier(MemoryBarrierFlags.AllBarrierBits);
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
            GL.MemoryBarrier(MemoryBarrierFlags.AllBarrierBits);
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
            GL.GenerateTextureMipmap(taaTexture.Handle);
        }

        private void RenderUpscalePass()
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, upscaleFramebuffer);
            GL.Viewport(0, 0, (int)(renderResolution.X * DefaultScale), (int)(renderResolution.Y * DefaultScale));
            GL.Disable(EnableCap.DepthTest);
            upscaleShader.Use();
            upscaleShader.SetVector2(new Vector2(renderResolution.X * DefaultScale, renderResolution.Y * DefaultScale), "wh");
            upscaleShader.SetFloat(DefaultScale, "scale");
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
            GL.GenerateTextureMipmap(upscaleTexture.Handle);
        }

        private void RenderPreviousFrameCopyPass()
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, previousFrameFramebuffer);
            GL.Viewport(0, 0, (int)(renderResolution.X * renderScale), (int)(renderResolution.Y * renderScale));
            GL.Disable(EnableCap.DepthTest);
            previousFrameCopyShader.Use();
            GL.DrawElements(PrimitiveType.Triangles, fullscreenQuadIndices.Length, DrawElementsType.UnsignedInt, 0);
            GL.GenerateTextureMipmap(prevTaaTexture.Handle);
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
            GL.Viewport(0, 0, (int)(renderResolution.X * DefaultScale), (int)(renderResolution.Y * DefaultScale));
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