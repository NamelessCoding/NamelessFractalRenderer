using LearnOpenTK.Common;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace RealTimeFractalRendererLibrary
{
    public class Renderer
    {
        int quadVBO;
        int quadVAO;
        int quadEBO;

        float[] quad =
        {
            //Position          Texture coordinates
            1.0f,  1.0f, 0.0f, 1.0f, 1.0f, // top right
            1.0f, -1.0f, 0.0f, 1.0f, 0.0f, // bottom right
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // bottom left
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f  // top left
        };

        uint[] quadindices = {  // note that we start from 0!
            0, 1, 3,   // first triangle
            1, 2, 3    // second triangle
        };

        int colortex;
        int colortexHoldInfo;

        int colortexReflection;
        int colortexPosition;
        int colortexNormal;
        int colortexAlbedo;
        int colortexFog;
        int colortexFogPos;
        int colortexFogPrev;

        int colortexSecondPosition;
        int colortexSecondAlbedo;

        int watpos;
        int watnorm;

        int colorbuff2;
        private Shader _initShader;
        int colorfog;
        int colorfog2;

        int colorbuff;
        Stopwatch stopWatch;

        float iFrame = 0.0f;

        private Shader _testShader;
        private Shader _finalShader;

        private Shader _TemporalRestirShader;
        int temporalbuff;
        int temporalWeigths;
        int temporalOutgoingRadiance;
        int temporalWeigthsFog;
        int temporalOutgoingRadianceFog;

        int temporalPosition;
        int temporalPositionFog;

        private Shader _SpatialRestirShader;
        int spatialbuff;
        int spatialWeigths;
        int spatialOutgoingRadiance;

        int spatialWeigthsFog;
        int spatialOutgoingRadianceFog;

        private Shader _swapShader;
        int swapbuff;

        private Shader _swapShader2;
        int swapbuff2;

        int prevTemporalWeigths;
        int prevTemporalOutgoingRadiance;
        int prevTemporalPosition;
        int prevSpatialWeigths;
        int prevSpatialOutgoingRadiance;
        int prevNormalDepth;
        int prevPosition;
        int prevSecondPosition;

        int prevAcc;
        int prevTAA;

        private Shader _tempAccumShader;
        int tempAccumbuff;
        int tempAccum;

        private Shader _upscale;
        int upscale;
        int upscale2;

        int upscaleBuff;


        private Shader _TAAShader;
        int TAAbuff;
        int TAA;

        private Shader _denoiseShader;
        private Shader _denoiseShader2;
        int den1;
        int den2;
        int den1buff;
        int den2buff;
        int var1;
        int var2;

        int rcpos;
        int rcnorm;
        int rcrad;
        int rcfog;

        private Shader _skyShader;

        int skybuff;
        int skytex;

        private Shader _sunShader;

        int sunbuff;
        int suntex;


        private Shader _RC;

        int worley;

        private Shader _worl;



        private Camera _camera;
        public Camera Camera => _camera;

        /// <summary>
        /// The framebuffer where the final image is rendered to.
        /// </summary>
        public int TargetFramebuffer { get; set; } = 0;

        private Vector3 prevCamPos = new Vector3(0.0f, 0.0f, 0.0f);

        private bool enableDebugging = true;
        private bool isEverythingLower = true;

        static float ScaleEverything = 1.5f;
        static int shadowScale = 2048;
        static float KeepScale = ScaleEverything;

        private Vector2 size = new Vector2(1920.0f / ScaleEverything, 1080.0f / ScaleEverything);

        private Matrix4 prevView;
        private Matrix4 prevProjection;

        public Vector3 ldir = new Vector3(0.0f, 0.1f, -0.9f);

        public float bright = 1.0f;
        
        public void SetBrightness(float value)
        {
            //TODO: demo, use actual uniforms
            bright = value;
        }


        public void SetLdir(float X, float Y, float Z) {
            ldir.X = X; ldir.Y = Y; ldir.Z = Z;
        }


        //https://gist.github.com/Vassalware/d47ff5e60580caf2cbbf0f31aa20af5d
        private static void DebugCallback(DebugSource source,
            DebugType type,
            int id,
            DebugSeverity severity,
            int length,
            IntPtr message,
            IntPtr userParam)
        {
            string messageString = Marshal.PtrToStringAnsi(message, length);

            Console.WriteLine($"{severity} {type} | {messageString}");

            if (type == DebugType.DebugTypeError)
            {
                throw new Exception(messageString);
            }
        }

        private static readonly DebugProc _debugProcCallback = DebugCallback;
        private static GCHandle _debugProcCallbackHandle;


        float[] borderColor = { 1.0f, 1.0f, 1.0f, 1.0f };
        void setupTex2D(int texture, bool scaleTexture = true, PixelInternalFormat format = PixelInternalFormat.Rgba16f,
            TextureMinFilter minFilter = TextureMinFilter.Nearest, TextureMagFilter magFilter = TextureMagFilter.Nearest,
            TextureWrapMode wrapMode = TextureWrapMode.MirroredRepeat, bool customSize = false, int sizeX = 0, int sizeY = 0) {
            GL.BindTexture(TextureTarget.Texture2D, texture);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)minFilter);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)magFilter);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)wrapMode);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)wrapMode);
            if (scaleTexture && !customSize)
            {
                GL.TexImage2D(TextureTarget.Texture2D, 0,
                format,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);
            }
            else if (!customSize)
            {
                GL.TexImage2D(TextureTarget.Texture2D, 0,
                format,
                (int)(size.X), (int)(size.Y), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);
            }
            else {
                GL.TexImage2D(TextureTarget.Texture2D, 0,
                    format,
                    (int)(sizeX), (int)(sizeY), 0, PixelFormat.Rgba,
                    PixelType.UnsignedByte, IntPtr.Zero);
            }
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
        }

        void setupTex3D(int texture, int sizeX, int sizeY, int sizeZ, PixelInternalFormat format = PixelInternalFormat.Rgba16f,
            TextureMinFilter minFilter = TextureMinFilter.Nearest, TextureMagFilter magFilter = TextureMagFilter.Nearest,
            TextureWrapMode wrapMode = TextureWrapMode.Repeat) {
            GL.BindTexture(TextureTarget.Texture3D, texture);
            // GL.BindImageTexture(0, rcpos, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            // GL.BindImageTexture(0, rcpos, 0, );
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureMinFilter, (int)minFilter);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureMagFilter, (int)magFilter);

            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapS, (int)wrapMode);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapT, (int)wrapMode);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapR, (int)wrapMode);

            GL.TexImage3D(TextureTarget.Texture3D, 0,
                format,
                sizeX, sizeY, sizeZ, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.BindImageTexture(0, texture, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

        }

        public void Load(Vector2i resolution)
        {
            // glEnable(GL_TEXTURE_3D);

            if (enableDebugging)
            {
                _debugProcCallbackHandle = GCHandle.Alloc(_debugProcCallback);
                GL.DebugMessageCallback(_debugProcCallback, IntPtr.Zero);
                GL.Enable(EnableCap.DebugOutput);
                GL.Enable(EnableCap.DebugOutputSynchronous);
            }

           // GL.Enable(EnableCap.Texture3DExt);
            GL.ClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            colortex = GL.GenTexture();
            colortexPosition = GL.GenTexture();
            colortexNormal = GL.GenTexture();
            colortexAlbedo = GL.GenTexture();
            colortexSecondPosition = GL.GenTexture();
            colortexReflection = GL.GenTexture();
            colortexSecondAlbedo = GL.GenTexture();
            colortexHoldInfo = GL.GenTexture();
            colortexFog = GL.GenTexture();
            colortexFogPos = GL.GenTexture();
            colortexFogPrev = GL.GenTexture();
            colorfog = GL.GenTexture();
            colorfog2 = GL.GenTexture();
            watpos = GL.GenTexture();
            watnorm = GL.GenTexture();
            prevSecondPosition = GL.GenTexture();
            rcpos = GL.GenTexture();
            // rcpos = GL.GenTextures();
            rcnorm = GL.GenTexture();
            rcrad = GL.GenTexture();
            rcfog = GL.GenTexture();

            skytex = GL.GenTexture();
            suntex = GL.GenTexture();

            upscaleBuff = GL.GenFramebuffer();
            upscale = GL.GenTexture();
            upscale2 = GL.GenTexture();
            colorbuff = GL.GenFramebuffer();
            colorbuff2 = GL.GenFramebuffer();
            skybuff = GL.GenFramebuffer();
            sunbuff = GL.GenFramebuffer();


            worley = GL.GenTexture();

            if (isEverythingLower) {
                ScaleEverything = 1.0f;
            }

           
            setupTex2D(colorfog2);          
            setupTex2D(colorfog);
            setupTex2D(watnorm);
            setupTex2D(watpos);
            setupTex2D(skytex, true, PixelInternalFormat.Rgba16f, TextureMinFilter.LinearMipmapLinear, TextureMagFilter.Linear);
            setupTex2D(suntex, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear, TextureWrapMode.ClampToBorder,
                true, shadowScale, shadowScale);
            setupTex3D(worley, 512, 512, 512, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);
            setupTex3D(rcpos, 180, 180, 180, PixelInternalFormat.Rgba16f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            setupTex3D(rcfog, 180, 180, 180, PixelInternalFormat.Rgba16f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            setupTex3D(rcnorm, 180, 180, 180, PixelInternalFormat.Rgba16f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            setupTex3D(rcrad, 180, 180, 180, PixelInternalFormat.Rgba16f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            setupTex2D(colortex, false);
            setupTex2D(colortexPosition, true, PixelInternalFormat.Rgba32f);
            setupTex2D(colortexFog, false);
            setupTex2D(colortexFogPos, false);
            setupTex2D(colortexFogPrev);
            setupTex2D(colortexHoldInfo);
            setupTex2D(colortexNormal, true, PixelInternalFormat.Rgba32f);
            setupTex2D(colortexAlbedo);
            setupTex2D(colortexSecondPosition, true, PixelInternalFormat.Rgba32f);
            setupTex2D(colortexReflection, false);
            setupTex2D(colortexSecondAlbedo, false);

            //====================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, skybuff);

            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, skytex, 0
                );

            DrawBuffersEnum[] attx = { DrawBuffersEnum.ColorAttachment0 };
            GL.DrawBuffers(1, attx);

            //=======================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, sunbuff);

            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, suntex, 0
                );

            DrawBuffersEnum[] attxs = { DrawBuffersEnum.ColorAttachment0 };
            GL.DrawBuffers(1, attxs);
            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, colorbuff2);

            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, colortexPosition, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment1,
                TextureTarget.Texture2D, colortexNormal, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment2,
                TextureTarget.Texture2D, colortexAlbedo, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment3,
                TextureTarget.Texture2D, colortexHoldInfo, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment4,
                TextureTarget.Texture2D, colorfog, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment5,
                TextureTarget.Texture2D, watpos, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment6,
                TextureTarget.Texture2D, watnorm, 0
                );
            DrawBuffersEnum[] att = { DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1,
                DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3, DrawBuffersEnum.ColorAttachment4,
             DrawBuffersEnum.ColorAttachment5, DrawBuffersEnum.ColorAttachment6};
            GL.DrawBuffers(7, att);


            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, colorbuff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, colortex, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment1,
                TextureTarget.Texture2D, colortexSecondPosition, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment2,
                TextureTarget.Texture2D, colortexReflection, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment3,
                TextureTarget.Texture2D, colortexSecondAlbedo, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment4,
                TextureTarget.Texture2D, colorfog, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment5,
                TextureTarget.Texture2D, colortexFogPos, 0
                );
            DrawBuffersEnum[] atts = { DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1,
                DrawBuffersEnum.ColorAttachment2,  DrawBuffersEnum.ColorAttachment3
            ,  DrawBuffersEnum.ColorAttachment4,  DrawBuffersEnum.ColorAttachment5};
            GL.DrawBuffers(6, atts);


            temporalWeigths = GL.GenTexture();
            temporalOutgoingRadiance = GL.GenTexture();
            temporalPosition = GL.GenTexture();
            temporalPositionFog = GL.GenTexture();
            temporalWeigthsFog = GL.GenTexture();
            temporalOutgoingRadianceFog = GL.GenTexture();
            temporalbuff = GL.GenFramebuffer();
            prevPosition = GL.GenTexture();

            
            setupTex2D(temporalWeigths, false);
            setupTex2D(temporalWeigthsFog, false);
            setupTex2D(temporalOutgoingRadianceFog, false);

            setupTex2D(prevPosition, true, PixelInternalFormat.Rgba32f);
            setupTex2D(prevSecondPosition, true, PixelInternalFormat.Rgba32f);
            setupTex2D(temporalOutgoingRadiance, false);
            setupTex2D(temporalPosition, false, PixelInternalFormat.Rgba32f);
            setupTex2D(temporalPositionFog, false);


            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, temporalbuff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, temporalWeigths, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment1,
                TextureTarget.Texture2D, temporalOutgoingRadiance, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment2,
                TextureTarget.Texture2D, temporalPosition, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment3,
                TextureTarget.Texture2D, temporalWeigthsFog, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment4,
                TextureTarget.Texture2D, temporalOutgoingRadianceFog, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment5,
                TextureTarget.Texture2D, temporalPositionFog, 0
                );
            DrawBuffersEnum[] att2 = { DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2
            , DrawBuffersEnum.ColorAttachment3, DrawBuffersEnum.ColorAttachment4, DrawBuffersEnum.ColorAttachment5};
            GL.DrawBuffers(6, att2);

            //private Shader _swapShader;



            spatialWeigths = GL.GenTexture();
            spatialOutgoingRadiance = GL.GenTexture();

            spatialWeigthsFog = GL.GenTexture();
            spatialOutgoingRadianceFog = GL.GenTexture();
            spatialbuff = GL.GenFramebuffer();

            
            setupTex2D(spatialWeigths, false);
            setupTex2D(spatialWeigthsFog, false);
            setupTex2D(spatialOutgoingRadianceFog, false);
            setupTex2D(spatialOutgoingRadiance, false);


            GL.BindFramebuffer(FramebufferTarget.Framebuffer, spatialbuff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, spatialWeigths, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment1,
                TextureTarget.Texture2D, spatialOutgoingRadiance, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment2,
                TextureTarget.Texture2D, spatialWeigthsFog, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment3,
                TextureTarget.Texture2D, spatialOutgoingRadianceFog, 0
                );
            DrawBuffersEnum[] att4 = { DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3 };
            GL.DrawBuffers(4, att4);

   
            TAA = GL.GenTexture();
            TAAbuff = GL.GenFramebuffer();

            
            setupTex2D(TAA, true, PixelInternalFormat.Rgba16f, TextureMinFilter.LinearMipmapLinear, TextureMagFilter.Linear);

            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, TAAbuff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, TAA, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment1,
                TextureTarget.Texture2D, prevPosition, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment2,
               TextureTarget.Texture2D, prevSecondPosition, 0
               );
            //colortexFogPrev
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment3,
               TextureTarget.Texture2D, colortexFogPrev, 0
               );
            DrawBuffersEnum[] att10 = { DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3 };
            GL.DrawBuffers(4, att10);

 
            setupTex2D(upscale, true, PixelInternalFormat.Rgba16f, TextureMinFilter.NearestMipmapLinear, TextureMagFilter.Nearest);
            setupTex2D(upscale2, true, PixelInternalFormat.Rgba16f, TextureMinFilter.NearestMipmapLinear, TextureMagFilter.Nearest);

            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, upscaleBuff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, upscale, 0
                );
            DrawBuffersEnum[] att11 = { DrawBuffersEnum.ColorAttachment0 };
            GL.DrawBuffers(1, att11);

            prevTemporalWeigths = GL.GenTexture();
            prevTemporalOutgoingRadiance = GL.GenTexture();
            prevTemporalPosition = GL.GenTexture();
            prevSpatialWeigths = GL.GenTexture();
            prevSpatialOutgoingRadiance = GL.GenTexture();
            prevNormalDepth = GL.GenTexture();
            prevAcc = GL.GenTexture();
            prevTAA = GL.GenTexture();
            swapbuff = GL.GenFramebuffer();
            swapbuff2 = GL.GenFramebuffer();
           
            setupTex2D(prevTemporalWeigths, false);
            
            setupTex2D(prevTemporalOutgoingRadiance, false);

           
            setupTex2D(prevTemporalPosition, false);


            setupTex2D(prevSpatialWeigths, false);

    
            setupTex2D(prevSpatialOutgoingRadiance, false);

            setupTex2D(prevNormalDepth, true, PixelInternalFormat.Rgba32f);


           
            setupTex2D(prevAcc, true);


            setupTex2D(prevTAA, true, PixelInternalFormat.Rgba16f, TextureMinFilter.LinearMipmapLinear, TextureMagFilter.Linear);

  
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, swapbuff);

            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, prevNormalDepth, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment1,
               TextureTarget.Texture2D, prevAcc, 0
               );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment2,
               TextureTarget.Texture2D, prevTAA, 0
               );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment3,
               TextureTarget.Texture2D, upscale2, 0
               );
            DrawBuffersEnum[] att3 = { DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3 };
            GL.DrawBuffers(4, att3);




            GL.BindFramebuffer(FramebufferTarget.Framebuffer, swapbuff2);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, prevTemporalWeigths, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment1,
                TextureTarget.Texture2D, prevTemporalOutgoingRadiance, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment2,
                TextureTarget.Texture2D, prevTemporalPosition, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment3,
                TextureTarget.Texture2D, prevSpatialWeigths, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment4,
                TextureTarget.Texture2D, prevSpatialOutgoingRadiance, 0
                );

            DrawBuffersEnum[] att3s = { DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2,
                DrawBuffersEnum.ColorAttachment3, DrawBuffersEnum.ColorAttachment4};
            GL.DrawBuffers(5, att3s);


            //private Shader _tempAccumShader;
            //int tempAccumbuff;
            //int tempAccum;

            //===============================================================
            den1 = GL.GenTexture();
            den2 = GL.GenTexture();
            den1buff = GL.GenFramebuffer();
            den2buff = GL.GenFramebuffer();
            var1 = GL.GenTexture();
            var2 = GL.GenTexture();
       
            setupTex2D(den1, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);
            setupTex2D(den2, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);

            setupTex2D(var1, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);

            setupTex2D(var2, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);


            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, den1buff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, den2, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment1,
                TextureTarget.Texture2D, var2, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment2,
               TextureTarget.Texture2D, colorfog2, 0
               );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment3,
               TextureTarget.Texture2D, colortexSecondAlbedo, 0
               );
            DrawBuffersEnum[] att8 = { DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3 };
            GL.DrawBuffers(4, att8);



            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, den2buff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, den1, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment1,
               TextureTarget.Texture2D, var1, 0
               );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
              FramebufferAttachment.ColorAttachment2,
              TextureTarget.Texture2D, colorfog, 0
              );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment3,
               TextureTarget.Texture2D, colortexSecondAlbedo, 0
               );
            DrawBuffersEnum[] att9 = { DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3 };
            GL.DrawBuffers(4, att9);



            //==========================================================-=
            tempAccum = GL.GenTexture();
            tempAccumbuff = GL.GenFramebuffer();
         
            setupTex2D(tempAccum);

            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, tempAccumbuff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, tempAccum, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment1,
                TextureTarget.Texture2D, den1, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment2,
                TextureTarget.Texture2D, var1, 0
                );
            DrawBuffersEnum[] att5 = { DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2 };
            GL.DrawBuffers(3, att5);



            _skyShader = new Shader("Shaders/sky.vert", "Shaders/sky.frag");
            _sunShader = new Shader("Shaders/sun.vert", "Shaders/sun.frag", true);

            _testShader = new Shader("Shaders/Test.vert", "Shaders/Test.frag", true);
            //_initShader
            _initShader = new Shader("Shaders/initial.vert", "Shaders/initial.frag", true);
            _finalShader = new Shader("Shaders/final.vert", "Shaders/final.frag", true);
            _TemporalRestirShader = new Shader("Shaders/temporal.vert", "Shaders/temporal.frag");
            _swapShader = new Shader("Shaders/swap.vert", "Shaders/swap.frag");
            _swapShader2 = new Shader("Shaders/swap2.vert", "Shaders/swap2.frag");

            _SpatialRestirShader = new Shader("Shaders/spatial.vert", "Shaders/spatial.frag", true);
            _tempAccumShader = new Shader("Shaders/tempAccum.vert", "Shaders/tempAccum.frag");
            _denoiseShader = new Shader("Shaders/denoise1.vert", "Shaders/denoise1.frag");
            _TAAShader = new Shader("Shaders/TAA.vert", "Shaders/TAA.frag");
            _RC = new Shader("Shaders/RadianceCaching.comp", true);

            _worl = new Shader("Shaders/worley.comp", false);


            _upscale = new Shader("Shaders/upscale.vert", "Shaders/upscale.frag");

            /*
              private Shader _upscale;
    int upscale;
    int upscaleBuff;
             */
            /*
                private Shader _denoiseShader;
                private Shader _denoiseShader2;
            */
            //tempAccum
            quadVBO = GL.GenBuffer();
            quadVAO = GL.GenVertexArray();
            quadEBO = GL.GenBuffer();

            _initShader.Use();
            _initShader.SetInt("skytex", 31);

            _skyShader.Use();
            _skyShader.SetInt("skyprev", 31);
            _skyShader.SetInt("worl", 13);//hack
            _skyShader.SetInt("suns", 18);

            _sunShader.Use();
            //_sunShader.SetInt("sunprev", 18);//TODO: review. this was commented out
            _sunShader.SetInt("worl", 13);

            _RC.Use();
            _RC.SetInt("skytex", 31);
            _RC.SetInt("suntex", 18);

            _testShader.Use();
            _testShader.SetInt("position", 2);
            _testShader.SetInt("normal", 3);
            _testShader.SetInt("albedo", 4);
            _testShader.SetInt("holdinfo", 26);
            _testShader.SetInt("colorfog", 27);
            _testShader.SetInt("skytex", 31);
            _testShader.SetInt("sunTex", 18);
            _testShader.SetInt("watpos", 19);
            _testShader.SetInt("watnorm", 0);

            _TemporalRestirShader.Use();
            _TemporalRestirShader.SetInt("color", 1);
            _TemporalRestirShader.SetInt("position", 2);
            _TemporalRestirShader.SetInt("normal", 3);
            _TemporalRestirShader.SetInt("secondpos", 5);
            _TemporalRestirShader.SetInt("prevW", 8);//hack
            _TemporalRestirShader.SetInt("prevL", 9);//hack
            _TemporalRestirShader.SetInt("prevP", 10);//hack
            _TemporalRestirShader.SetInt("prevN", 22);
            _TemporalRestirShader.SetInt("reflAlb", 25);
            _TemporalRestirShader.SetInt("prevPosition", 12);//hack
            _TemporalRestirShader.SetInt("prevSecondPosition", 13);//hack

            _SpatialRestirShader.Use();
            _SpatialRestirShader.SetInt("color", 1);
            _SpatialRestirShader.SetInt("position", 2);
            _SpatialRestirShader.SetInt("normal", 3);
            _SpatialRestirShader.SetInt("albedo", 4);
            _SpatialRestirShader.SetInt("secondpos", 5);
            _SpatialRestirShader.SetInt("temppos", 20);
            _SpatialRestirShader.SetInt("prevW", 8);//hack
            _SpatialRestirShader.SetInt("prevL", 9);//hack
            _SpatialRestirShader.SetInt("tempW", 6);
            _SpatialRestirShader.SetInt("tempL", 7);
            _SpatialRestirShader.SetInt("prevN", 22);
            _SpatialRestirShader.SetInt("reflAlb", 25);
            _SpatialRestirShader.SetInt("prevPosition", 12);//hack
            _SpatialRestirShader.SetInt("prevSecondPosition", 13);//hack
            //TODO: review, these are unused in the shader so I didnt bother
            //_SpatialRestirShader.SetInt("tempfog; //temporalWeigthsFog
            //_SpatialRestirShader.SetInt("tempLofog; //temporalOutgoingRadianceFog
            //_SpatialRestirShader.SetInt("temppos2; //temporalPositionFog
            //_SpatialRestirShader.SetInt("wightfog", 16);
            //_SpatialRestirShader.SetInt("Lofog", 30);
            //_SpatialRestirShader.SetInt("fosecg; //colortexFogPos

            _tempAccumShader.Use();
            _tempAccumShader.SetInt("color", 1);
            _tempAccumShader.SetInt("position", 2);
            _tempAccumShader.SetInt("normal", 3);
            _tempAccumShader.SetInt("albedo", 4);
            _tempAccumShader.SetInt("secondpos", 5);
            _tempAccumShader.SetInt("weigth", 6);
            _tempAccumShader.SetInt("outgoingr", 7);
            _tempAccumShader.SetInt("weightS", 8);
            _tempAccumShader.SetInt("outgoingrS", 9);
            _tempAccumShader.SetInt("prevN", 22);
            _tempAccumShader.SetInt("prevAcc", 0);//hack
            _tempAccumShader.SetInt("reflAlb", 25);
            _tempAccumShader.SetInt("prevPosition", 10);//hack
            _tempAccumShader.SetInt("prevSecondPosition", 12);//hack
            _tempAccumShader.SetInt("prevvar", 23);
            _tempAccumShader.SetInt("prevden", 11);

            _denoiseShader.Use();
            _denoiseShader.SetInt("color", 1);
            _denoiseShader.SetInt("position", 2);
            _denoiseShader.SetInt("normal", 3);
            _denoiseShader.SetInt("albedo", 4);
            _denoiseShader.SetInt("secondpos", 5);
            _denoiseShader.SetInt("weigth", 6);
            _denoiseShader.SetInt("outgoingr", 7);
            _denoiseShader.SetInt("weightS", 8);
            _denoiseShader.SetInt("outgoingrS", 9);
            _denoiseShader.SetInt("prevN", 22);
            _denoiseShader.SetInt("ACC", 10);
            _denoiseShader.SetInt("reflN", 28);
            _denoiseShader.SetInt("reflectionAlb", 25);
            _denoiseShader.SetInt("holdinfo", 26);
            _denoiseShader.SetInt("watpos", 19);
            _denoiseShader.SetInt("watnorm", 0);
            //_denoiseShader - some ping-pong textures are set each frame so they're omitted here

            _TAAShader.Use();
            //_TAAShader.SetInt("color", 1); //TODO: review. This was commented out.
            _TAAShader.SetInt("position", 2);
            _TAAShader.SetInt("normal", 3);
            _TAAShader.SetInt("albedo", 4);
            _TAAShader.SetInt("secondpos", 5);
            _TAAShader.SetInt("weigth", 6);
            _TAAShader.SetInt("outgoingr", 7);
            _TAAShader.SetInt("weightS", 8);
            _TAAShader.SetInt("outgoingrS", 9);
            _TAAShader.SetInt("Acc", 10);
            _TAAShader.SetInt("den1", 11);//den2 bound here
            _TAAShader.SetInt("var1", 23);//var2 bound here
            _TAAShader.SetInt("prevTAA", 24);
            _TAAShader.SetInt("reflAlb", 25);
            _TAAShader.SetInt("inf", 26);//holdinfo
            _TAAShader.SetInt("colorfog", 27);
            _TAAShader.SetInt("reflnorm", 28);
            //colortexFog
            _TAAShader.SetInt("colortexFog", 29);
            _TAAShader.SetInt("colortexFogP", 15);
            _TAAShader.SetInt("spatfog", 16);
            _TAAShader.SetInt("spatfogLO", 30);
            _TAAShader.SetInt("holdinfo", 26);//TODO: this 'holdinfo' is the same as 'inf'
            _TAAShader.SetInt("sunTex", 18);
            _TAAShader.SetInt("skyTex", 31);
            _TAAShader.SetInt("watpos", 19);
            _TAAShader.SetInt("watnorm", 0);
            _TAAShader.SetInt("worl", 13);//hack

            //uniform sampler2D TAA;
            //uniform sampler2D upbefore;
            _upscale.Use();
            _upscale.SetInt("TAA", 17);
            _upscale.SetInt("upbefore", !isEverythingLower ? 21 : 13);
            _upscale.SetInt("position", 2);
            _upscale.SetInt("normal", 3);
            _upscale.SetInt("prevN", 22);
            _upscale.SetInt("albedo", 4);

            _swapShader.Use();
            _swapShader.SetInt("tempWeights", 6);
            _swapShader.SetInt("tempOutL", 7);
            _swapShader.SetInt("tempPos", 20);
            _swapShader.SetInt("spatWe", 8);
            _swapShader.SetInt("spatLO", 9);
            _swapShader.SetInt("position", 2);
            _swapShader.SetInt("normal", 3);
            _swapShader.SetInt("Acc", 10);
            _swapShader.SetInt("TAA", 17);
            _swapShader.SetInt("up", 13);

            _swapShader2.Use();
            _swapShader2.SetInt("tempWeights", 6);
            _swapShader2.SetInt("tempOutL", 7);
            _swapShader2.SetInt("tempPos", 20);
            _swapShader2.SetInt("spatWe", 8);
            _swapShader2.SetInt("spatLO", 9);
            _swapShader2.SetInt("position", 2);
            _swapShader2.SetInt("normal", 3);
            _swapShader2.SetInt("Acc", 10);
            _swapShader2.SetInt("TAA", 17);

            _finalShader.Use();
            _finalShader.SetInt("color", 1);
            _finalShader.SetInt("position", 2);
            _finalShader.SetInt("normal", 3);
            _finalShader.SetInt("albedo", 4);
            _finalShader.SetInt("secondpos", 5);
            _finalShader.SetInt("weigth", 6);
            _finalShader.SetInt("outgoingr", 7);
            _finalShader.SetInt("weightS", 8);
            _finalShader.SetInt("outgoingrS", 9);
            _finalShader.SetInt("Acc", 10);
            _finalShader.SetInt("den1", 11); //TODO: review. "den1" sampler is bound to "den2" here.
            _finalShader.SetInt("var1", 12);
            _finalShader.SetInt("TAA", 13); //TODO: review. "TAA" sampler is bound to "upscale" here.
            _finalShader.SetInt("colorfog", 14); //colorfog2
            _finalShader.SetInt("colortexFog", 15); //colortexFogPrev
            _finalShader.SetInt("fogw", 16); //spatialWeigthsFog
            _finalShader.SetInt("fogLO", 17); //TAA
            _finalShader.SetInt("sunTex", 18);
            _finalShader.SetInt("watpos", 19);

            GL.BindVertexArray(quadVAO);
            //Seclect the VertexBufferObject
            GL.BindBuffer(BufferTarget.ArrayBuffer, quadVBO);
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, quadEBO);

            //Using the selected vertex buffer object, write to it the vertices data using static draw(meaning that data won't change)
            GL.BufferData(BufferTarget.ArrayBuffer, quad.Length * sizeof(float), quad, BufferUsageHint.StaticDraw);
            GL.BufferData(BufferTarget.ElementArrayBuffer, quadindices.Length * sizeof(uint), quadindices, BufferUsageHint.StaticDraw);

            GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 3 * sizeof(float), 0);
            int vertexPosition = GL.GetAttribLocation(_testShader.Handle, "aPosition");
            GL.EnableVertexAttribArray(vertexPosition);
            GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 5 * sizeof(float), 0);
            GL.EnableVertexAttribArray(0);
            int texCoordLocation2 = GL.GetAttribLocation(_testShader.Handle, "aTexCoord");
            GL.VertexAttribPointer(texCoordLocation2, 2, VertexAttribPointerType.Float, false, 5 * sizeof(float), 3 * sizeof(float));
            GL.EnableVertexAttribArray(texCoordLocation2);


            _camera = new Camera(Vector3.UnitZ * 3, resolution.X / (float)resolution.Y);
            stopWatch = new Stopwatch();
            stopWatch.Start();

            GL.BindImageTexture(0, worley, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            _worl.Use();
            GL.DispatchCompute(64, 64, 64);

            //CursorState = CursorState.Grabbed;

            SetupTextureBindings();

        }


        public void RenderFrame()
        {
            //GL.Clear(ClearBufferMask.ColorBufferBit);

            ldir = Vector3.Normalize(ldir);

            Vector3 lpos = new Vector3( _camera.Position.X + ldir.X * 270.0f, ldir.Y * 270.0f, _camera.Position.Z + ldir.Z * 270.0f );

            Matrix4 lightview = Matrix4.LookAt(
                lpos,
                new Vector3(_camera.Position.X, 0.0f, _camera.Position.Z),
                new Vector3(0.0f, 1.0f, 0.0f)
                );

            //Matrix4 lightview = Matrix4.

            //Matrix4 lightproj = Matrix4.CreateOrthographicOffCenter(
            //    -60.0f, 60.0f, -60.0f, 60.0f, 1.0f, 100.0f
            //    );
            Matrix4 lightproj = Matrix4.CreateOrthographic(200.0f, 200.0f, 0.0f, 127.5f);

            Vector3 nm = Vector3.Normalize(lpos - new Vector3(_camera.Position.X, 0.0f, _camera.Position.Z));

             Vector3 lpos2 = new Vector3(_camera.Position.X + ldir.X * 360.0f, ldir.Y * 360.0f, _camera.Position.Z + ldir.Z * 360.0f);
            //lpos2 = lpos;
            //Vector3 lpos2 = ldir * 8200.0f;
           // Vector3 lpos2 = new Vector3(_camera.Position.X, 0.0f, _camera.Position.Z) + nm * 8200.0f;
            //lpos2.Y = 8200.0f;
           // Vector3 lpos2 = new Vector3(0.0f, 7200.0f, 0.0f);
            Matrix4 lightview2 = Matrix4.LookAt(
                lpos2,
                new Vector3(_camera.Position.X, 0.0f, _camera.Position.Z),
                new Vector3(0.0f, 1.0f, 0.0f)
                );
            Matrix4 lightproj2 = Matrix4.CreateOrthographic(400.0f, 400.0f, 0.0f, 800.0f);

            //Matrix4 lightproj2 = Matrix4.CreatePerspectiveFieldOfView(3.14159f / 1.2f, 1.0f, 0.01f, 120.0f);

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, colorbuff2);

            Matrix4 view = _camera.GetViewMatrix();
            Matrix4 projection = _camera.GetProjectionMatrix();
            GL.Viewport(0, 0, (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything));
            //GL.ClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            //GL.Clear(ClearBufferMask.ColorBufferBit);
            GL.Disable(EnableCap.DepthTest);
            _initShader.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _initShader.setMatrix4(view, "view");
            _initShader.setMatrix4(projection, "projection");
            _initShader.setMatrix4(view.Inverted(), "invview");
            _initShader.setMatrix4(projection.Inverted(), "invproj");
            _initShader.setVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _initShader.setFloat(iFrame, "time");
            _initShader.setFloat(stopWatch.ElapsedMilliseconds, "time2");
            _initShader.setVector3(_camera.Position, "viewPos");
            _initShader.setVector3(prevCamPos, "lastViewPos");
            _initShader.setVector3(ldir, "ldir");

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);
            GL.MemoryBarrier(MemoryBarrierFlags.AllBarrierBits);

            //==============================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, skybuff);

            GL.Viewport(0, 0, (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything));
            //GL.ClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            //GL.Clear(ClearBufferMask.ColorBufferBit);
            GL.Disable(EnableCap.DepthTest);
            _skyShader.Use();
            _skyShader.setVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _skyShader.setFloat(iFrame, "time");
            _skyShader.setVector3(ldir, "ldir");
            _skyShader.setVector3(_camera.Position, "viewPos");
            _skyShader.setVector3(lpos2, "lpos2");
            _skyShader.setMatrix4(lightproj2, "lightproj2");
            _skyShader.setMatrix4(lightview2, "lightview2");

            GL.BindVertexArray(quadVAO);

            GL.ActiveTexture(TextureUnit.Texture13);//hack
            GL.BindTexture(TextureTarget.Texture3D, worley);

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            GL.BindTexture(TextureTarget.Texture2D, skytex);
            GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);

            GL.ActiveTexture(TextureUnit.Texture13);//restore hack
            GL.BindTexture(TextureTarget.Texture2D, upscale);

            //=================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, sunbuff);

            GL.Viewport(0, 0, (int)shadowScale, (int)shadowScale);
            //GL.ClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            //GL.Clear(ClearBufferMask.ColorBufferBit);
            GL.Disable(EnableCap.DepthTest);
            _sunShader.Use();
            _sunShader.setVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _sunShader.setFloat(iFrame, "time");
            _sunShader.setVector3(ldir, "ldir");
            _sunShader.setVector3(lpos, "lpos");
            _sunShader.setMatrix4(lightproj, "lightproj");
            _sunShader.setMatrix4(lightview, "lightview");
            _sunShader.setMatrix4(lightview.Inverted(), "invview");
            _sunShader.setMatrix4(lightproj.Inverted(), "invproj");
            _sunShader.setFloat((float)shadowScale, "scale");
            _sunShader.setVector3(_camera.Position, "viewPos");
            _sunShader.setMatrix4(lightview2.Inverted(), "invview2");
            _sunShader.setMatrix4(lightproj2.Inverted(), "invproj2");
            _sunShader.setVector3(lpos2, "lpos2");
            _sunShader.setMatrix4(lightproj2, "lightproj2");
            _sunShader.setMatrix4(lightview2, "lightview2");
            GL.BindVertexArray(quadVAO);

            GL.ActiveTexture(TextureUnit.Texture13);//hack
            GL.BindTexture(TextureTarget.Texture3D, worley);

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            
            GL.ActiveTexture(TextureUnit.Texture13);//restore hack
            GL.BindTexture(TextureTarget.Texture2D, upscale);

            //===================================================

            _RC.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _RC.setMatrix4(view, "view");
            _RC.setMatrix4(projection, "projection");
            _RC.setMatrix4(view.Inverted(), "invview");
            _RC.setMatrix4(projection.Inverted(), "invproj");
            _RC.setVector2(new Vector2((float)size.X * KeepScale, (float)size.Y* KeepScale), "wh");
            _RC.setFloat(iFrame, "time");
            _RC.setFloat(stopWatch.ElapsedMilliseconds, "time2");
            _RC.setVector3(_camera.Position, "viewPos");
            _RC.setVector3(prevCamPos, "lastViewPos");
            _RC.setVector3(ldir, "ldir");
            _RC.setMatrix4(lightproj, "lightproj");
            _RC.setMatrix4(lightview, "lightview");
            _RC.setVector3(lpos, "lpos");

            GL.DispatchCompute(45, 45, 45);

            GL.MemoryBarrier(MemoryBarrierFlags.AllBarrierBits);

            //===================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, colorbuff);


            GL.Viewport(0, 0, (int)size.X, (int)size.Y);
            //GL.ClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            //GL.Clear(ClearBufferMask.ColorBufferBit);
            GL.Disable(EnableCap.DepthTest);
            _testShader.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _testShader.setMatrix4(view, "view");
            _testShader.setMatrix4(projection, "projection");
            _testShader.setMatrix4(view.Inverted(), "invview");
            _testShader.setMatrix4(projection.Inverted(), "invproj");
            _testShader.setVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _testShader.setFloat(iFrame, "time");
            _testShader.setFloat(stopWatch.ElapsedMilliseconds, "time2");
            _testShader.setVector3(_camera.Position, "viewPos");
            _testShader.setVector3(ldir, "ldir");
            _testShader.setMatrix4(lightproj, "lightproj");
            _testShader.setMatrix4(lightview, "lightview");
            _testShader.setVector3(lpos, "lpos");

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);


            ///==========================================================
            
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, temporalbuff);

            GL.Viewport(0, 0, (int)size.X, (int)size.Y);
            GL.Disable(EnableCap.DepthTest);
            _TemporalRestirShader.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _TemporalRestirShader.setMatrix4(view, "view");
            _TemporalRestirShader.setMatrix4(projection, "projection");
            _TemporalRestirShader.setMatrix4(view.Inverted(), "invview");
            _TemporalRestirShader.setMatrix4(projection.Inverted(), "invproj");
            _TemporalRestirShader.setMatrix4(prevView, "prevview");
            _TemporalRestirShader.setMatrix4(prevProjection, "prevproj");

            _TemporalRestirShader.setVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _TemporalRestirShader.setFloat(iFrame, "time");
            _TemporalRestirShader.setVector3(_camera.Position, "viewPos");
            _TemporalRestirShader.setVector3(prevCamPos, "lastViewPos");
            _TemporalRestirShader.setVector3(ldir, "ldir");
            
            GL.ActiveTexture(TextureUnit.Texture8);//hack
            GL.BindTexture(TextureTarget.Texture2D, prevTemporalWeigths);
            GL.ActiveTexture(TextureUnit.Texture9);//hack
            GL.BindTexture(TextureTarget.Texture2D, prevTemporalOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture10);//hack
            GL.BindTexture(TextureTarget.Texture2D, prevTemporalPosition);
            GL.ActiveTexture(TextureUnit.Texture12);//hack
            GL.BindTexture(TextureTarget.Texture2D, prevPosition);
            GL.ActiveTexture(TextureUnit.Texture13);//hack
            GL.BindTexture(TextureTarget.Texture2D, prevSecondPosition);

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            //restore hack
            GL.ActiveTexture(TextureUnit.Texture8);//hack
            GL.BindTexture(TextureTarget.Texture2D, spatialWeigths);
            GL.ActiveTexture(TextureUnit.Texture9);//hack
            GL.BindTexture(TextureTarget.Texture2D, spatialOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture10);//hack
            GL.BindTexture(TextureTarget.Texture2D, tempAccum);
            GL.ActiveTexture(TextureUnit.Texture12);//hack
            GL.BindTexture(TextureTarget.Texture2D, var1);
            GL.ActiveTexture(TextureUnit.Texture13);//hack
            GL.BindTexture(TextureTarget.Texture2D, upscale);

            ///==========================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, spatialbuff);


            GL.Viewport(0, 0, (int)size.X, (int)size.Y);
            GL.Disable(EnableCap.DepthTest);
            _SpatialRestirShader.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _SpatialRestirShader.setMatrix4(view, "view");
            _SpatialRestirShader.setMatrix4(projection, "projection");
            _SpatialRestirShader.setMatrix4(view.Inverted(), "invview");
            _SpatialRestirShader.setMatrix4(projection.Inverted(), "invproj");
            _SpatialRestirShader.setMatrix4(prevView, "prevview");
            _SpatialRestirShader.setMatrix4(prevProjection, "prevproj");
            _SpatialRestirShader.setVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _SpatialRestirShader.setFloat(iFrame, "time");
            _SpatialRestirShader.setFloat(stopWatch.ElapsedMilliseconds, "time2");
            _SpatialRestirShader.setVector3(_camera.Position, "viewPos");
            _SpatialRestirShader.setVector3(prevCamPos, "lastViewPos");
            _SpatialRestirShader.setVector3(ldir, "ldir");

            GL.ActiveTexture(TextureUnit.Texture8);//hack
            GL.BindTexture(TextureTarget.Texture2D, prevSpatialWeigths);
            GL.ActiveTexture(TextureUnit.Texture9);//hack
            GL.BindTexture(TextureTarget.Texture2D, prevSpatialOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture12);//hack
            GL.BindTexture(TextureTarget.Texture2D, prevPosition);
            GL.ActiveTexture(TextureUnit.Texture13);//hack
            GL.BindTexture(TextureTarget.Texture2D, prevSecondPosition);

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            //restore hack
            GL.ActiveTexture(TextureUnit.Texture8);//hack
            GL.BindTexture(TextureTarget.Texture2D, spatialWeigths);
            GL.ActiveTexture(TextureUnit.Texture9);//hack
            GL.BindTexture(TextureTarget.Texture2D, spatialOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture12);//hack
            GL.BindTexture(TextureTarget.Texture2D, var1);
            GL.ActiveTexture(TextureUnit.Texture13);//hack
            GL.BindTexture(TextureTarget.Texture2D, upscale);

            ///==========================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, tempAccumbuff);

            GL.Viewport(0, 0, (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything));
            GL.Disable(EnableCap.DepthTest);
            _tempAccumShader.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _tempAccumShader.setMatrix4(view, "view");
            _tempAccumShader.setMatrix4(projection, "projection");
            _tempAccumShader.setMatrix4(view.Inverted(), "invview");
            _tempAccumShader.setMatrix4(projection.Inverted(), "invproj");
            _tempAccumShader.setMatrix4(prevView, "prevview");
            _tempAccumShader.setMatrix4(prevProjection, "prevproj");
            _tempAccumShader.setVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _tempAccumShader.setFloat(iFrame, "time");
            _tempAccumShader.setVector3(_camera.Position, "viewPos");
            _tempAccumShader.setVector3(prevCamPos, "lastViewPos");
            _tempAccumShader.setVector3(ldir, "ldir");

            GL.ActiveTexture(TextureUnit.Texture0);//hack
            GL.BindTexture(TextureTarget.Texture2D, prevAcc);
            GL.ActiveTexture(TextureUnit.Texture10);//hack
            GL.BindTexture(TextureTarget.Texture2D, prevPosition);
            GL.ActiveTexture(TextureUnit.Texture12);//hack
            GL.BindTexture(TextureTarget.Texture2D, prevSecondPosition);

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);
            GL.MemoryBarrier(MemoryBarrierFlags.AllBarrierBits);

            //hack restore
            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, watnorm);
            GL.ActiveTexture(TextureUnit.Texture10);
            GL.BindTexture(TextureTarget.Texture2D, tempAccum);
            GL.ActiveTexture(TextureUnit.Texture12);
            GL.BindTexture(TextureTarget.Texture2D, var1);

            ///==========================================================

            GL.Viewport(0, 0, (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything));
            GL.Disable(EnableCap.DepthTest);
            _denoiseShader.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _denoiseShader.setMatrix4(view, "view");
            _denoiseShader.setMatrix4(projection, "projection");
            _denoiseShader.setMatrix4(view.Inverted(), "invview");
            _denoiseShader.setMatrix4(projection.Inverted(), "invproj");
            _denoiseShader.setMatrix4(prevView, "prevview");
            _denoiseShader.setMatrix4(prevProjection, "prevproj");
            _denoiseShader.setVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _denoiseShader.setFloat(iFrame, "time");
            _denoiseShader.setVector3(_camera.Position, "viewPos");
            _denoiseShader.setVector3(prevCamPos, "lastViewPos");
            _denoiseShader.setVector3(ldir, "ldir");

            for (int i = 0; i < 7; i++)
            {
                if (i % 2 == 0)
                {
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, den1buff);
                    //TODO: hack, no remaining texture unit for den2..
                    GL.ActiveTexture(TextureUnit.Texture11);
                    GL.BindTexture(TextureTarget.Texture2D, den1);
                    _denoiseShader.SetInt("den1", 11);
                    _denoiseShader.SetInt("var1", 12);
                    _denoiseShader.SetInt("colorfog", 27);
                }
                else
                {
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, den2buff);
                    //TODO: hack, no remaining texture unit for den2..
                    GL.ActiveTexture(TextureUnit.Texture11);
                    GL.BindTexture(TextureTarget.Texture2D, den2);
                    _denoiseShader.SetInt("den1", 11);
                    _denoiseShader.SetInt("var1", 23);
                    _denoiseShader.SetInt("colorfog", 14);
                }


                GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);
                GL.MemoryBarrier(MemoryBarrierFlags.AllBarrierBits);
            }

            //restore hack
            GL.ActiveTexture(TextureUnit.Texture11);
            GL.BindTexture(TextureTarget.Texture2D, den2);

            ///==========================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, TAAbuff);

            GL.Viewport(0, 0, (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything));
            GL.Disable(EnableCap.DepthTest);
            _TAAShader.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _TAAShader.setMatrix4(view, "view");
            _TAAShader.setMatrix4(projection, "projection");
            _TAAShader.setMatrix4(view.Inverted(), "invview");
            _TAAShader.setMatrix4(projection.Inverted(), "invproj");
            _TAAShader.setMatrix4(prevView, "prevview");
            _TAAShader.setMatrix4(prevProjection, "prevproj");
            _TAAShader.setVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _TAAShader.setFloat(iFrame, "time");
            _TAAShader.setVector3(_camera.Position, "viewPos");
            _TAAShader.setVector3(prevCamPos, "lastViewPos");
            _TAAShader.setVector3(ldir, "ldir");
            _TAAShader.setVector3(lpos, "lpos");
            _TAAShader.setMatrix4(lightproj, "lightproj");
            _TAAShader.setMatrix4(lightview, "lightview");

            //TODO: hack, we ran out of texture units..
            GL.ActiveTexture(TextureUnit.Texture13);
            GL.BindTexture(TextureTarget.Texture3D, worley);
            
            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            //TODO: revert the hack
            GL.ActiveTexture(TextureUnit.Texture13);
            GL.BindTexture(TextureTarget.Texture2D, upscale);
            
            ///========

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, upscaleBuff);

            GL.Viewport(0, 0, (int)(size.X * KeepScale), (int)(size.Y * KeepScale));
            GL.Disable(EnableCap.DepthTest);
            _upscale.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _upscale.setMatrix4(view, "view");
            _upscale.setMatrix4(projection, "projection");
            _upscale.setMatrix4(view.Inverted(), "invview");
            _upscale.setMatrix4(projection.Inverted(), "invproj");
            _upscale.setMatrix4(prevView, "prevview");
            _upscale.setMatrix4(prevProjection, "prevproj");
            _upscale.setVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _upscale.setFloat(iFrame, "time");
            _upscale.setVector3(_camera.Position, "viewPos");
            _upscale.setVector3(prevCamPos, "lastViewPos");
            _upscale.setVector3(ldir, "ldir");

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            ///=========================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, swapbuff);

            GL.Viewport(0, 0, (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything));
            GL.Disable(EnableCap.DepthTest);
            _swapShader.Use();
            /*
               prevTemporalWeigths = GL.GenTexture();
            prevTemporalOutgoingRadiance = GL.GenTexture();
            prevTemporalPosition = GL.GenTexture();
            prevSpatialWeigths = GL.GenTexture();
            prevSpatialOutgoingRadiance = GL.GenTexture(); 
              */
            GL.BindVertexArray(quadVAO);
            //TODO: consider if we really need these uniforms in swapshaders
            // _testShader.SetMatrix4(model, "model");
            _swapShader.setMatrix4(view, "view");
            _swapShader.setMatrix4(projection, "projection");
            _swapShader.setMatrix4(view.Inverted(), "invview");
            _swapShader.setMatrix4(projection.Inverted(), "invproj");
            _swapShader.setMatrix4(prevView, "prevview");
            _swapShader.setMatrix4(prevProjection, "prevproj");
            _swapShader.setVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _swapShader.setFloat(iFrame, "time");
            _swapShader.setVector3(_camera.Position, "viewPos");
            _swapShader.setVector3(prevCamPos, "lastViewPos");
            _swapShader.setVector3(ldir, "ldir");

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            ///==========================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, swapbuff2);

            GL.Viewport(0, 0, (int)(size.X), (int)(size.Y ));
            GL.Disable(EnableCap.DepthTest);
            _swapShader2.Use();
            /*
               prevTemporalWeigths = GL.GenTexture();
            prevTemporalOutgoingRadiance = GL.GenTexture();
            prevTemporalPosition = GL.GenTexture();
            prevSpatialWeigths = GL.GenTexture();
            prevSpatialOutgoingRadiance = GL.GenTexture(); 
              */
            GL.BindVertexArray(quadVAO);
            //TODO: consider if we really need these uniforms in swapshaders
            // _testShader.SetMatrix4(model, "model");
            _swapShader2.setMatrix4(view, "view");
            _swapShader2.setMatrix4(projection, "projection");
            _swapShader2.setMatrix4(view.Inverted(), "invview");
            _swapShader2.setMatrix4(projection.Inverted(), "invproj");
            _swapShader2.setMatrix4(prevView, "prevview");
            _swapShader2.setMatrix4(prevProjection, "prevproj");
            _swapShader2.setVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _swapShader2.setFloat(iFrame, "time");
            _swapShader2.setVector3(_camera.Position, "viewPos");
            _swapShader2.setVector3(prevCamPos, "lastViewPos");
            _swapShader2.setVector3(ldir, "ldir");

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            //==========================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, TargetFramebuffer);


            GL.Viewport(0, 0, (int)(size.X * KeepScale), (int)(size.Y * KeepScale));
            GL.Disable(EnableCap.DepthTest);
            _finalShader.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _finalShader.setMatrix4(view, "view");
            _finalShader.setMatrix4(projection, "projection");
            _finalShader.setMatrix4(view.Inverted(), "invview");
            _finalShader.setMatrix4(projection.Inverted(), "invproj");
            _finalShader.setVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _finalShader.setFloat(iFrame, "time");
            _finalShader.setVector3(_camera.Position, "viewPos");
            _finalShader.setVector3(ldir, "ldir");
            _finalShader.setMatrix4(lightproj, "lightproj");
            _finalShader.setMatrix4(lightview, "lightview");
            _finalShader.setVector3(lpos, "lpos");
            _finalShader.setMatrix4(lightview.Inverted(), "linvview");
            _finalShader.setMatrix4(lightproj.Inverted(), "linvproj");
            _finalShader.setFloat(bright, "brightness");

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            prevView = view;
            prevProjection = projection;
            prevCamPos = _camera.Position;
            iFrame += 1.0f;

            //SwapBuffers(); //buffer swapping is handled by gui
        }

        public void UpdateResolution(Vector2i resolution)
        {
            _camera.AspectRatio = resolution.X / (float)resolution.Y;
            GL.Viewport(0, 0, resolution.X, resolution.Y);
        }

        private void SetupTextureBindings()
        {
            GL.BindImageTexture(0, rcpos, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            GL.BindImageTexture(1, rcnorm, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            GL.BindImageTexture(2, rcrad, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            GL.BindImageTexture(3, rcfog, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            GL.BindImageTexture(4, suntex, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, watnorm);
            GL.ActiveTexture(TextureUnit.Texture1);
            GL.BindTexture(TextureTarget.Texture2D, colortex);
            GL.ActiveTexture(TextureUnit.Texture2);
            GL.BindTexture(TextureTarget.Texture2D, colortexPosition);
            GL.ActiveTexture(TextureUnit.Texture3);
            GL.BindTexture(TextureTarget.Texture2D, colortexNormal);
            GL.ActiveTexture(TextureUnit.Texture4);
            GL.BindTexture(TextureTarget.Texture2D, colortexAlbedo);
            GL.ActiveTexture(TextureUnit.Texture5);
            GL.BindTexture(TextureTarget.Texture2D, colortexSecondPosition);
            GL.ActiveTexture(TextureUnit.Texture6);
            GL.BindTexture(TextureTarget.Texture2D, temporalWeigths);
            GL.ActiveTexture(TextureUnit.Texture7);
            GL.BindTexture(TextureTarget.Texture2D, temporalOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture8);
            GL.BindTexture(TextureTarget.Texture2D, spatialWeigths);
            GL.ActiveTexture(TextureUnit.Texture9);
            GL.BindTexture(TextureTarget.Texture2D, spatialOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture10);
            GL.BindTexture(TextureTarget.Texture2D, tempAccum);
            GL.ActiveTexture(TextureUnit.Texture11);
            GL.BindTexture(TextureTarget.Texture2D, den2);
            GL.ActiveTexture(TextureUnit.Texture12);
            GL.BindTexture(TextureTarget.Texture2D, var1);
            GL.ActiveTexture(TextureUnit.Texture13);
            GL.BindTexture(TextureTarget.Texture2D, upscale);
            GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);
            GL.ActiveTexture(TextureUnit.Texture14);
            GL.BindTexture(TextureTarget.Texture2D, colorfog2);
            //colortexFog
            GL.ActiveTexture(TextureUnit.Texture15);
            GL.BindTexture(TextureTarget.Texture2D, colortexFogPrev);
            GL.ActiveTexture(TextureUnit.Texture16);
            GL.BindTexture(TextureTarget.Texture2D, spatialWeigthsFog);
            GL.ActiveTexture(TextureUnit.Texture17);
            GL.BindTexture(TextureTarget.Texture2D, TAA);
            GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);
            GL.ActiveTexture(TextureUnit.Texture18);
            GL.BindTexture(TextureTarget.Texture2D, suntex);
            GL.ActiveTexture(TextureUnit.Texture19);
            GL.BindTexture(TextureTarget.Texture2D, watpos);
            //
            GL.ActiveTexture(TextureUnit.Texture20);
            GL.BindTexture(TextureTarget.Texture2D, temporalPosition);
            GL.ActiveTexture(TextureUnit.Texture21);
            GL.BindTexture(TextureTarget.Texture2D, upscale2);
            GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);
            GL.ActiveTexture(TextureUnit.Texture22);
            GL.BindTexture(TextureTarget.Texture2D, prevNormalDepth);
            GL.ActiveTexture(TextureUnit.Texture23);
            GL.BindTexture(TextureTarget.Texture2D, var2);
            GL.ActiveTexture(TextureUnit.Texture24);
            GL.BindTexture(TextureTarget.Texture2D, prevTAA);
            GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);
            GL.ActiveTexture(TextureUnit.Texture25);
            GL.BindTexture(TextureTarget.Texture2D, colortexSecondAlbedo);
            GL.ActiveTexture(TextureUnit.Texture26);
            GL.BindTexture(TextureTarget.Texture2D, colortexHoldInfo);
            GL.ActiveTexture(TextureUnit.Texture27);
            GL.BindTexture(TextureTarget.Texture2D, colorfog);
            GL.ActiveTexture(TextureUnit.Texture28);
            GL.BindTexture(TextureTarget.Texture2D, colortexReflection);
            GL.ActiveTexture(TextureUnit.Texture29);
            GL.BindTexture(TextureTarget.Texture2D, colortexFog);
            GL.ActiveTexture(TextureUnit.Texture30);
            GL.BindTexture(TextureTarget.Texture2D, spatialOutgoingRadianceFog);
            GL.ActiveTexture(TextureUnit.Texture31);
            GL.BindTexture(TextureTarget.Texture2D, skytex);
            GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);
        }
    }
}
