using LearnOpenTK.Common;
using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace RealTimeFractalRendererLibrary
{
    public class Renderer
    {
        int quadVBO;
        int quadVAO;
        int quadEBO;

        readonly float[] quad =
        [
            //Position          Texture coordinates
            1.0f,  1.0f, 0.0f, 1.0f, 1.0f, // top right
            1.0f, -1.0f, 0.0f, 1.0f, 0.0f, // bottom right
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // bottom left
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f  // top left
        ];

        readonly uint[] quadindices = [  // note that we start from 0!
            0, 1, 3,   // first triangle
            1, 2, 3    // second triangle
        ];

        private readonly TextureBinding colortex = new();
        private readonly TextureBinding colortexHoldInfo = new();

        private readonly TextureBinding colortexReflection = new();
        private readonly TextureBinding colortexPosition = new();
        private readonly TextureBinding colortexNormal = new();
        private readonly TextureBinding colortexAlbedo = new();
        private readonly TextureBinding colortexFogPos = new();

        private readonly TextureBinding colortexSecondPosition = new();
        private readonly TextureBinding colortexSecondAlbedo = new();

        private readonly TextureBinding watpos = new();
        private readonly TextureBinding watnorm = new();

        int colorbuff2;
        private Shader _initShader = default!;
        private readonly TextureBinding colorfog = new();
        private readonly TextureBinding colorfog2 = new();

        int colorbuff;
        Stopwatch stopWatch = default!;

        float iFrame = 0.0f;

        private Shader _testShader = default!;
        private Shader _finalShader = default!;

        private Shader _TemporalRestirShader = default!;
        int temporalbuff;
        private readonly TextureBinding temporalWeigths = new();
        private readonly TextureBinding temporalOutgoingRadiance = new();

        private readonly TextureBinding temporalPosition = new();

        private Shader _SpatialRestirShader = default!;
        int spatialbuff;
        private readonly TextureBinding spatialWeigths = new();
        private readonly TextureBinding spatialOutgoingRadiance = new();

        private Shader _swapShader = default!;
        int swapbuff;

        private Shader _swapShader2 = default!;
        int swapbuff2;

        private readonly TextureBinding prevTemporalWeigths = new();
        private readonly TextureBinding prevTemporalOutgoingRadiance = new();
        private readonly TextureBinding prevTemporalPosition = new();
        private readonly TextureBinding prevSpatialWeigths = new();
        private readonly TextureBinding prevSpatialOutgoingRadiance = new();
        private readonly TextureBinding prevNormalDepth = new();

        private readonly TextureBinding prevAcc = new();
        private readonly TextureBinding prevTAA = new();

        private Shader _tempAccumShader = default!;
        int tempAccumbuff;
        private readonly TextureBinding tempAccum = new();

        private Shader _upscale = default!;
        private readonly TextureBinding upscale = new();
        private readonly TextureBinding upscale2 = new();

        int upscaleBuff;


        private Shader _TAAShader = default!;
        int TAAbuff;
        private readonly TextureBinding TAA = new();

        private Shader _denoiseShader = default!;
        //private Shader _denoiseShader2 = default!;
        private readonly TextureBinding den1 = new();
        private readonly TextureBinding den2 = new();
        int den1buff;
        int den2buff;
        private readonly TextureBinding var1 = new();
        private readonly TextureBinding var2 = new();

        int rcpos;
        int rcnorm;
        int rcrad;
        int rcfog;

        private Shader _skyShader = default!;

        int skybuff;
        private readonly TextureBinding skytex = new();

        private Shader _sunShader = default!;

        int sunbuff;
        private readonly TextureBinding suntex = new();


        private Shader _RC = default!;

        private readonly TextureBinding worley = new(TextureTarget.Texture3D);

        private Shader _worl = default!;



        private Camera _camera = default!;
        public Camera Camera => _camera;

        /// <summary>
        /// The framebuffer where the final image is rendered to.
        /// </summary>
        public int TargetFramebuffer { get; set; } = 0;

        private Vector3 prevCamPos = new(0.0f, 0.0f, 0.0f);

        private const bool EnableDebugging = true;
        private const bool isEverythingLower = true;

        private const float KeepScale = 1.0f; //the default scale
        private static float ScaleEverything = KeepScale;

        private const int shadowScale = 2048;

        private Vector2 size = new(1920.0f / ScaleEverything, 1080.0f / ScaleEverything);

        private Matrix4 prevView;
        private Matrix4 prevProjection;

        public Vector3 ldir = new(0.0f, 0.1f, -0.9f);

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

        private bool isInitialized = false;

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

            if (EnableDebugging)
            {
                _debugProcCallbackHandle = GCHandle.Alloc(_debugProcCallback);
                GL.DebugMessageCallback(_debugProcCallback, IntPtr.Zero);
                GL.Enable(EnableCap.DebugOutput);
                GL.Enable(EnableCap.DebugOutputSynchronous);
            }

           // GL.Enable(EnableCap.Texture3DExt);
            GL.ClearColor(0.2f, 0.3f, 0.3f, 1.0f);

            upscaleBuff = GL.GenFramebuffer();
            colorbuff = GL.GenFramebuffer();
            colorbuff2 = GL.GenFramebuffer();
            skybuff = GL.GenFramebuffer();
            sunbuff = GL.GenFramebuffer();

            if (isEverythingLower) {
                ScaleEverything = 1.0f;
            }

                 
            setupTex2D(colorfog2.Handle);          
            setupTex2D(colorfog.Handle);
            setupTex2D(watnorm.Handle);
            setupTex2D(watpos.Handle);
            setupTex2D(skytex.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.LinearMipmapLinear, TextureMagFilter.Linear);
            setupTex2D(suntex.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear, TextureWrapMode.ClampToBorder,
                true, shadowScale, shadowScale);
            setupTex3D(worley.Handle, 512, 512, 512, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);
            setupTex3D(rcpos, 180, 180, 180, PixelInternalFormat.Rgba16f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            setupTex3D(rcfog, 180, 180, 180, PixelInternalFormat.Rgba16f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            setupTex3D(rcnorm, 180, 180, 180, PixelInternalFormat.Rgba16f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            setupTex3D(rcrad, 180, 180, 180, PixelInternalFormat.Rgba16f, TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.MirroredRepeat);
            setupTex2D(colortex.Handle, false);
            setupTex2D(colortexPosition.Handle, true, PixelInternalFormat.Rgba32f);
            setupTex2D(colortexFogPos.Handle, false);
            setupTex2D(colortexHoldInfo.Handle);
            setupTex2D(colortexNormal.Handle, true, PixelInternalFormat.Rgba32f);
            setupTex2D(colortexAlbedo.Handle);
            setupTex2D(colortexSecondPosition.Handle, true, PixelInternalFormat.Rgba32f);
            setupTex2D(colortexReflection.Handle, false);
            setupTex2D(colortexSecondAlbedo.Handle, false);

            //====================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, skybuff);

            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, skytex.Handle, 0
                );

            DrawBuffersEnum[] attx = [DrawBuffersEnum.ColorAttachment0];
            GL.DrawBuffers(1, attx);

            //=======================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, sunbuff);

            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, suntex.Handle, 0
                );

            DrawBuffersEnum[] attxs = [DrawBuffersEnum.ColorAttachment0];
            GL.DrawBuffers(1, attxs);
            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, colorbuff2);

            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, colortexPosition.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment1,
                TextureTarget.Texture2D, colortexNormal.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment2,
                TextureTarget.Texture2D, colortexAlbedo.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment3,
                TextureTarget.Texture2D, colortexHoldInfo.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment4,
                TextureTarget.Texture2D, colorfog.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment5,
                TextureTarget.Texture2D, watpos.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment6,
                TextureTarget.Texture2D, watnorm.Handle, 0
                );
            DrawBuffersEnum[] att = [ DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1,
                DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3, DrawBuffersEnum.ColorAttachment4,
             DrawBuffersEnum.ColorAttachment5, DrawBuffersEnum.ColorAttachment6];
            GL.DrawBuffers(7, att);


            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, colorbuff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, colortex.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment1,
                TextureTarget.Texture2D, colortexSecondPosition.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment2,
                TextureTarget.Texture2D, colortexReflection.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment3,
                TextureTarget.Texture2D, colortexSecondAlbedo.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment4,
                TextureTarget.Texture2D, colorfog.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment5,
                TextureTarget.Texture2D, colortexFogPos.Handle, 0
                );
            DrawBuffersEnum[] atts = [ DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1,
                DrawBuffersEnum.ColorAttachment2,  DrawBuffersEnum.ColorAttachment3
            ,  DrawBuffersEnum.ColorAttachment4,  DrawBuffersEnum.ColorAttachment5];
            GL.DrawBuffers(6, atts);


            temporalbuff = GL.GenFramebuffer();

            
            setupTex2D(temporalWeigths.Handle, false);

            setupTex2D(temporalOutgoingRadiance.Handle, false);
            setupTex2D(temporalPosition.Handle, false, PixelInternalFormat.Rgba32f);


            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, temporalbuff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, temporalWeigths.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment1,
                TextureTarget.Texture2D, temporalOutgoingRadiance.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment2,
                TextureTarget.Texture2D, temporalPosition.Handle, 0
                );
            DrawBuffersEnum[] att2 = [ DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2];
            GL.DrawBuffers(3, att2);

            spatialbuff = GL.GenFramebuffer();

            
            setupTex2D(spatialWeigths.Handle, false);
            setupTex2D(spatialOutgoingRadiance.Handle, false);


            GL.BindFramebuffer(FramebufferTarget.Framebuffer, spatialbuff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, spatialWeigths.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment1,
                TextureTarget.Texture2D, spatialOutgoingRadiance.Handle, 0
                );
            DrawBuffersEnum[] att4 = [DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1];
            GL.DrawBuffers(2, att4);

            TAAbuff = GL.GenFramebuffer();

            
            setupTex2D(TAA.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.LinearMipmapLinear, TextureMagFilter.Linear);

            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, TAAbuff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, TAA.Handle, 0
                );
            DrawBuffersEnum[] att10 = [DrawBuffersEnum.ColorAttachment0];
            GL.DrawBuffers(1, att10);

 
            setupTex2D(upscale.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.NearestMipmapLinear, TextureMagFilter.Nearest);
            setupTex2D(upscale2.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.NearestMipmapLinear, TextureMagFilter.Nearest);

            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, upscaleBuff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, upscale.Handle, 0
                );
            DrawBuffersEnum[] att11 = [DrawBuffersEnum.ColorAttachment0];
            GL.DrawBuffers(1, att11);

            swapbuff = GL.GenFramebuffer();
            swapbuff2 = GL.GenFramebuffer();
           
            setupTex2D(prevTemporalWeigths.Handle, false);
            
            setupTex2D(prevTemporalOutgoingRadiance.Handle, false);

           
            setupTex2D(prevTemporalPosition.Handle, false);


            setupTex2D(prevSpatialWeigths.Handle, false);

    
            setupTex2D(prevSpatialOutgoingRadiance.Handle, false);

            setupTex2D(prevNormalDepth.Handle, true, PixelInternalFormat.Rgba32f);


           
            setupTex2D(prevAcc.Handle, true);


            setupTex2D(prevTAA.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.LinearMipmapLinear, TextureMagFilter.Linear);

  
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, swapbuff);

            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, prevNormalDepth.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment1,
               TextureTarget.Texture2D, prevAcc.Handle, 0
               );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment2,
               TextureTarget.Texture2D, prevTAA.Handle, 0
               );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment3,
               TextureTarget.Texture2D, upscale2.Handle, 0
               );
            DrawBuffersEnum[] att3 = [DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3];
            GL.DrawBuffers(4, att3);




            GL.BindFramebuffer(FramebufferTarget.Framebuffer, swapbuff2);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, prevTemporalWeigths.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment1,
                TextureTarget.Texture2D, prevTemporalOutgoingRadiance.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment2,
                TextureTarget.Texture2D, prevTemporalPosition.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment3,
                TextureTarget.Texture2D, prevSpatialWeigths.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment4,
                TextureTarget.Texture2D, prevSpatialOutgoingRadiance.Handle, 0
                );

            DrawBuffersEnum[] att3s = [ DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2,
                DrawBuffersEnum.ColorAttachment3, DrawBuffersEnum.ColorAttachment4];
            GL.DrawBuffers(5, att3s);


            //private Shader _tempAccumShader;
            //int tempAccumbuff;
            //int tempAccum;

            //===============================================================

            den1buff = GL.GenFramebuffer();
            den2buff = GL.GenFramebuffer();
       
            setupTex2D(den1.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);
            setupTex2D(den2.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);

            setupTex2D(var1.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);

            setupTex2D(var2.Handle, true, PixelInternalFormat.Rgba16f, TextureMinFilter.Linear, TextureMagFilter.Linear);


            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, den1buff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, den2.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment1,
                TextureTarget.Texture2D, var2.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment2,
               TextureTarget.Texture2D, colorfog2.Handle, 0
               );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment3,
               TextureTarget.Texture2D, colortexSecondAlbedo.Handle, 0
               );
            DrawBuffersEnum[] att8 = [DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3];
            GL.DrawBuffers(4, att8);



            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, den2buff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, den1.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment1,
               TextureTarget.Texture2D, var1.Handle, 0
               );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment2,
              TextureTarget.Texture2D, colorfog.Handle, 0
              );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
               FramebufferAttachment.ColorAttachment3,
               TextureTarget.Texture2D, colortexSecondAlbedo.Handle, 0
               );
            DrawBuffersEnum[] att9 = [DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3];
            GL.DrawBuffers(4, att9);



            //==========================================================-=

            tempAccumbuff = GL.GenFramebuffer();
         
            setupTex2D(tempAccum.Handle);

            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, tempAccumbuff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, tempAccum.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment1,
                TextureTarget.Texture2D, den1.Handle, 0
                );
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment2,
                TextureTarget.Texture2D, var1.Handle, 0
                );
            DrawBuffersEnum[] att5 = [DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2];
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

            //TODO: Some of these uniforms are commented out in the shaders. Decide what to do with them (delete or start feature branches).

            //tempAccum
            quadVBO = GL.GenBuffer();
            quadVAO = GL.GenVertexArray();
            quadEBO = GL.GenBuffer();

            _initShader.Use();
            _initShader.SetInt("skytex", skytex.Unit);

            _skyShader.Use();
            _skyShader.SetInt("skyprev", skytex.Unit);
            _skyShader.SetInt("worl", worley.Unit);
            //_skyShader.SetInt("suns", suntex.Unit);

            _sunShader.Use();
            //_sunShader.SetInt("sunprev", suntex.Unit);
            //_sunShader.SetInt("worl", worley.Unit);

            _RC.Use();
            _RC.SetInt("skytex", skytex.Unit);
            _RC.SetInt("suntex", suntex.Unit);

            _testShader.Use();
            _testShader.SetInt("position", colortexPosition.Unit);
            _testShader.SetInt("normal", colortexNormal.Unit);
            _testShader.SetInt("albedo", colortexAlbedo.Unit);
            _testShader.SetInt("holdinfo", colortexHoldInfo.Unit);
            //_testShader.SetInt("colorfog", colorfog.Unit);
            _testShader.SetInt("skytex", skytex.Unit);
            _testShader.SetInt("suntex", suntex.Unit);
            _testShader.SetInt("watpos", watpos.Unit);
            _testShader.SetInt("watnorm", watnorm.Unit);

            _TemporalRestirShader.Use();
            _TemporalRestirShader.SetInt("color", colortex.Unit);
            _TemporalRestirShader.SetInt("position", colortexPosition.Unit);
            _TemporalRestirShader.SetInt("normal", colortexNormal.Unit);
            _TemporalRestirShader.SetInt("secondpos", colortexSecondPosition.Unit);
            _TemporalRestirShader.SetInt("prevW", prevTemporalWeigths.Unit);
            _TemporalRestirShader.SetInt("prevL", prevTemporalOutgoingRadiance.Unit);
            _TemporalRestirShader.SetInt("prevP", prevTemporalPosition.Unit);
            _TemporalRestirShader.SetInt("prevN", prevNormalDepth.Unit);
            //_TemporalRestirShader.SetInt("reflAlb", colortexSecondAlbedo.Unit);

            _SpatialRestirShader.Use();
            //_SpatialRestirShader.SetInt("color", colortex.Unit);
            _SpatialRestirShader.SetInt("position", colortexPosition.Unit);
            _SpatialRestirShader.SetInt("normal", colortexNormal.Unit);
            _SpatialRestirShader.SetInt("albedo", colortexAlbedo.Unit);
            //_SpatialRestirShader.SetInt("secondpos", colortexSecondPosition.Unit);
            _SpatialRestirShader.SetInt("temppos", temporalPosition.Unit);
            _SpatialRestirShader.SetInt("prevW", prevSpatialWeigths.Unit);
            _SpatialRestirShader.SetInt("prevL", prevSpatialOutgoingRadiance.Unit);
            _SpatialRestirShader.SetInt("tempW", temporalWeigths.Unit);
            _SpatialRestirShader.SetInt("tempL", temporalOutgoingRadiance.Unit);
            _SpatialRestirShader.SetInt("prevN", prevNormalDepth.Unit);
         
            _tempAccumShader.Use();
            //_tempAccumShader.SetInt("color", colortex.Unit);
            _tempAccumShader.SetInt("position", colortexPosition.Unit);
            _tempAccumShader.SetInt("normal", colortexNormal.Unit);
            _tempAccumShader.SetInt("albedo", colortexAlbedo.Unit);
            _tempAccumShader.SetInt("weightS", spatialWeigths.Unit);
            _tempAccumShader.SetInt("outgoingrS", spatialOutgoingRadiance.Unit);
            _tempAccumShader.SetInt("prevN", prevNormalDepth.Unit);
            _tempAccumShader.SetInt("prevAcc", prevAcc.Unit);
            _tempAccumShader.SetInt("prevvar", var2.Unit);

            _denoiseShader.Use();
            _denoiseShader.SetInt("color", colortex.Unit);
            //_denoiseShader.SetInt("position", colortexPosition.Unit);
            _denoiseShader.SetInt("normal", colortexNormal.Unit);
            _denoiseShader.SetInt("albedo", colortexAlbedo.Unit);
            _denoiseShader.SetInt("secondpos", colortexSecondPosition.Unit);
           
            _denoiseShader.SetInt("reflN", colortexReflection.Unit);
            _denoiseShader.SetInt("reflectionAlb", colortexSecondAlbedo.Unit);
            _denoiseShader.SetInt("holdinfo", colortexHoldInfo.Unit);
            _denoiseShader.SetInt("watpos", watpos.Unit);
            _denoiseShader.SetInt("watnorm", watnorm.Unit);
            //_denoiseShader - some ping-pong textures are set each frame so they're omitted here

            _TAAShader.Use();
            //_TAAShader.SetInt("color", colortex.Unit); //TODO: review. This was commented out.
            _TAAShader.SetInt("position", colortexPosition.Unit);
            _TAAShader.SetInt("normal", colortexNormal.Unit);
            _TAAShader.SetInt("albedo", colortexAlbedo.Unit);
            _TAAShader.SetInt("secondpos", colortexSecondPosition.Unit);
            //_TAAShader.SetInt("Acc", tempAccum.Unit);
            _TAAShader.SetInt("den1", den2.Unit);//den2 bound here
            _TAAShader.SetInt("prevTAA", prevTAA.Unit);
            _TAAShader.SetInt("reflAlb", colortexSecondAlbedo.Unit);
            _TAAShader.SetInt("colorfog", colorfog.Unit);
            //_TAAShader.SetInt("holdinfo", colortexHoldInfo.Unit);
            _TAAShader.SetInt("suntex", suntex.Unit);
            _TAAShader.SetInt("skyTex", skytex.Unit);
            _TAAShader.SetInt("watpos", watpos.Unit);
            _TAAShader.SetInt("watnorm", watnorm.Unit);
            _TAAShader.SetInt("worl", worley.Unit);

            _upscale.Use();
            _upscale.SetInt("TAA", TAA.Unit);
           

            _swapShader.Use();
          
            _swapShader.SetInt("normal", colortexNormal.Unit);
            _swapShader.SetInt("Acc", tempAccum.Unit);
            _swapShader.SetInt("TAA", TAA.Unit);
            _swapShader.SetInt("up", upscale.Unit);

            _swapShader2.Use();
            _swapShader2.SetInt("tempWeights", temporalWeigths.Unit);
            _swapShader2.SetInt("tempOutL", temporalOutgoingRadiance.Unit);
            _swapShader2.SetInt("tempPos", temporalPosition.Unit);
            _swapShader2.SetInt("spatWe", spatialWeigths.Unit);
            _swapShader2.SetInt("spatLO", spatialOutgoingRadiance.Unit);
        

            _finalShader.Use();
            //_finalShader.SetInt("color", colortex.Unit);
            _finalShader.SetInt("normal", colortexNormal.Unit);
            _finalShader.SetInt("albedo", colortexAlbedo.Unit);
   
            _finalShader.SetInt("TAA", upscale.Unit); //TODO: review. "TAA" sampler is bound to "upscale" here.
            _finalShader.SetInt("fogLO", TAA.Unit); //TODO: review. "fogLO" sampler is bound to "TAA" here.
            _finalShader.SetInt("watpos", watpos.Unit);





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

            GL.BindImageTexture(0, worley.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            _worl.Use();
            GL.DispatchCompute(64, 64, 64);

            //setup image unit bindings
            GL.BindImageTexture(0, rcpos, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            GL.BindImageTexture(1, rcnorm, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            GL.BindImageTexture(2, rcrad, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            GL.BindImageTexture(3, rcfog, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            GL.BindImageTexture(4, suntex.Handle, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            //CursorState = CursorState.Grabbed;

            isInitialized = true;
        }


        public void RenderFrame()
        {
            if (!isInitialized) throw new InvalidOperationException("Unable to render frame before successful initialization.");

            //GL.Clear(ClearBufferMask.ColorBufferBit);

            ldir = Vector3.Normalize(ldir);

            Vector3 lpos = new( _camera.Position.X + ldir.X * 270.0f, ldir.Y * 270.0f, _camera.Position.Z + ldir.Z * 270.0f );

            Matrix4 lightview = Matrix4.LookAt(
                lpos,
                new Vector3(_camera.Position.X, 0.0f, _camera.Position.Z),
                new Vector3(0.0f, 1.0f, 0.0f)
                );

            //Matrix4 lightview = Matrix4.

            //Matrix4 lightproj = Matrix4.CreateOrthographicOffCenter(
            //    -60.0f, 60.0f, -60.0f, 60.0f, 1.0f, 100.0f
            //    );
            Matrix4 lightproj = Matrix4.CreateOrthographic(400.0f, 400.0f, 0.0f, 127.5f);

            //Vector3 nm = Vector3.Normalize(lpos - new Vector3(_camera.Position.X, 0.0f, _camera.Position.Z));

             Vector3 lpos2 = new(_camera.Position.X + ldir.X * 360.0f, ldir.Y * 360.0f, _camera.Position.Z + ldir.Z * 360.0f);
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
            _initShader.SetMatrix4(view.Inverted(), "invview");
            _initShader.SetMatrix4(projection.Inverted(), "invproj");
            _initShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _initShader.SetFloat(iFrame, "time");
            _initShader.SetVector3(_camera.Position, "viewPos");


            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);
            GL.MemoryBarrier(MemoryBarrierFlags.AllBarrierBits);

            //==============================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, skybuff);

            GL.Viewport(0, 0, (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything));
            //GL.ClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            //GL.Clear(ClearBufferMask.ColorBufferBit);
            GL.Disable(EnableCap.DepthTest);
            _skyShader.Use();
            _skyShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _skyShader.SetFloat(iFrame, "time");
            _skyShader.SetVector3(ldir, "ldir");
     

            GL.BindVertexArray(quadVAO);

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);
            GL.GenerateTextureMipmap(skytex.Handle);

            //=================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, sunbuff);

            GL.Viewport(0, 0, (int)shadowScale, (int)shadowScale);
            //GL.ClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            //GL.Clear(ClearBufferMask.ColorBufferBit);
            GL.Disable(EnableCap.DepthTest);
            _sunShader.Use();
            _sunShader.SetVector3(ldir, "ldir");
            _sunShader.SetVector3(lpos, "lpos");
            _sunShader.SetMatrix4(lightproj, "lightproj");
            _sunShader.SetMatrix4(lightview, "lightview");
            _sunShader.SetFloat((float)shadowScale, "scale");
          
            GL.BindVertexArray(quadVAO);

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            //===================================================

            _RC.Use();
            GL.BindVertexArray(quadVAO);
           
            _RC.SetFloat(iFrame, "time");
            //_RC.SetFloat(stopWatch.ElapsedMilliseconds, "time2");
            _RC.SetVector3(_camera.Position, "viewPos");
            _RC.SetVector3(prevCamPos, "lastViewPos");
            _RC.SetVector3(ldir, "ldir");
            _RC.SetMatrix4(lightproj, "lightproj");
            _RC.SetMatrix4(lightview, "lightview");
            //_RC.SetVector3(lpos, "lpos");

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
         
            _testShader.SetMatrix4(view.Inverted(), "invview");
            _testShader.SetMatrix4(projection.Inverted(), "invproj");
            _testShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _testShader.SetFloat(iFrame, "time");
            //_testShader.SetFloat(stopWatch.ElapsedMilliseconds, "time2");
            _testShader.SetVector3(_camera.Position, "viewPos");
            _testShader.SetVector3(ldir, "ldir");
            _testShader.SetMatrix4(lightproj, "lightproj");
            _testShader.SetMatrix4(lightview, "lightview");
            //_testShader.SetVector3(lpos, "lpos");

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);


            ///==========================================================
            
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, temporalbuff);

            GL.Viewport(0, 0, (int)size.X, (int)size.Y);
            GL.Disable(EnableCap.DepthTest);
            _TemporalRestirShader.Use();
            GL.BindVertexArray(quadVAO);
      
            _TemporalRestirShader.SetMatrix4(prevView, "prevview");
            _TemporalRestirShader.SetMatrix4(prevProjection, "prevproj");

            _TemporalRestirShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _TemporalRestirShader.SetFloat(iFrame, "time");
          
            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            ///==========================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, spatialbuff);

            GL.Viewport(0, 0, (int)size.X, (int)size.Y);
            GL.Disable(EnableCap.DepthTest);
            _SpatialRestirShader.Use();
            GL.BindVertexArray(quadVAO);
        
            _SpatialRestirShader.SetMatrix4(prevView, "prevview");
            _SpatialRestirShader.SetMatrix4(prevProjection, "prevproj");
            _SpatialRestirShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _SpatialRestirShader.SetFloat(iFrame, "time");
           
            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            ///==========================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, tempAccumbuff);

            GL.Viewport(0, 0, (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything));
            GL.Disable(EnableCap.DepthTest);
            _tempAccumShader.Use();
            GL.BindVertexArray(quadVAO);
      
            _tempAccumShader.SetMatrix4(prevView, "prevview");
            _tempAccumShader.SetMatrix4(prevProjection, "prevproj");
            _tempAccumShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _tempAccumShader.SetFloat(iFrame, "time");

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);
            GL.MemoryBarrier(MemoryBarrierFlags.AllBarrierBits);

            ///==========================================================

            GL.Viewport(0, 0, (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything));
            GL.Disable(EnableCap.DepthTest);
            _denoiseShader.Use();
            GL.BindVertexArray(quadVAO);
    
            _denoiseShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");

            for (int i = 0; i < 7; i++)
            {
                if (i % 2 == 0)
                {
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, den1buff);
                    _denoiseShader.SetInt("den1", den1.Unit);
                    _denoiseShader.SetInt("var1", var1.Unit);
                    _denoiseShader.SetInt("colorfog", colorfog.Unit);
                }
                else
                {
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, den2buff);
                    _denoiseShader.SetInt("den1", den2.Unit);
                    _denoiseShader.SetInt("var1", var2.Unit);
                    _denoiseShader.SetInt("colorfog", colorfog2.Unit);
                }


                GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);
                GL.MemoryBarrier(MemoryBarrierFlags.AllBarrierBits);
            }

            ///==========================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, TAAbuff);

            GL.Viewport(0, 0, (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything));
            GL.Disable(EnableCap.DepthTest);
            _TAAShader.Use();
            GL.BindVertexArray(quadVAO);
            _TAAShader.SetMatrix4(view.Inverted(), "invview");
            _TAAShader.SetMatrix4(projection.Inverted(), "invproj");
            _TAAShader.SetMatrix4(prevView, "prevview");
            _TAAShader.SetMatrix4(prevProjection, "prevproj");
            _TAAShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _TAAShader.SetFloat(iFrame, "time");
            _TAAShader.SetVector3(_camera.Position, "viewPos");
            _TAAShader.SetVector3(ldir, "ldir");
            _TAAShader.SetMatrix4(lightproj, "lightproj");
            _TAAShader.SetMatrix4(lightview, "lightview");
                        
            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);
            GL.GenerateTextureMipmap(TAA.Handle);
                        
            ///========

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, upscaleBuff);

            GL.Viewport(0, 0, (int)(size.X * KeepScale), (int)(size.Y * KeepScale));
            GL.Disable(EnableCap.DepthTest);
            _upscale.Use();
            GL.BindVertexArray(quadVAO);
         
            _upscale.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
         
            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);
            //GL.GenerateTextureMipmap(!isEverythingLower ? upscale2 : upscale); //TODO: review, this seems intended but causes a "ghost" sky effect

            ///=========================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, swapbuff);

            GL.Viewport(0, 0, (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything));
            GL.Disable(EnableCap.DepthTest);
            _swapShader.Use();
            GL.BindVertexArray(quadVAO);

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);
            GL.GenerateTextureMipmap(prevTAA.Handle);
            GL.GenerateTextureMipmap(upscale2.Handle);

            ///==========================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, swapbuff2);

            GL.Viewport(0, 0, (int)(size.X), (int)(size.Y ));
            GL.Disable(EnableCap.DepthTest);
            _swapShader2.Use();
            GL.BindVertexArray(quadVAO);

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            //==========================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, TargetFramebuffer);

            GL.Viewport(0, 0, (int)(size.X * KeepScale), (int)(size.Y * KeepScale));
            GL.Disable(EnableCap.DepthTest);
            _finalShader.Use();
            GL.BindVertexArray(quadVAO);
      
            _finalShader.SetMatrix4(projection, "projection");
    
            _finalShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _finalShader.SetFloat(iFrame, "time");
      
            _finalShader.SetFloat(bright, "brightness");
            _finalShader.SetVector3(_camera.Position, "viewPos");

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

    }
}
