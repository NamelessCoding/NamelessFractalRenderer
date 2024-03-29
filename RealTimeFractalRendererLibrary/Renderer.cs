﻿using LearnOpenTK.Common;
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
        private Shader _initShader = default!;
        int colorfog;
        int colorfog2;

        int colorbuff;
        Stopwatch stopWatch = default!;

        float iFrame = 0.0f;

        private Shader _testShader = default!;
        private Shader _finalShader = default!;

        private Shader _TemporalRestirShader = default!;
        int temporalbuff;
        int temporalWeigths;
        int temporalOutgoingRadiance;
        int temporalWeigthsFog;
        int temporalOutgoingRadianceFog;

        int temporalPosition;
        int temporalPositionFog;

        private Shader _SpatialRestirShader = default!;
        int spatialbuff;
        int spatialWeigths;
        int spatialOutgoingRadiance;

        int spatialWeigthsFog;
        int spatialOutgoingRadianceFog;

        private Shader _swapShader = default!;
        int swapbuff;

        private Shader _swapShader2 = default!;
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

        private Shader _tempAccumShader = default!;
        int tempAccumbuff;
        int tempAccum;

        private Shader _upscale = default!;
        int upscale;
        int upscale2;

        int upscaleBuff;


        private Shader _TAAShader = default!;
        int TAAbuff;
        int TAA;

        private Shader _denoiseShader = default!;
        //private Shader _denoiseShader2 = default!;
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

        private Shader _skyShader = default!;

        int skybuff;
        int skytex;

        private Shader _sunShader = default!;

        int sunbuff;
        int suntex;


        private Shader _RC = default!;

        int worley;

        private Shader _worl = default!;



        private Camera _camera = default!;
        public Camera Camera => _camera;

        /// <summary>
        /// The framebuffer where the final image is rendered to.
        /// </summary>
        public int TargetFramebuffer { get; set; } = 0;

        private Vector3 prevCamPos = new(0.0f, 0.0f, 0.0f);

        private readonly bool enableDebugging = false;
        private readonly bool isEverythingLower = true;

        private const float KeepScale = 1.5f; //the default scale
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

            GL.BindTexture(TextureTarget.Texture2D, colorfog2);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);
            float[] borderColor = [1.0f, 1.0f, 1.0f, 1.0f];

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);


            GL.BindTexture(TextureTarget.Texture2D, colorfog);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);

            GL.BindTexture(TextureTarget.Texture2D, watnorm);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);


            GL.BindTexture(TextureTarget.Texture2D, watpos);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);



            GL.BindTexture(TextureTarget.Texture2D, skytex);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);

           /* GL.BindTexture(TextureTarget.Texture2D, suntex);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)shadowScale, (int)shadowScale, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
           */

            GL.BindTexture(TextureTarget.Texture2D, suntex);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToBorder);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToBorder);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                shadowScale, shadowScale, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);


            /*
             glBindTexture(GL_TEXTURE_3D, texname);
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_Nearest);
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_Nearest);
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB8, WIDTH, HEIGHT, DEPTH, 0, GL_RGB, 
             GL_UNSIGNED_BYTE, texels);
             */


            //glGenTextures(1, &HeightMap);
            //glBindTexture(GL_TEXTURE_2D, HeightMap);
            //glTexImage2D(GL_TEXTURE_2D, 0, GL_Rgba16f, 513, 513, 0, GL_RGBA, GL_FLOAT, 0);
            //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_Nearest);
            //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_Nearest);
            //glGenerateMipmap(GL_TEXTURE_2D);

            GL.BindTexture(TextureTarget.Texture3D, worley);
            // GL.BindImageTexture(0, rcpos, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            // GL.BindImageTexture(0, rcpos, 0, );
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);

            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapR, (int)TextureWrapMode.Repeat);

            GL.TexImage3D(TextureTarget.Texture3D, 0,
                PixelInternalFormat.Rgba16f,
                512, 512, 512, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.BindImageTexture(0, worley, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);



            GL.BindTexture(TextureTarget.Texture3D, rcpos);
            // GL.BindImageTexture(0, rcpos, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            // GL.BindImageTexture(0, rcpos, 0, );
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapR, (int)TextureWrapMode.MirroredRepeat);

            GL.TexImage3D(TextureTarget.Texture3D, 0,
                PixelInternalFormat.Rgba16f,
                180, 180, 180, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.BindImageTexture(0, rcpos, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);


            GL.BindTexture(TextureTarget.Texture3D, rcfog);
            // GL.BindImageTexture(0, rcpos, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            // GL.BindImageTexture(0, rcpos, 0, );
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapR, (int)TextureWrapMode.MirroredRepeat);

            GL.TexImage3D(TextureTarget.Texture3D, 0,
                PixelInternalFormat.Rgba16f,
                180, 180, 180, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.BindImageTexture(0, rcfog, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);




            GL.BindTexture(TextureTarget.Texture3D, rcnorm);
            // GL.BindImageTexture(0, rcpos, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            // GL.BindImageTexture(0, rcpos, 0, );
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapR, (int)TextureWrapMode.MirroredRepeat);

            GL.TexImage3D(TextureTarget.Texture3D, 0,
                PixelInternalFormat.Rgba16f,
                180, 180, 180, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.BindImageTexture(0, rcnorm, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);


            GL.BindTexture(TextureTarget.Texture3D, rcrad);
            // GL.BindImageTexture(0, rcpos, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            // GL.BindImageTexture(0, rcpos, 0, );
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture3D, TextureParameterName.TextureWrapR, (int)TextureWrapMode.MirroredRepeat);

            GL.TexImage3D(TextureTarget.Texture3D, 0,
                PixelInternalFormat.Rgba16f,
                180, 180, 180, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.BindImageTexture(0, rcrad, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);


            // GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);



            GL.BindTexture(TextureTarget.Texture2D, colortex);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);

            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, colortexPosition);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba32f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);


            GL.BindTexture(TextureTarget.Texture2D, colortexFog);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X), (int)(size.Y), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);

            GL.BindTexture(TextureTarget.Texture2D, colortexFogPos);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X), (int)(size.Y), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);


            GL.BindTexture(TextureTarget.Texture2D, colortexFogPrev);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);

            //====================================================================

            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, colortexHoldInfo);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            //====================================================================

            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, colortexNormal);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba32f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            //====================================================================
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, colortexAlbedo);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            //====================================================================
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, colortexSecondPosition);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba32f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            //====================================================================
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, colortexReflection);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            //====================================================================
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, colortexSecondAlbedo);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X), (int)(size.Y), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);


            //====================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, skybuff);

            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, skytex, 0
                );

            DrawBuffersEnum[] attx = [DrawBuffersEnum.ColorAttachment0];
            GL.DrawBuffers(1, attx);

            //=======================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, sunbuff);

            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, suntex, 0
                );

            DrawBuffersEnum[] attxs = [DrawBuffersEnum.ColorAttachment0];
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
            DrawBuffersEnum[] att = [ DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1,
                DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3, DrawBuffersEnum.ColorAttachment4,
             DrawBuffersEnum.ColorAttachment5, DrawBuffersEnum.ColorAttachment6];
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
            DrawBuffersEnum[] atts = [ DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1,
                DrawBuffersEnum.ColorAttachment2,  DrawBuffersEnum.ColorAttachment3
            ,  DrawBuffersEnum.ColorAttachment4,  DrawBuffersEnum.ColorAttachment5];
            GL.DrawBuffers(6, atts);

            /*
                        private Shader _TemporalRestirShader;
                    int temporalbuff;
                    int temporalWeigths;
                    int temporalOutgoingRadiance;
                    int temporalPosition;

                    private Shader _SpatialRestirShader;
                    int spatialbuff;
                    int spatialWeigths;
                    int SpatialOutgoingRadiance;
            */
            temporalWeigths = GL.GenTexture();
            temporalOutgoingRadiance = GL.GenTexture();
            temporalPosition = GL.GenTexture();
            temporalPositionFog = GL.GenTexture();
            temporalWeigthsFog = GL.GenTexture();
            temporalOutgoingRadianceFog = GL.GenTexture();
            temporalbuff = GL.GenFramebuffer();
            prevPosition = GL.GenTexture();

            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, temporalWeigths);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);

            GL.BindTexture(TextureTarget.Texture2D, temporalWeigthsFog);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);

            GL.BindTexture(TextureTarget.Texture2D, temporalOutgoingRadianceFog);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);


            GL.BindTexture(TextureTarget.Texture2D, prevPosition);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba32f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);

            GL.BindTexture(TextureTarget.Texture2D, prevSecondPosition);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba32f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);

            //====================================================================
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, temporalOutgoingRadiance);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            //====================================================================
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, temporalPosition);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba32f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);


            GL.BindTexture(TextureTarget.Texture2D, temporalPositionFog);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
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
            DrawBuffersEnum[] att2 = [ DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2
            , DrawBuffersEnum.ColorAttachment3, DrawBuffersEnum.ColorAttachment4, DrawBuffersEnum.ColorAttachment5];
            GL.DrawBuffers(6, att2);

            //private Shader _swapShader;



            spatialWeigths = GL.GenTexture();
            spatialOutgoingRadiance = GL.GenTexture();

            spatialWeigthsFog = GL.GenTexture();
            spatialOutgoingRadianceFog = GL.GenTexture();
            spatialbuff = GL.GenFramebuffer();

            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, spatialWeigths);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);

            GL.BindTexture(TextureTarget.Texture2D, spatialWeigthsFog);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);

            GL.BindTexture(TextureTarget.Texture2D, spatialOutgoingRadianceFog);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);

            //====================================================================
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, spatialOutgoingRadiance);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            //====================================================================


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
            DrawBuffersEnum[] att4 = [DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3];
            GL.DrawBuffers(4, att4);

            /*
             private Shader _TAAShader;
        int TAAbuff;
        int TAA; 
              */
            TAA = GL.GenTexture();
            TAAbuff = GL.GenFramebuffer();

            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, TAA);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
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
            DrawBuffersEnum[] att10 = [DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3];
            GL.DrawBuffers(4, att10);

            /*
             private Shader _SpatialRestirShader;
        int spatialbuff;
        int spatialWeigths;
        int spatialOutgoingRadiance; 
             */
            /*
             private Shader _upscale;
        int upscale;
        int upscaleBuff;
             */

            GL.BindTexture(TextureTarget.Texture2D, upscale);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.NearestMipmapLinear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * KeepScale), (int)(size.Y * KeepScale), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            GL.BindTexture(TextureTarget.Texture2D, upscale2);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.NearestMipmapLinear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            //====================================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, upscaleBuff);
            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer,
                FramebufferAttachment.ColorAttachment0,
                TextureTarget.Texture2D, upscale, 0
                );
            DrawBuffersEnum[] att11 = [DrawBuffersEnum.ColorAttachment0];
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
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, prevTemporalWeigths);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            //====================================================================
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, prevTemporalOutgoingRadiance);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            //====================================================================
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, prevTemporalPosition);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            //====================================================================
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, prevSpatialWeigths);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            //====================================================================
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, prevSpatialOutgoingRadiance);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)size.X, (int)size.Y, 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            //====================================================================
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, prevNormalDepth);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba32f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);


            //prevPosition
            //====================================================================
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, prevAcc);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, prevTAA);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.LinearMipmapLinear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
            //====================================================================
            /*
             prevTemporalWeigths = GL.GenTexture();
            prevTemporalOutgoingRadiance = GL.GenTexture();
            prevTemporalPosition = GL.GenTexture();
            prevSpatialWeigths = GL.GenTexture();
            prevSpatialOutgoingRadiance = GL.GenTexture();
            swapbuff = GL.GenFramebuffer();
             */
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
            DrawBuffersEnum[] att3 = [DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3];
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

            DrawBuffersEnum[] att3s = [ DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2,
                DrawBuffersEnum.ColorAttachment3, DrawBuffersEnum.ColorAttachment4];
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
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, den1);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);

            GL.BindTexture(TextureTarget.Texture2D, den2);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);


            GL.BindTexture(TextureTarget.Texture2D, var1);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);


            GL.BindTexture(TextureTarget.Texture2D, var2);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);


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
            DrawBuffersEnum[] att8 = [DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3];
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
            DrawBuffersEnum[] att9 = [DrawBuffersEnum.ColorAttachment0, DrawBuffersEnum.ColorAttachment1, DrawBuffersEnum.ColorAttachment2, DrawBuffersEnum.ColorAttachment3];
            GL.DrawBuffers(4, att9);



            //==========================================================-=
            tempAccum = GL.GenTexture();
            tempAccumbuff = GL.GenFramebuffer();
            //====================================================================
            GL.BindTexture(TextureTarget.Texture2D, tempAccum);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.MirroredRepeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.MirroredRepeat);
            GL.TexImage2D(TextureTarget.Texture2D, 0,
                PixelInternalFormat.Rgba16f,
                (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything), 0, PixelFormat.Rgba,
                PixelType.UnsignedByte, IntPtr.Zero);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);
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
            _initShader.SetInt("rcpos", 0);
            _initShader.SetInt("rcnorm", 1);
            _initShader.SetInt("skytex", 2);

            _skyShader.Use();
            _skyShader.SetInt("skyprev", 0);
            _skyShader.SetInt("worl", 1);
            _skyShader.SetInt("suns", 2);

            _sunShader.Use();
            _sunShader.SetInt("sunprev", 0);
            _sunShader.SetInt("worl", 1);



            _RC.Use();
            _RC.SetInt("rcpos", 0);
            _RC.SetInt("rcnorm", 1);
            _RC.SetInt("rcrad", 2);
            _RC.SetInt("rcfog", 3);
            _RC.SetInt("skytex", 4);
            _RC.SetInt("sunTex", 5);


            _testShader.Use();
            _testShader.SetInt("position", 0);
            _testShader.SetInt("normal", 1);
            _testShader.SetInt("albedo", 2);
            _testShader.SetInt("holdinfo", 3);
            _testShader.SetInt("colorfog", 4);
            _testShader.SetInt("rcrad", 5);
            _testShader.SetInt("skytex", 6);
            _testShader.SetInt("sunTex", 7);
            _testShader.SetInt("watpos", 8);
            _testShader.SetInt("watnorm", 9);



            // _finalShader.SetInt("color", 0);
            // _finalShader.SetInt("position", 1);
            // _finalShader.SetInt("normal", 2);
            // _finalShader.SetInt("albedo", 3);

            //_testShader.SetInt("color", 0);
            //_testShader.SetInt("depth", 1);
            //_testShader.SetInt("norm", 2);
            //_testShader.SetInt("posit", 3);

            _TemporalRestirShader.Use();
            _TemporalRestirShader.SetInt("color", 0);
            _TemporalRestirShader.SetInt("position", 1);
            _TemporalRestirShader.SetInt("normal", 2);
            _TemporalRestirShader.SetInt("albedo", 3);
            _TemporalRestirShader.SetInt("secondpos", 4);
            _TemporalRestirShader.SetInt("prevW", 5);
            _TemporalRestirShader.SetInt("prevL", 6);
            _TemporalRestirShader.SetInt("prevP", 7);
            _TemporalRestirShader.SetInt("prevN", 8);
            _TemporalRestirShader.SetInt("reflAlb", 9);
            _TemporalRestirShader.SetInt("prevPosition", 10);
            _TemporalRestirShader.SetInt("prevSecondPosition", 11);
            _TemporalRestirShader.SetInt("tempFog", 12);
            _TemporalRestirShader.SetInt("LOFog", 13);
            _TemporalRestirShader.SetInt("fog", 14);
            _TemporalRestirShader.SetInt("fogsecpos", 15);
            _TemporalRestirShader.SetInt("tempfogpos", 16);

            //temporalPositionFog

            //temporalWeigthsFog
            //prevSecondPosition
            _SpatialRestirShader.Use();
            _SpatialRestirShader.SetInt("color", 0);
            _SpatialRestirShader.SetInt("position", 1);
            _SpatialRestirShader.SetInt("normal", 2);
            _SpatialRestirShader.SetInt("albedo", 3);
            _SpatialRestirShader.SetInt("secondpos", 4);
            _SpatialRestirShader.SetInt("temppos", 5);
            _SpatialRestirShader.SetInt("prevW", 6);
            _SpatialRestirShader.SetInt("prevL", 7);
            _SpatialRestirShader.SetInt("tempW", 8);
            _SpatialRestirShader.SetInt("tempL", 9);
            _SpatialRestirShader.SetInt("prevN", 10);
            _SpatialRestirShader.SetInt("reflAlb", 11);
            _SpatialRestirShader.SetInt("prevPosition", 12);
            _SpatialRestirShader.SetInt("prevSecondPosition", 13);

            _SpatialRestirShader.SetInt("tempfog", 14);
            _SpatialRestirShader.SetInt("tempLofog", 15);
            _SpatialRestirShader.SetInt("temppos2", 16);
            _SpatialRestirShader.SetInt("wightfog", 17);
            _SpatialRestirShader.SetInt("Lofog", 18);
            _SpatialRestirShader.SetInt("fosecg", 19);



            _tempAccumShader.Use();
            _tempAccumShader.SetInt("color", 0);
            _tempAccumShader.SetInt("position", 1);
            _tempAccumShader.SetInt("normal", 2);
            _tempAccumShader.SetInt("albedo", 3);
            _tempAccumShader.SetInt("secondpos", 4);
            _tempAccumShader.SetInt("weigth", 5);
            _tempAccumShader.SetInt("outgoingr", 6);
            _tempAccumShader.SetInt("weightS", 7);
            _tempAccumShader.SetInt("outgoingrS", 8);
            _tempAccumShader.SetInt("prevN", 9);
            _tempAccumShader.SetInt("prevAcc", 10);
            _tempAccumShader.SetInt("reflAlb", 11);
            _tempAccumShader.SetInt("prevPosition", 12);
            _tempAccumShader.SetInt("prevSecondPosition", 13);
            _tempAccumShader.SetInt("prevvar", 14);
            _tempAccumShader.SetInt("prevden", 15);

            _denoiseShader.Use();
            _denoiseShader.SetInt("color", 0);
            _denoiseShader.SetInt("position", 1);
            _denoiseShader.SetInt("normal", 2);
            _denoiseShader.SetInt("albedo", 3);
            _denoiseShader.SetInt("secondpos", 4);
            _denoiseShader.SetInt("weigth", 5);
            _denoiseShader.SetInt("outgoingr", 6);
            _denoiseShader.SetInt("weightS", 7);
            _denoiseShader.SetInt("outgoingrS", 8);
            _denoiseShader.SetInt("prevN", 9);
            _denoiseShader.SetInt("ACC", 10);
            _denoiseShader.SetInt("den1", 11);
            _denoiseShader.SetInt("var1", 12);
            _denoiseShader.SetInt("reflN", 13);
            _denoiseShader.SetInt("reflectionAlb", 14);
            _denoiseShader.SetInt("colorfog", 15);
            _denoiseShader.SetInt("holdinfo", 16);
            _denoiseShader.SetInt("watpos", 17);
            _denoiseShader.SetInt("watnorm", 18);
            //_TAAShader

            _TAAShader.Use();
            _TAAShader.SetInt("rcpos", 0);
            _TAAShader.SetInt("position", 1);
            _TAAShader.SetInt("normal", 2);
            _TAAShader.SetInt("albedo", 3);
            _TAAShader.SetInt("secondpos", 4);
            _TAAShader.SetInt("weigth", 5);
            _TAAShader.SetInt("outgoingr", 6);
            _TAAShader.SetInt("weightS", 7);
            _TAAShader.SetInt("outgoingrS", 8);
            _TAAShader.SetInt("Acc", 9);
            _TAAShader.SetInt("den1", 10);
            _TAAShader.SetInt("var1", 11);
            _TAAShader.SetInt("prevTAA", 12);
            _TAAShader.SetInt("reflAlb", 13);
            _TAAShader.SetInt("inf", 14);
            _TAAShader.SetInt("colorfog", 15);
            _TAAShader.SetInt("reflnorm", 16);
            //colortexFog
            _TAAShader.SetInt("colortexFog", 17);
            _TAAShader.SetInt("colortexFogP", 18);
            _TAAShader.SetInt("spatfog", 19);
            _TAAShader.SetInt("spatfogLO", 20);
            _TAAShader.SetInt("holdinfo", 21);
            _TAAShader.SetInt("sunTex", 22);
            _TAAShader.SetInt("skyTex", 23);
            _TAAShader.SetInt("watpos", 24);
            _TAAShader.SetInt("watnorm", 25);
            _TAAShader.SetInt("worl", 26);

            //uniform sampler2D TAA;
            //uniform sampler2D upbefore;
            _upscale.Use();
            _upscale.SetInt("TAA", 0);
            _upscale.SetInt("upbefore", 1);
            _upscale.SetInt("position", 2);
            _upscale.SetInt("normal", 3);
            _upscale.SetInt("prevN", 4);
            _upscale.SetInt("albedo", 5);


            _finalShader.Use();
            _finalShader.SetInt("rcpos", 0);
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
            _finalShader.SetInt("den1", 11);
            _finalShader.SetInt("var1", 12);
            _finalShader.SetInt("TAA", 13);
            _finalShader.SetInt("colorfog", 14);
            _finalShader.SetInt("colortexFog", 15);
            _finalShader.SetInt("fogw", 16);
            _finalShader.SetInt("fogLO", 17);
            _finalShader.SetInt("sunTex", 18);
            _finalShader.SetInt("watpos", 19);

            _swapShader.Use();
            _swapShader.SetInt("tempWeights", 0);
            _swapShader.SetInt("tempOutL", 1);
            _swapShader.SetInt("tempPos", 2);
            _swapShader.SetInt("spatWe", 3);
            _swapShader.SetInt("spatLO", 4);
            _swapShader.SetInt("position", 5);
            _swapShader.SetInt("normal", 6);
            _swapShader.SetInt("Acc", 7);
            _swapShader.SetInt("TAA", 8);
            _swapShader.SetInt("up", 9);


            _swapShader2.Use();
            _swapShader2.SetInt("tempWeights", 0);
            _swapShader2.SetInt("tempOutL", 1);
            _swapShader2.SetInt("tempPos", 2);
            _swapShader2.SetInt("spatWe", 3);
            _swapShader2.SetInt("spatLO", 4);
            _swapShader2.SetInt("position", 5);
            _swapShader2.SetInt("normal", 6);
            _swapShader2.SetInt("Acc", 7);
            _swapShader2.SetInt("TAA", 8);


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



            _worl.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
          
            GL.ActiveTexture(TextureUnit.Texture0);
            // GL.BindTexture(TextureTarget.Texture3D, rcpos);
            GL.BindImageTexture(0, worley, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

          //  DebugCallback();


            GL.DispatchCompute(64, 64, 64);

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
            Matrix4 lightproj = Matrix4.CreateOrthographic(200.0f, 200.0f, 0.0f, 127.5f);

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
            // _testShader.SetMatrix4(model, "model");
            _initShader.SetMatrix4(view, "view");
            _initShader.SetMatrix4(projection, "projection");
            _initShader.SetMatrix4(view.Inverted(), "invview");
            _initShader.SetMatrix4(projection.Inverted(), "invproj");

            _initShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _initShader.SetFloat(iFrame, "time");
            _initShader.SetFloat(stopWatch.ElapsedMilliseconds, "time2");

            _initShader.SetVector3(_camera.Position, "viewPos");
            _initShader.SetVector3(prevCamPos, "lastViewPos");

            _initShader.SetVector3(ldir, "ldir");

            GL.ActiveTexture(TextureUnit.Texture0);
            // GL.BindTexture(TextureTarget.Texture3D, rcpos);
            GL.BindImageTexture(0, rcpos, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            GL.ActiveTexture(TextureUnit.Texture1);
            // GL.BindTexture(TextureTarget.Texture3D, rcpos);
            GL.BindImageTexture(1, rcnorm, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            GL.ActiveTexture(TextureUnit.Texture2);
            GL.BindTexture(TextureTarget.Texture2D, skytex);

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
            _skyShader.SetVector3(_camera.Position, "viewPos");
            _skyShader.SetVector3(lpos2, "lpos2");

            _skyShader.SetMatrix4(lightproj2, "lightproj2");
            _skyShader.SetMatrix4(lightview2, "lightview2");

            GL.BindVertexArray(quadVAO);

            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, skytex);
            GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);

            GL.ActiveTexture(TextureUnit.Texture1);
            GL.BindTexture(TextureTarget.Texture3D, worley);
            GL.ActiveTexture(TextureUnit.Texture2);
            GL.BindTexture(TextureTarget.Texture2D, suntex);
            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);


            //=================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, sunbuff);


            GL.Viewport(0, 0, (int)shadowScale, (int)shadowScale);
            //GL.ClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            //GL.Clear(ClearBufferMask.ColorBufferBit);
            GL.Disable(EnableCap.DepthTest);
            _sunShader.Use();
            _sunShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _sunShader.SetFloat(iFrame, "time");
            _sunShader.SetVector3(ldir, "ldir");

            _sunShader.SetVector3(lpos, "lpos");

            _sunShader.SetMatrix4(lightproj, "lightproj");
            _sunShader.SetMatrix4(lightview, "lightview");


            _sunShader.SetMatrix4(lightview.Inverted(), "invview");
            _sunShader.SetMatrix4(lightproj.Inverted(), "invproj");
            _sunShader.SetFloat((float)shadowScale, "scale");
            _sunShader.SetVector3(_camera.Position, "viewPos");

            _sunShader.SetMatrix4(lightview2.Inverted(), "invview2");
            _sunShader.SetMatrix4(lightproj2.Inverted(), "invproj2");




            _sunShader.SetVector3(lpos2, "lpos2");

            _sunShader.SetMatrix4(lightproj2, "lightproj2");
            _sunShader.SetMatrix4(lightview2, "lightview2");
            GL.BindVertexArray(quadVAO);

            //GL.ActiveTexture(TextureUnit.Texture0);
            //GL.BindTexture(TextureTarget.Texture2D, suntex);

            GL.ActiveTexture(TextureUnit.Texture0);
            // GL.BindTexture(TextureTarget.Texture3D, rcpos);
            GL.BindImageTexture(0, suntex, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            GL.ActiveTexture(TextureUnit.Texture1);
            GL.BindTexture(TextureTarget.Texture3D, worley);

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);



            //===================================================

            _RC.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _RC.SetMatrix4(view, "view");
            _RC.SetMatrix4(projection, "projection");
            _RC.SetMatrix4(view.Inverted(), "invview");
            _RC.SetMatrix4(projection.Inverted(), "invproj");

            _RC.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y* KeepScale), "wh");
            _RC.SetFloat(iFrame, "time");
            _RC.SetFloat(stopWatch.ElapsedMilliseconds, "time2");

            _RC.SetVector3(_camera.Position, "viewPos");
            _RC.SetVector3(prevCamPos, "lastViewPos");
            _RC.SetVector3(ldir, "ldir");
            _RC.SetMatrix4(lightproj, "lightproj");
            _RC.SetMatrix4(lightview, "lightview");
            _RC.SetVector3(lpos, "lpos");


            GL.ActiveTexture(TextureUnit.Texture0);
            // GL.BindTexture(TextureTarget.Texture3D, rcpos);
            GL.BindImageTexture(0, rcpos, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            GL.ActiveTexture(TextureUnit.Texture1);
            // GL.BindTexture(TextureTarget.Texture3D, rcpos);
            GL.BindImageTexture(1, rcnorm, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            GL.ActiveTexture(TextureUnit.Texture2);
            // GL.BindTexture(TextureTarget.Texture3D, rcpos);
            GL.BindImageTexture(2, rcrad, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            GL.ActiveTexture(TextureUnit.Texture3);
            // GL.BindTexture(TextureTarget.Texture3D, rcpos);
            GL.BindImageTexture(3, rcfog, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            GL.ActiveTexture(TextureUnit.Texture4);
            GL.BindTexture(TextureTarget.Texture2D, skytex);
            GL.ActiveTexture(TextureUnit.Texture5);
            GL.BindTexture(TextureTarget.Texture2D, suntex);


            GL.DispatchCompute(45, 45, 45);
            //GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);
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
            _testShader.SetMatrix4(view, "view");
            _testShader.SetMatrix4(projection, "projection");
            _testShader.SetMatrix4(view.Inverted(), "invview");
            _testShader.SetMatrix4(projection.Inverted(), "invproj");

            _testShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _testShader.SetFloat(iFrame, "time");
            _testShader.SetFloat(stopWatch.ElapsedMilliseconds, "time2");

            _testShader.SetVector3(_camera.Position, "viewPos");
            _testShader.SetVector3(ldir, "ldir");
            _testShader.SetMatrix4(lightproj, "lightproj");
            _testShader.SetMatrix4(lightview, "lightview");
            _testShader.SetVector3(lpos, "lpos");

            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, colortexPosition);
            GL.ActiveTexture(TextureUnit.Texture1);
            GL.BindTexture(TextureTarget.Texture2D, colortexNormal);
            GL.ActiveTexture(TextureUnit.Texture2);
            GL.BindTexture(TextureTarget.Texture2D, colortexAlbedo);
            GL.ActiveTexture(TextureUnit.Texture3);
            GL.BindTexture(TextureTarget.Texture2D, colortexHoldInfo);
            GL.ActiveTexture(TextureUnit.Texture4);
            GL.BindTexture(TextureTarget.Texture2D, colorfog);
            //GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);
            GL.ActiveTexture(TextureUnit.Texture5);
            // GL.BindTexture(TextureTarget.Texture3D, rcpos);
            GL.BindImageTexture(5, rcrad, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);
            GL.ActiveTexture(TextureUnit.Texture6);
            GL.BindTexture(TextureTarget.Texture2D, skytex);
            GL.ActiveTexture(TextureUnit.Texture7);
            GL.BindTexture(TextureTarget.Texture2D, suntex);
            GL.ActiveTexture(TextureUnit.Texture8);
            GL.BindTexture(TextureTarget.Texture2D, watpos);
            GL.ActiveTexture(TextureUnit.Texture9);
            GL.BindTexture(TextureTarget.Texture2D, watnorm);

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);


            ///==========================================================
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, temporalbuff);


            GL.Viewport(0, 0, (int)size.X, (int)size.Y);
            GL.Disable(EnableCap.DepthTest);
            _TemporalRestirShader.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _TemporalRestirShader.SetMatrix4(view, "view");
            _TemporalRestirShader.SetMatrix4(projection, "projection");
            _TemporalRestirShader.SetMatrix4(view.Inverted(), "invview");
            _TemporalRestirShader.SetMatrix4(projection.Inverted(), "invproj");
            _TemporalRestirShader.SetMatrix4(prevView, "prevview");
            _TemporalRestirShader.SetMatrix4(prevProjection, "prevproj");

            _TemporalRestirShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _TemporalRestirShader.SetFloat(iFrame, "time");
            _TemporalRestirShader.SetVector3(_camera.Position, "viewPos");
            _TemporalRestirShader.SetVector3(prevCamPos, "lastViewPos");
            _TemporalRestirShader.SetVector3(ldir, "ldir");

            /*
              _TemporalRestirShader.SetInt("prevW", 5);
            _TemporalRestirShader.SetInt("prevL", 6);
            _TemporalRestirShader.SetInt("prevP", 7);
             */
            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, colortex);
            GL.ActiveTexture(TextureUnit.Texture1);
            GL.BindTexture(TextureTarget.Texture2D, colortexPosition);
            GL.ActiveTexture(TextureUnit.Texture2);
            GL.BindTexture(TextureTarget.Texture2D, colortexNormal);
            GL.ActiveTexture(TextureUnit.Texture3);
            GL.BindTexture(TextureTarget.Texture2D, colortexAlbedo);
            GL.ActiveTexture(TextureUnit.Texture4);
            GL.BindTexture(TextureTarget.Texture2D, colortexSecondPosition);

            GL.ActiveTexture(TextureUnit.Texture5);
            GL.BindTexture(TextureTarget.Texture2D, prevTemporalWeigths);
            GL.ActiveTexture(TextureUnit.Texture6);
            GL.BindTexture(TextureTarget.Texture2D, prevTemporalOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture7);
            GL.BindTexture(TextureTarget.Texture2D, prevTemporalPosition);
            GL.ActiveTexture(TextureUnit.Texture8);
            GL.BindTexture(TextureTarget.Texture2D, prevNormalDepth);
            GL.ActiveTexture(TextureUnit.Texture9);
            GL.BindTexture(TextureTarget.Texture2D, colortexSecondAlbedo);
            GL.ActiveTexture(TextureUnit.Texture10);
            GL.BindTexture(TextureTarget.Texture2D, prevPosition);
            GL.ActiveTexture(TextureUnit.Texture11);
            GL.BindTexture(TextureTarget.Texture2D, prevSecondPosition);
            GL.ActiveTexture(TextureUnit.Texture12);
            GL.BindTexture(TextureTarget.Texture2D, temporalWeigthsFog);
            GL.ActiveTexture(TextureUnit.Texture13);
            GL.BindTexture(TextureTarget.Texture2D, temporalOutgoingRadianceFog);
            GL.ActiveTexture(TextureUnit.Texture14);
            GL.BindTexture(TextureTarget.Texture2D, colortexFog);
            GL.ActiveTexture(TextureUnit.Texture15);
            GL.BindTexture(TextureTarget.Texture2D, colortexFogPos);
            //temporalPositionFog
            GL.ActiveTexture(TextureUnit.Texture15);
            GL.BindTexture(TextureTarget.Texture2D, temporalPositionFog);
            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            ///==========================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, spatialbuff);


            GL.Viewport(0, 0, (int)size.X, (int)size.Y);
            GL.Disable(EnableCap.DepthTest);
            _SpatialRestirShader.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _SpatialRestirShader.SetMatrix4(view, "view");
            _SpatialRestirShader.SetMatrix4(projection, "projection");
            _SpatialRestirShader.SetMatrix4(view.Inverted(), "invview");
            _SpatialRestirShader.SetMatrix4(projection.Inverted(), "invproj");
            _SpatialRestirShader.SetMatrix4(prevView, "prevview");
            _SpatialRestirShader.SetMatrix4(prevProjection, "prevproj");

            _SpatialRestirShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _SpatialRestirShader.SetFloat(iFrame, "time");
            _SpatialRestirShader.SetFloat(stopWatch.ElapsedMilliseconds, "time2");

            _SpatialRestirShader.SetVector3(_camera.Position, "viewPos");
            _SpatialRestirShader.SetVector3(prevCamPos, "lastViewPos");
            _SpatialRestirShader.SetVector3(ldir, "ldir");

            /*
              _TemporalRestirShader.SetInt("prevW", 5);
            _TemporalRestirShader.SetInt("prevL", 6);
            _TemporalRestirShader.SetInt("prevP", 7);
             */
            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, colortex);
            GL.ActiveTexture(TextureUnit.Texture1);
            GL.BindTexture(TextureTarget.Texture2D, colortexPosition);
            GL.ActiveTexture(TextureUnit.Texture2);
            GL.BindTexture(TextureTarget.Texture2D, colortexNormal);
            GL.ActiveTexture(TextureUnit.Texture3);
            GL.BindTexture(TextureTarget.Texture2D, colortexAlbedo);
            GL.ActiveTexture(TextureUnit.Texture4);
            GL.BindTexture(TextureTarget.Texture2D, colortexSecondPosition);

            GL.ActiveTexture(TextureUnit.Texture5);
            GL.BindTexture(TextureTarget.Texture2D, temporalPosition);
            GL.ActiveTexture(TextureUnit.Texture6);
            GL.BindTexture(TextureTarget.Texture2D, prevSpatialWeigths);
            GL.ActiveTexture(TextureUnit.Texture7);
            GL.BindTexture(TextureTarget.Texture2D, prevSpatialOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture8);
            GL.BindTexture(TextureTarget.Texture2D, temporalWeigths);
            GL.ActiveTexture(TextureUnit.Texture9);
            GL.BindTexture(TextureTarget.Texture2D, temporalOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture10);
            GL.BindTexture(TextureTarget.Texture2D, prevNormalDepth);
            GL.ActiveTexture(TextureUnit.Texture11);
            GL.BindTexture(TextureTarget.Texture2D, colortexSecondAlbedo);
            GL.ActiveTexture(TextureUnit.Texture12);
            GL.BindTexture(TextureTarget.Texture2D, prevPosition);
            GL.ActiveTexture(TextureUnit.Texture13);
            GL.BindTexture(TextureTarget.Texture2D, prevSecondPosition);

            GL.ActiveTexture(TextureUnit.Texture14);
            GL.BindTexture(TextureTarget.Texture2D, temporalWeigthsFog);
            GL.ActiveTexture(TextureUnit.Texture15);
            GL.BindTexture(TextureTarget.Texture2D, temporalOutgoingRadianceFog);
            GL.ActiveTexture(TextureUnit.Texture16);
            GL.BindTexture(TextureTarget.Texture2D, temporalPositionFog);
            GL.ActiveTexture(TextureUnit.Texture17);
            GL.BindTexture(TextureTarget.Texture2D, spatialWeigthsFog);
            GL.ActiveTexture(TextureUnit.Texture18);
            GL.BindTexture(TextureTarget.Texture2D, spatialOutgoingRadianceFog);
            GL.ActiveTexture(TextureUnit.Texture19);
            GL.BindTexture(TextureTarget.Texture2D, colortexFogPos);
            /*
              _SpatialRestirShader.SetInt("tempfog", 14);
            _SpatialRestirShader.SetInt("tempLofog", 15);
            _SpatialRestirShader.SetInt("temppos", 16);
            _SpatialRestirShader.SetInt("wightfog", 17);
            _SpatialRestirShader.SetInt("Lofog", 18);
             */


            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            ///==========================================================

            /*
             _tempAccumShader.Use();
            _tempAccumShader.SetInt("color", 0);
            _tempAccumShader.SetInt("position", 1);
            _tempAccumShader.SetInt("normal", 2);
            _tempAccumShader.SetInt("albedo", 3);
            _tempAccumShader.SetInt("secondpos", 4);
            _tempAccumShader.SetInt("weigth", 5);
            _tempAccumShader.SetInt("outgoingr", 6);
            _tempAccumShader.SetInt("weightS", 7);
            _tempAccumShader.SetInt("outgoingrS", 8);
            _tempAccumShader.SetInt("prevN", 9);

             */

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, tempAccumbuff);


            GL.Viewport(0, 0, (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything));
            GL.Disable(EnableCap.DepthTest);
            _tempAccumShader.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _tempAccumShader.SetMatrix4(view, "view");
            _tempAccumShader.SetMatrix4(projection, "projection");
            _tempAccumShader.SetMatrix4(view.Inverted(), "invview");
            _tempAccumShader.SetMatrix4(projection.Inverted(), "invproj");
            _tempAccumShader.SetMatrix4(prevView, "prevview");
            _tempAccumShader.SetMatrix4(prevProjection, "prevproj");

            _tempAccumShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _tempAccumShader.SetFloat(iFrame, "time");
            _tempAccumShader.SetVector3(_camera.Position, "viewPos");
            _tempAccumShader.SetVector3(prevCamPos, "lastViewPos");

            _tempAccumShader.SetVector3(ldir, "ldir");

            /*
              _TemporalRestirShader.SetInt("prevW", 5);
            _TemporalRestirShader.SetInt("prevL", 6);
            _TemporalRestirShader.SetInt("prevP", 7);
             */
            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, colortex);
            GL.ActiveTexture(TextureUnit.Texture1);
            GL.BindTexture(TextureTarget.Texture2D, colortexPosition);
            GL.ActiveTexture(TextureUnit.Texture2);
            GL.BindTexture(TextureTarget.Texture2D, colortexNormal);
            GL.ActiveTexture(TextureUnit.Texture3);
            GL.BindTexture(TextureTarget.Texture2D, colortexAlbedo);
            GL.ActiveTexture(TextureUnit.Texture4);
            GL.BindTexture(TextureTarget.Texture2D, colortexSecondPosition);
            GL.ActiveTexture(TextureUnit.Texture5);
            GL.BindTexture(TextureTarget.Texture2D, temporalWeigths);
            GL.ActiveTexture(TextureUnit.Texture6);
            GL.BindTexture(TextureTarget.Texture2D, temporalOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture7);
            GL.BindTexture(TextureTarget.Texture2D, spatialWeigths);
            GL.ActiveTexture(TextureUnit.Texture8);
            GL.BindTexture(TextureTarget.Texture2D, spatialOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture9);
            GL.BindTexture(TextureTarget.Texture2D, prevNormalDepth);
            GL.ActiveTexture(TextureUnit.Texture10);
            GL.BindTexture(TextureTarget.Texture2D, prevAcc);
            GL.ActiveTexture(TextureUnit.Texture11);
            GL.BindTexture(TextureTarget.Texture2D, colortexSecondAlbedo);
            GL.ActiveTexture(TextureUnit.Texture12);
            GL.BindTexture(TextureTarget.Texture2D, prevPosition);
            GL.ActiveTexture(TextureUnit.Texture13);
            GL.BindTexture(TextureTarget.Texture2D, prevSecondPosition);
            GL.ActiveTexture(TextureUnit.Texture14);
            GL.BindTexture(TextureTarget.Texture2D, var2);
            GL.ActiveTexture(TextureUnit.Texture15);
            GL.BindTexture(TextureTarget.Texture2D, den2);


            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);
            GL.MemoryBarrier(MemoryBarrierFlags.AllBarrierBits);

            ///==========================================================

            for (int i = 0; i < 7; i++)
            {
                if (i % 2 == 0)
                {
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, den1buff);
                }
                else
                {
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, den2buff);
                }

                GL.Viewport(0, 0, (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything));
                GL.Disable(EnableCap.DepthTest);
                _denoiseShader.Use();

                GL.BindVertexArray(quadVAO);
                // _testShader.SetMatrix4(model, "model");
                _denoiseShader.SetMatrix4(view, "view");
                _denoiseShader.SetMatrix4(projection, "projection");
                _denoiseShader.SetMatrix4(view.Inverted(), "invview");
                _denoiseShader.SetMatrix4(projection.Inverted(), "invproj");
                _denoiseShader.SetMatrix4(prevView, "prevview");
                _denoiseShader.SetMatrix4(prevProjection, "prevproj");

                _denoiseShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
                _denoiseShader.SetFloat(iFrame, "time");
                _denoiseShader.SetVector3(_camera.Position, "viewPos");
                _denoiseShader.SetVector3(prevCamPos, "lastViewPos");
                _denoiseShader.SetVector3(ldir, "ldir");


                GL.ActiveTexture(TextureUnit.Texture0);
                GL.BindTexture(TextureTarget.Texture2D, colortex);
                GL.ActiveTexture(TextureUnit.Texture1);
                GL.BindTexture(TextureTarget.Texture2D, colortexPosition);
                GL.ActiveTexture(TextureUnit.Texture2);
                GL.BindTexture(TextureTarget.Texture2D, colortexNormal);
                GL.ActiveTexture(TextureUnit.Texture3);
                GL.BindTexture(TextureTarget.Texture2D, colortexAlbedo);
                GL.ActiveTexture(TextureUnit.Texture4);
                GL.BindTexture(TextureTarget.Texture2D, colortexSecondPosition);
                GL.ActiveTexture(TextureUnit.Texture5);
                GL.BindTexture(TextureTarget.Texture2D, temporalWeigths);
                GL.ActiveTexture(TextureUnit.Texture6);
                GL.BindTexture(TextureTarget.Texture2D, temporalOutgoingRadiance);
                GL.ActiveTexture(TextureUnit.Texture7);
                GL.BindTexture(TextureTarget.Texture2D, spatialWeigths);
                GL.ActiveTexture(TextureUnit.Texture8);
                GL.BindTexture(TextureTarget.Texture2D, spatialOutgoingRadiance);
                GL.ActiveTexture(TextureUnit.Texture9);
                GL.BindTexture(TextureTarget.Texture2D, prevNormalDepth);
                GL.ActiveTexture(TextureUnit.Texture10);
                GL.BindTexture(TextureTarget.Texture2D, tempAccum);
                if (i % 2 == 0)
                {
                    GL.ActiveTexture(TextureUnit.Texture11);
                    GL.BindTexture(TextureTarget.Texture2D, den1);
                    GL.ActiveTexture(TextureUnit.Texture12);
                    GL.BindTexture(TextureTarget.Texture2D, var1);
                }
                else
                {
                    GL.ActiveTexture(TextureUnit.Texture11);
                    GL.BindTexture(TextureTarget.Texture2D, den2);
                    GL.ActiveTexture(TextureUnit.Texture12);
                    GL.BindTexture(TextureTarget.Texture2D, var2);
                }
                GL.ActiveTexture(TextureUnit.Texture13);
                GL.BindTexture(TextureTarget.Texture2D, colortexReflection);
                GL.ActiveTexture(TextureUnit.Texture14);
                GL.BindTexture(TextureTarget.Texture2D, colortexSecondAlbedo);


                if (i % 2 == 0)
                {
                    GL.ActiveTexture(TextureUnit.Texture15);
                    GL.BindTexture(TextureTarget.Texture2D, colorfog);
                }
                else
                {
                    GL.ActiveTexture(TextureUnit.Texture15);
                    GL.BindTexture(TextureTarget.Texture2D, colorfog2);
                }
                GL.ActiveTexture(TextureUnit.Texture16);
                GL.BindTexture(TextureTarget.Texture2D, colortexHoldInfo);
                GL.ActiveTexture(TextureUnit.Texture17);
                GL.BindTexture(TextureTarget.Texture2D, watpos);
                GL.ActiveTexture(TextureUnit.Texture18);
                GL.BindTexture(TextureTarget.Texture2D, watnorm);
                GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);
                GL.MemoryBarrier(MemoryBarrierFlags.AllBarrierBits);
            }
            ///==========================================================

            GL.BindFramebuffer(FramebufferTarget.Framebuffer, TAAbuff);


            GL.Viewport(0, 0, (int)(size.X * ScaleEverything), (int)(size.Y * ScaleEverything));
            GL.Disable(EnableCap.DepthTest);
            _TAAShader.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _TAAShader.SetMatrix4(view, "view");
            _TAAShader.SetMatrix4(projection, "projection");
            _TAAShader.SetMatrix4(view.Inverted(), "invview");
            _TAAShader.SetMatrix4(projection.Inverted(), "invproj");
            _TAAShader.SetMatrix4(prevView, "prevview");
            _TAAShader.SetMatrix4(prevProjection, "prevproj");

            _TAAShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _TAAShader.SetFloat(iFrame, "time");
            _TAAShader.SetVector3(_camera.Position, "viewPos");
            _TAAShader.SetVector3(prevCamPos, "lastViewPos");
            _TAAShader.SetVector3(ldir, "ldir");
            _TAAShader.SetVector3(lpos, "lpos");
            _TAAShader.SetMatrix4(lightproj, "lightproj");
            _TAAShader.SetMatrix4(lightview, "lightview");

            GL.ActiveTexture(TextureUnit.Texture0);
            //GL.BindTexture(TextureTarget.Texture3D, rcpos);
            GL.BindImageTexture(0, rcfog, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

            //GL.ActiveTexture(TextureUnit.Texture0);
            //GL.BindTexture(TextureTarget.Texture2D, colortex);
            GL.ActiveTexture(TextureUnit.Texture1);
            GL.BindTexture(TextureTarget.Texture2D, colortexPosition);
            GL.ActiveTexture(TextureUnit.Texture2);
            GL.BindTexture(TextureTarget.Texture2D, colortexNormal);
            GL.ActiveTexture(TextureUnit.Texture3);
            GL.BindTexture(TextureTarget.Texture2D, colortexAlbedo);
            GL.ActiveTexture(TextureUnit.Texture4);
            GL.BindTexture(TextureTarget.Texture2D, colortexSecondPosition);
            GL.ActiveTexture(TextureUnit.Texture5);
            GL.BindTexture(TextureTarget.Texture2D, temporalWeigths);
            GL.ActiveTexture(TextureUnit.Texture6);
            GL.BindTexture(TextureTarget.Texture2D, temporalOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture7);
            GL.BindTexture(TextureTarget.Texture2D, spatialWeigths);
            GL.ActiveTexture(TextureUnit.Texture8);
            GL.BindTexture(TextureTarget.Texture2D, spatialOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture9);
            GL.BindTexture(TextureTarget.Texture2D, tempAccum);
            GL.ActiveTexture(TextureUnit.Texture10);
            GL.BindTexture(TextureTarget.Texture2D, den2);
            GL.ActiveTexture(TextureUnit.Texture11);
            GL.BindTexture(TextureTarget.Texture2D, var2);
            GL.ActiveTexture(TextureUnit.Texture12);
            GL.BindTexture(TextureTarget.Texture2D, prevTAA);
            GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);

            GL.ActiveTexture(TextureUnit.Texture13);
            GL.BindTexture(TextureTarget.Texture2D, colortexSecondAlbedo);
            GL.ActiveTexture(TextureUnit.Texture14);
            GL.BindTexture(TextureTarget.Texture2D, colortexHoldInfo);
            GL.ActiveTexture(TextureUnit.Texture15);
            GL.BindTexture(TextureTarget.Texture2D, colorfog);
            GL.ActiveTexture(TextureUnit.Texture16);
            GL.BindTexture(TextureTarget.Texture2D, colortexReflection);
            GL.ActiveTexture(TextureUnit.Texture17);
            GL.BindTexture(TextureTarget.Texture2D, colortexFog);
            GL.ActiveTexture(TextureUnit.Texture18);
            GL.BindTexture(TextureTarget.Texture2D, colortexFogPrev);
            GL.ActiveTexture(TextureUnit.Texture19);
            GL.BindTexture(TextureTarget.Texture2D, spatialWeigthsFog) ;
            GL.ActiveTexture(TextureUnit.Texture20);
            GL.BindTexture(TextureTarget.Texture2D, spatialOutgoingRadianceFog);
            GL.ActiveTexture(TextureUnit.Texture21);
            GL.BindTexture(TextureTarget.Texture2D, colortexHoldInfo);
            GL.ActiveTexture(TextureUnit.Texture22);
            GL.BindTexture(TextureTarget.Texture2D, suntex);
            GL.ActiveTexture(TextureUnit.Texture23);
            GL.BindTexture(TextureTarget.Texture2D, skytex);
            GL.ActiveTexture(TextureUnit.Texture24);
            GL.BindTexture(TextureTarget.Texture2D, watpos);
            GL.ActiveTexture(TextureUnit.Texture25);
            GL.BindTexture(TextureTarget.Texture2D, watnorm);
            GL.ActiveTexture(TextureUnit.Texture26);
            GL.BindTexture(TextureTarget.Texture3D, worley);

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);


            /*_upscale.Use();
            _upscale.SetInt("TAA", 0);
            _upscale.SetInt("upbefore", 1);
*/


            GL.BindFramebuffer(FramebufferTarget.Framebuffer, upscaleBuff);


            GL.Viewport(0, 0, (int)(size.X * KeepScale), (int)(size.Y * KeepScale));
            GL.Disable(EnableCap.DepthTest);
            _upscale.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _upscale.SetMatrix4(view, "view");
            _upscale.SetMatrix4(projection, "projection");
            _upscale.SetMatrix4(view.Inverted(), "invview");
            _upscale.SetMatrix4(projection.Inverted(), "invproj");
            _upscale.SetMatrix4(prevView, "prevview");
            _upscale.SetMatrix4(prevProjection, "prevproj");

            _upscale.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _upscale.SetFloat(iFrame, "time");
            _upscale.SetVector3(_camera.Position, "viewPos");
            _upscale.SetVector3(prevCamPos, "lastViewPos");
            _upscale.SetVector3(ldir, "ldir");

            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, TAA);
            //glGenerateMipmap(GL_TEXTURE_2D);
            GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);
            if (!isEverythingLower) {
                GL.ActiveTexture(TextureUnit.Texture1);
                GL.BindTexture(TextureTarget.Texture2D, upscale2);
                GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);

            }
            else
            { //KeepScale
                GL.ActiveTexture(TextureUnit.Texture1);
                GL.BindTexture(TextureTarget.Texture2D, upscale);
                GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);

            }
            GL.ActiveTexture(TextureUnit.Texture2);
            GL.BindTexture(TextureTarget.Texture2D, colortexPosition);
            GL.ActiveTexture(TextureUnit.Texture3);
            GL.BindTexture(TextureTarget.Texture2D, colortexNormal);
            GL.ActiveTexture(TextureUnit.Texture4);
            GL.BindTexture(TextureTarget.Texture2D, prevNormalDepth);
            GL.ActiveTexture(TextureUnit.Texture5);
            GL.BindTexture(TextureTarget.Texture2D, colortexAlbedo);

            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);


            //=========================================================

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
            // _testShader.SetMatrix4(model, "model");
            _swapShader.SetMatrix4(view, "view");
            _swapShader.SetMatrix4(projection, "projection");
            _swapShader.SetMatrix4(view.Inverted(), "invview");
            _swapShader.SetMatrix4(projection.Inverted(), "invproj");
            _swapShader.SetMatrix4(prevView, "prevview");
            _swapShader.SetMatrix4(prevProjection, "prevproj");

            _swapShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _swapShader.SetFloat(iFrame, "time");
            _swapShader.SetVector3(_camera.Position, "viewPos");
            _swapShader.SetVector3(prevCamPos, "lastViewPos");
            _swapShader.SetVector3(ldir, "ldir");

            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, temporalWeigths);
            GL.ActiveTexture(TextureUnit.Texture1);
            GL.BindTexture(TextureTarget.Texture2D, temporalOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture2);
            GL.BindTexture(TextureTarget.Texture2D, temporalPosition);
            GL.ActiveTexture(TextureUnit.Texture3);
            GL.BindTexture(TextureTarget.Texture2D, spatialWeigths);
            GL.ActiveTexture(TextureUnit.Texture4);
            GL.BindTexture(TextureTarget.Texture2D, spatialOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture5);
            GL.BindTexture(TextureTarget.Texture2D, colortexPosition);
            GL.ActiveTexture(TextureUnit.Texture6);
            GL.BindTexture(TextureTarget.Texture2D, colortexNormal);
            GL.ActiveTexture(TextureUnit.Texture7);
            GL.BindTexture(TextureTarget.Texture2D, tempAccum);
            GL.ActiveTexture(TextureUnit.Texture8);
            GL.BindTexture(TextureTarget.Texture2D, TAA);
            GL.ActiveTexture(TextureUnit.Texture9);
            GL.BindTexture(TextureTarget.Texture2D, upscale);


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
            // _testShader.SetMatrix4(model, "model");
            _swapShader2.SetMatrix4(view, "view");
            _swapShader2.SetMatrix4(projection, "projection");
            _swapShader2.SetMatrix4(view.Inverted(), "invview");
            _swapShader2.SetMatrix4(projection.Inverted(), "invproj");
            _swapShader2.SetMatrix4(prevView, "prevview");
            _swapShader2.SetMatrix4(prevProjection, "prevproj");

            _swapShader2.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _swapShader2.SetFloat(iFrame, "time");
            _swapShader2.SetVector3(_camera.Position, "viewPos");
            _swapShader2.SetVector3(prevCamPos, "lastViewPos");
            _swapShader2.SetVector3(ldir, "ldir");

            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, temporalWeigths);
            GL.ActiveTexture(TextureUnit.Texture1);
            GL.BindTexture(TextureTarget.Texture2D, temporalOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture2);
            GL.BindTexture(TextureTarget.Texture2D, temporalPosition);
            GL.ActiveTexture(TextureUnit.Texture3);
            GL.BindTexture(TextureTarget.Texture2D, spatialWeigths);
            GL.ActiveTexture(TextureUnit.Texture4);
            GL.BindTexture(TextureTarget.Texture2D, spatialOutgoingRadiance);
            GL.ActiveTexture(TextureUnit.Texture5);
            GL.BindTexture(TextureTarget.Texture2D, colortexPosition);
            GL.ActiveTexture(TextureUnit.Texture6);
            GL.BindTexture(TextureTarget.Texture2D, colortexNormal);
            GL.ActiveTexture(TextureUnit.Texture7);
            GL.BindTexture(TextureTarget.Texture2D, tempAccum);
            GL.ActiveTexture(TextureUnit.Texture8);
            GL.BindTexture(TextureTarget.Texture2D, TAA);


            GL.DrawElements(PrimitiveType.Triangles, quadindices.Length, DrawElementsType.UnsignedInt, 0);

            //==========================================================


            GL.BindFramebuffer(FramebufferTarget.Framebuffer, TargetFramebuffer);


            GL.Viewport(0, 0, (int)(size.X * KeepScale), (int)(size.Y * KeepScale));
            GL.Disable(EnableCap.DepthTest);
            _finalShader.Use();
            GL.BindVertexArray(quadVAO);
            // _testShader.SetMatrix4(model, "model");
            _finalShader.SetMatrix4(view, "view");
            _finalShader.SetMatrix4(projection, "projection");
            _finalShader.SetMatrix4(view.Inverted(), "invview");
            _finalShader.SetMatrix4(projection.Inverted(), "invproj");

            _finalShader.SetVector2(new Vector2((float)size.X * KeepScale, (float)size.Y * KeepScale), "wh");
            _finalShader.SetFloat(iFrame, "time");
            _finalShader.SetVector3(_camera.Position, "viewPos");
            _finalShader.SetVector3(ldir, "ldir");
            _finalShader.SetMatrix4(lightproj, "lightproj");
            _finalShader.SetMatrix4(lightview, "lightview");
            _finalShader.SetVector3(lpos, "lpos");
            _finalShader.SetMatrix4(lightview.Inverted(), "linvview");
            _finalShader.SetMatrix4(lightproj.Inverted(), "linvproj");
            _finalShader.SetFloat(bright, "brightness");
            GL.ActiveTexture(TextureUnit.Texture0);
            //GL.BindTexture(TextureTarget.Texture3D, rcpos);
            GL.BindImageTexture(0, rcfog, 0, true, 0, TextureAccess.ReadWrite, SizedInternalFormat.Rgba16f);

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
            GL.ActiveTexture(TextureUnit.Texture14);
            GL.BindTexture(TextureTarget.Texture2D, colorfog2);
            //colortexFog
            GL.ActiveTexture(TextureUnit.Texture15);
            GL.BindTexture(TextureTarget.Texture2D, colortexFogPrev);
            GL.ActiveTexture(TextureUnit.Texture16);
            GL.BindTexture(TextureTarget.Texture2D, spatialWeigthsFog);
            GL.ActiveTexture(TextureUnit.Texture17);
            GL.BindTexture(TextureTarget.Texture2D, TAA);
            GL.ActiveTexture(TextureUnit.Texture18);
            GL.BindTexture(TextureTarget.Texture2D, suntex);
            GL.ActiveTexture(TextureUnit.Texture19);
            GL.BindTexture(TextureTarget.Texture2D, watpos);


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
