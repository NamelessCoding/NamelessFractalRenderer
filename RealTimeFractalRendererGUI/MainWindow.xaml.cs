using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Wpf;
using RealTimeFractalRendererLibrary;
using System;
using System.ComponentModel;
using System.Timers;
using System.Windows;
using System.Windows.Input;
using Window = System.Windows.Window;

namespace RealTimeFractalRendererGUI
{
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        private Renderer renderer = default!;
        private bool isRendererInitialized = false;
        private bool resolutionChanged = true;
        private readonly KeyboardHelper input = default!;
        private bool _firstMove = true;
        private Point _lastPos;
        private float keepDelta;
        const float cameraSpeed = 15.5f;
        const float sensitivity = 0.2f;
        private readonly Timer renderTimer = new(TimeSpan.FromSeconds(1.0/60.0));
        private bool isRendering = false;
        public event PropertyChangedEventHandler? PropertyChanged;

        private float realYaw = 0.0f;
        private float smoothedYaw = 0.0f;

        private float realPitch = 0.0f;
        private float smoothedPitch = 0.0f;


        public float Brightness { get; set; } = 1.0f;
        public float Contrast { get; set; } = 1.0f;


        public float XL { get; set; } = 0.0f;
        public float YL { get; set; } = 0.1f;
        public float ZL { get; set; } = -0.9f;

        public float CameraYaw
        {
            get => renderer?.Camera.Yaw ?? 0;
            set => renderer.Camera.Yaw = value;
        }

        public MainWindow()
        {
            InitializeComponent();
            input = new KeyboardHelper(this);

            display.Start(new GLWpfControlSettings
            {
                MajorVersion = 4,
                MinorVersion = 5,
                // This is needed to run on macos
                GraphicsContextFlags = ContextFlags.ForwardCompatible,
                RenderContinuously = false,//avoid relying on WPF's CompositionTarget.Rendering
            });

            renderTimer.Elapsed += (s, e) => { 
                if (!isRendering) 
                { 
                    isRendering = true;
                    Dispatcher.Invoke(display.InvalidateVisual, 
                        System.Windows.Threading.DispatcherPriority.Render);
                } 
            };

        }

        //setup renderer resources after display is ready to take GL calls
        private void display_Ready()
        {
            renderer = new Renderer();
            renderer.Load(new Vector2i(display.FrameBufferWidth, display.FrameBufferHeight));
            isRendererInitialized = true;
            renderTimer.Start();
        }

        private void display_OnRender(TimeSpan delta)
        {
            if(isRendererInitialized)
            {
                renderer.TargetFramebuffer = display.Framebuffer;

                if (resolutionChanged)
                {
                    renderer.UpdateResolution(new Vector2i(display.FrameBufferWidth, display.FrameBufferHeight));
                    resolutionChanged = false;
                }

                UpdateFrame((float)delta.TotalSeconds);
                keepDelta = (float)delta.TotalSeconds;
                renderer.RenderFrame();
                isRendering = false;
            }
        }

        private void display_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            resolutionChanged = true;
        }

        private void display_MouseWheel(object sender, System.Windows.Input.MouseWheelEventArgs e)
        {
            renderer.Camera.Fov -= e.Delta / 10.0f;
        }

        private void UpdateFrame(float delta)
        {
            if (input.IsKeyDown(Key.Escape))
            {
                renderTimer.Stop();
                Application.Current.Shutdown();
            }

            var cam = renderer.Camera;
            if (input.IsKeyDown(Key.W))
            {
                cam.Position += cam.Front * cameraSpeed * delta; // Forward
            }
            if (input.IsKeyDown(Key.S))
            {
                cam.Position -= cam.Front * cameraSpeed * delta; // Backwards
            }
            if (input.IsKeyDown(Key.A))
            {
                cam.Position -= cam.Right * cameraSpeed * delta; // Left
            }
            if (input.IsKeyDown(Key.D))
            {
                cam.Position += cam.Right * cameraSpeed * delta; // Right
            }
            if (input.IsKeyDown(Key.Space))
            {
                cam.Position += cam.Up * cameraSpeed * delta; // Up
            }
            if (input.IsKeyDown(Key.LeftShift))
            {
                cam.Position -= cam.Up * cameraSpeed * delta; // Down
            }

            //update uniforms
            renderer.SetBrightness(Brightness);
           // renderer.SetContrast(Contrast);
            renderer.SetLdir(XL, YL, ZL);
            //TODO: add uniforms to update from gui

        }

        private void display_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                var mousePos = e.GetPosition(display);
                if (_firstMove)
                {
                    _lastPos = mousePos;
                    _firstMove = false;
                }
                else
                {
                    var deltaX = mousePos.X - _lastPos.X;
                    var deltaY = mousePos.Y - _lastPos.Y;
                   
                    _lastPos = mousePos;

                    //renderer.Camera.Yaw += (float)deltaX * sensitivity;

                    //smoothed yaw = mix(smoothed yaw, real yaw, deltatime * k);
                    realYaw =  (float)deltaX * sensitivity;
                    //smoothedYaw = smoothedYaw + (realYaw - smoothedYaw) * keepDelta * 1.0f;
                    renderer.Camera.Yaw += realYaw;

                    //smoothedYaw = 0.0f;

                    //renderer.Camera.Pitch -= (float)deltaY * sensitivity;

                    realPitch =  (float)deltaY * sensitivity;
                   // smoothedPitch = smoothedPitch + (realPitch - smoothedPitch) * keepDelta * 1.0f;
                    renderer.Camera.Pitch -= realPitch;



                    //Updates GUI when value has changed not by the gui control
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(CameraYaw)));
                }
            }
        }

        /// <summary>
        /// Takes keyboard focus away from other gui elements.
        /// </summary>
        private void display_MouseDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            Keyboard.ClearFocus();
            Keyboard.Focus(display);
        }
    }
}
