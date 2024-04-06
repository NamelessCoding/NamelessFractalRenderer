using OpenTK.Graphics.OpenGL4;
using OpenTK.Mathematics;
using System;
using System.Collections.Generic;
using System.IO;

namespace Common
{
    public class Shader : IDisposable
    {
        public readonly int Handle;
        public string Name;
        private readonly Dictionary<string, int> _uniformLocations = [];

        public Shader(string vertPath, string fragPath, bool incl = false)
        {
          
            //includeFile.include
            Name = fragPath;
            var include = File.ReadAllText("Shaders/includeFile.frag");
            include += File.ReadAllText("Shaders/includeGlobals.frag");

            var shaderSource = File.ReadAllText(vertPath);

            var vertexShader = GL.CreateShader(ShaderType.VertexShader);

            GL.ShaderSource(vertexShader, shaderSource);

            CompileShader(vertexShader, Name);

            shaderSource = File.ReadAllText(fragPath);
            if (incl)
            {
                shaderSource = include + shaderSource;
            }
            var fragmentShader = GL.CreateShader(ShaderType.FragmentShader);
            GL.ShaderSource(fragmentShader, shaderSource);
            CompileShader(fragmentShader, Name);

        
            Handle = GL.CreateProgram();

            GL.AttachShader(Handle, vertexShader);
            GL.AttachShader(Handle, fragmentShader);

            LinkProgram(Handle, Name);

            GL.DetachShader(Handle, vertexShader);
            GL.DetachShader(Handle, fragmentShader);
            GL.DeleteShader(fragmentShader);
            GL.DeleteShader(vertexShader);

            
            GL.GetProgram(Handle, GetProgramParameterName.ActiveUniforms, out var numberOfUniforms);

            for (var i = 0; i < numberOfUniforms; i++)
            {
                var key = GL.GetActiveUniform(Handle, i, out _, out _);

                var location = GL.GetUniformLocation(Handle, key);

                _uniformLocations.Add(key, location);
            }
        }


        public Shader(string compPath, bool incl = false)
        {
          
            Name = compPath;

            var include = File.ReadAllText("Shaders/includeFile.frag");
            include += File.ReadAllText("Shaders/includeGlobals.frag");
            var shaderSource = File.ReadAllText(compPath);
            if (incl)
            {
                shaderSource = include + shaderSource;
            }
            var compShader = GL.CreateShader(ShaderType.ComputeShader);

            GL.ShaderSource(compShader, shaderSource);

            CompileShader(compShader, Name);


    
            Handle = GL.CreateProgram();

            GL.AttachShader(Handle, compShader);

            LinkProgram(Handle, Name);

         
            GL.DetachShader(Handle, compShader);
            GL.DeleteShader(compShader);

            GL.GetProgram(Handle, GetProgramParameterName.ActiveUniforms, out var numberOfUniforms);

            for (var i = 0; i < numberOfUniforms; i++)
            {
                var key = GL.GetActiveUniform(Handle, i, out _, out _);

                var location = GL.GetUniformLocation(Handle, key);

                _uniformLocations.Add(key, location);
            }
        }

        private static void CompileShader(int shader, string Name)
        {
            GL.CompileShader(shader);

            GL.GetShader(shader, ShaderParameter.CompileStatus, out var code);
            if (code != (int)All.True)
            {
                var infoLog = GL.GetShaderInfoLog(shader);
                throw new Exception($"Error occurred whilst compiling Shader({shader}).({Name})\n\n{infoLog}");
            }
        }

        private static void LinkProgram(int program, string Name)
        {
            GL.LinkProgram(program);

            GL.GetProgram(program, GetProgramParameterName.LinkStatus, out var code);
            if (code != (int)All.True)
            {
                throw new Exception($"Error occurred whilst linking Program({program}) {Name}");
            }
        }

        public void Use()
        {
            GL.UseProgram(Handle);
        }


        public int GetAttribLocation(string attribName)
        {
            return GL.GetAttribLocation(Handle, attribName);
        }

        public void SetMatrix4(Matrix4 matrix, string uniformName)
        {
            GL.UniformMatrix4(GetUniformLocation(uniformName), false, ref matrix);
        }

        public void SetFloat(float num, string uniformName)
        {
            GL.Uniform1(GetUniformLocation(uniformName), num);
        }
        public void SetVector3(Vector3 vec, string uniformName)
        {
            GL.Uniform3(GetUniformLocation(uniformName), vec.X, vec.Y, vec.Z);
        }

        public void SetVector2(Vector2 vec, string uniformName)
        {
            GL.Uniform2(GetUniformLocation(uniformName), vec.X, vec.Y);
        }

        public void SetInt(string uniformName, int value)
        {
            GL.Uniform1(GetUniformLocation(uniformName), value);
        }

        private int GetUniformLocation(string uniformName)
        {
            if (!_uniformLocations.TryGetValue(uniformName, out var location))
                location = GL.GetUniformLocation(Handle, uniformName);

       
            return location;
        }

        private bool disposedValue = false;

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                GL.DeleteProgram(Handle);

                disposedValue = true;
            }
        }

        ~Shader()
        {
            GL.DeleteProgram(Handle);
        }


        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }
}
