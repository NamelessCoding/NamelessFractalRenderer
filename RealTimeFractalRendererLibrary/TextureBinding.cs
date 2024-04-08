using OpenTK.Graphics.OpenGL4;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RealTimeFractalRendererLibrary
{
    /// <summary>
    /// Represents an opengl texture that is bound to a texture unit. The units are taken incrementally.
    /// Assumes we'll never run out of available units (MaxCombinedTextureImageUnits).
    /// </summary>
    public class TextureBinding
    {
        //Start counting from 1, because unit 0 is reserved for setup operations.
        private static int unitCounter = 1;

        public int Handle { get; private set; }

        public string Name { get; private set; } = "cica";

        /// <summary>
        /// The TextureUnit that this Texture is bound to.
        /// </summary>
        public int Unit { get; private set; }

        public TextureBinding(TextureTarget target = TextureTarget.Texture2D)
        {
            Handle = GL.GenTexture();
            Unit = unitCounter;
            //remark: not using the glenum here because it only goes up to 32 units, even though a _minimum_ of 80 is supported according to 4.5 spec.
            GL.ActiveTexture(TextureUnit.Texture0 + Unit);
            GL.BindTexture(target, Handle);
            unitCounter++;
        }

        //TODO: Dispose
    }
}
