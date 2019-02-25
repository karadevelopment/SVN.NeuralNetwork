using SVN.Drawing;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace SVN.NeuralNetwork.Helpers
{
    public class Input
    {
        private List<Color> Pixels { get; } = new List<Color>();

        private Input()
        {
        }

        public static Input FromImageFile(string path)
        {
            var input = new Input();

            using (var bitmap = new Bitmap(path))
            {
                input.SetPixels(bitmap);
            }

            return input;
        }

        private void SetPixels(Bitmap bitmap)
        {
            this.Pixels.Clear();

            for (var y = 1; y <= bitmap.Height; y++)
            {
                for (var x = 1; x <= bitmap.Width; x++)
                {
                    this.Pixels.Add(bitmap.GetPixel(x - 1, y - 1));
                }
            }
        }

        public double[] GetArray()
        {
            return this.Pixels.Select(x => x.Sigmoid()).ToArray();
        }
    }
}