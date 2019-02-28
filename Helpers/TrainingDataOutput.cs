using System.Collections.Generic;

namespace SVN.NeuralNetwork.Helpers
{
    internal class TrainingDataOutput
    {
        public List<double> Values { get; } = new List<double>();

        public static TrainingDataOutput Empty
        {
            get => new TrainingDataOutput();
        }

        public TrainingDataOutput(params double[] values)
        {
            this.Values.AddRange(values);
        }

        public double[] ToArray()
        {
            return this.Values.ToArray();
        }
    }
}