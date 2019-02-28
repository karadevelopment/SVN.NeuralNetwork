using System.Collections.Generic;

namespace SVN.NeuralNetwork.Helpers
{
    internal class TrainingDataInput
    {
        public List<double> Values { get; } = new List<double>();

        public static TrainingDataInput Empty
        {
            get => new TrainingDataInput();
        }

        public TrainingDataInput(params double[] values)
        {
            this.Values.AddRange(values);
        }

        public double[] ToArray()
        {
            return this.Values.ToArray();
        }
    }
}