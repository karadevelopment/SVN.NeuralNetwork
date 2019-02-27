using SVN.Math2;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SVN.NeuralNetwork.Helpers
{
    public class TrainingData
    {
        private readonly List<TrainingDataInput> Inputs = new List<TrainingDataInput>();
        private readonly List<TrainingDataOutput> Outputs = new List<TrainingDataOutput>();

        public int InputLength
        {
            get => this.Inputs.Select(x => x.Values.Count).Max();
        }

        public int OutputLength
        {
            get => this.Outputs.Select(x => x.Values.Count).Max();
        }

        internal (double[] input, double[] output) Random
        {
            get
            {
                var length = Math.Min(this.Inputs.Count, this.Outputs.Count);
                var index = RNG.Int(1, length) - 1;
                var input = this.Inputs.ElementAtOrDefault(index) ?? TrainingDataInput.Empty;
                var output = this.Outputs.ElementAtOrDefault(index) ?? TrainingDataOutput.Empty;
                return (input.ToArray(), output.ToArray());
            }
        }

        public TrainingData()
        {
        }

        public void AddInput(params double[] values)
        {
            this.Inputs.Add(new TrainingDataInput(values));
        }

        public void AddOutput(params double[] values)
        {
            this.Outputs.Add(new TrainingDataOutput(values));
        }
    }
}