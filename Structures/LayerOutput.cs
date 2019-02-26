using System;
using System.Linq;

namespace SVN.NeuralNetwork.Structures
{
    internal class LayerOutput : Layer
    {
        public LayerOutput(int neurons) : base(neurons)
        {
        }

        protected override void CreateNeurons(int neurons)
        {
            for (var i = 1; i <= neurons; i++)
            {
                var neuron = new NeuronOutput();
                base.Neurons.Add(neuron);
            }
        }

        public override double? GetError(params double[] values)
        {
            var result = default(double);

            foreach (var neuron in base.Neurons.Where(x => x is NeuronOutput))
            {
                var index = base.Neurons.IndexOf(neuron);
                var value = values.ElementAt(index);
                result += neuron.GetError(value).Value;
            }

            result /= values.Length;
            result = Math.Sqrt(result);

            return result;
        }

        public override void CalculateValues()
        {
            foreach (var neuron in base.Neurons.Where(x => x is NeuronHidden || x is NeuronOutput))
            {
                neuron.CalculateValues();
            }
        }

        public override void CalculateGradients(params double[] values)
        {
            foreach (var neuron in base.Neurons.Where(x => x is NeuronHidden || x is NeuronOutput))
            {
                var index = base.Neurons.IndexOf(neuron);
                var value = values.ElementAt(index);
                neuron.CalculateGradient(value);
            }
        }

        public override void UpdateWeights(double alpha, double eta)
        {
            foreach (var neuron in this.Neurons)
            {
                neuron.UpdateWeight(alpha, eta);
            }
        }
    }
}