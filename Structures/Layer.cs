using SVN.Core.Linq;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SVN.NeuralNetwork.Structures
{
    internal class Layer
    {
        private List<Neuron> Neurons { get; } = new List<Neuron>();

        public Layer(int neuron)
        {
            this.CreateNeurons(neuron);
        }

        private void CreateNeurons(int amount)
        {
            for (var i = 1; i <= amount; i++)
            {
                var neuron = new Neuron();
                this.Neurons.Add(neuron);
            }
            var bias = new Bias();
            this.Neurons.Add(bias);
        }

        public static void Connect(Layer layer1, Layer layer2)
        {
            foreach (var neuron1 in layer1.Neurons)
            {
                foreach (var neuron2 in layer2.Neurons.Where(x => !(x is Bias)))
                {
                    Neuron.Connect(neuron1, neuron2);
                }
            }
        }

        public void SetValues(params double[] values)
        {
            for (var i = 1; i <= values.Length; i++)
            {
                var neuron = this.Neurons.ElementAt(i - 1);
                var value = values.ElementAt(i - 1);
                neuron.OutputValue = value;
            }
        }

        public void CalculateValues()
        {
            foreach (var neuron in this.Neurons.Where(x => !(x is Bias)))
            {
                neuron.CalculateValue();
            }
        }

        public double GetError(params double[] values)
        {
            var result = default(double);

            for (var i = 1; i <= values.Length; i++)
            {
                var neuron = this.Neurons.ElementAt(i - 1);
                var value = values.ElementAt(i - 1);
                result += neuron.GetError(value);
            }

            result /= values.Length;
            result = Math.Sqrt(result);

            return result;
        }

        public void CalculateHiddenGradients()
        {
            foreach (var neuron in this.Neurons)
            {
                neuron.CalculateHiddenGradient();
            }
        }

        public void CalculateOutputGradients(params double[] values)
        {
            for (var i = 1; i <= values.Length; i++)
            {
                var neuron = this.Neurons.ElementAt(i - 1);
                var value = values.ElementAt(i - 1);
                neuron.CalculateOutputGradient(value);
            }
        }

        public void UpdateWeights(double eta, double alpha)
        {
            foreach (var neuron in this.Neurons)
            {
                neuron.UpdateWeight(eta, alpha);
            }
        }

        public override string ToString()
        {
            return this.Neurons.Select(x => x.ToString()).Join(Enumerable.Range(1, 10).Select(x => " ").Join(string.Empty));
        }
    }
}