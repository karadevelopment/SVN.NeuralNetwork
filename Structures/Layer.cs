using SVN.Core.Linq;
using System.Collections.Generic;
using System.Linq;

namespace SVN.NeuralNetwork.Structures
{
    internal class Layer
    {
        protected List<Neuron> Neurons { get; } = new List<Neuron>();

        protected Layer(int neurons)
        {
            this.CreateNeurons(neurons);
        }

        public void Import(string data, string separator)
        {
            var items = data.Split(Enumerable.Range(1, 2).Select(x => separator).Join(string.Empty)).ToList();

            foreach (var item in items)
            {
                var index = items.IndexOf(item);
                var neuron = this.Neurons.ElementAt(index);
                neuron.Import(item, separator);
            }
        }

        public string Export(string separator)
        {
            return this.Neurons.Select(x => x.Export(separator)).Join(Enumerable.Range(1, 2).Select(x => separator).Join(string.Empty));
        }

        protected virtual void CreateNeurons(int neurons)
        {
        }

        public static void Connect(Layer layer1, Layer layer2, double initialeWeightRange)
        {
            foreach (var neuron1 in layer1.Neurons.Where(x => x is NeuronInput || x is NeuronHidden || x is NeuronBias))
            {
                foreach (var neuron2 in layer2.Neurons.Where(x => x is NeuronHidden || x is NeuronOutput))
                {
                    Neuron.Connect(neuron1, neuron2, initialeWeightRange);
                }
            }
        }

        public virtual double? GetError(params double[] values)
        {
            return null;
        }

        public virtual void SetOutputValues(params double[] values)
        {
        }

        public virtual void CalculateValues()
        {
        }

        public virtual void CalculateGradients(params double[] values)
        {
        }

        public virtual void UpdateWeights(double alpha, double eta)
        {
        }

        public virtual IEnumerable<double> GetOutputValues()
        {
            yield break;
        }

        public override string ToString()
        {
            return this.Neurons.Select(x => $"{x}").Join("\n");
        }
    }
}