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

        protected virtual void CreateNeurons(int neurons)
        {
        }

        public static void Connect(Layer layer1, Layer layer2)
        {
            foreach (var neuron1 in layer1.Neurons.Where(x => x is NeuronInput || x is NeuronHidden || x is NeuronBias))
            {
                foreach (var neuron2 in layer2.Neurons.Where(x => x is NeuronHidden || x is NeuronOutput))
                {
                    Neuron.Connect(neuron1, neuron2);
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

        public override string ToString()
        {
            return this.Neurons.Select(x => x.ToString()).Join(Enumerable.Range(1, 10).Select(x => " ").Join(string.Empty));
        }
    }
}