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

        public void Import(string data)
        {
            var items = data.Split(Enumerable.Range(1, 2).Select(x => Network.DATA_SEPARATOR).Join(string.Empty)).ToList();

            foreach (var item in items)
            {
                var index = items.IndexOf(item);
                var neuron = this.Neurons.ElementAt(index);
                neuron.Import(item);
            }
        }

        public string Export()
        {
            return this.Neurons.Select(x => x.Export()).Join(Enumerable.Range(1, 2).Select(x => Network.DATA_SEPARATOR).Join(string.Empty));
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

        public virtual IEnumerable<int> GetOutputValues()
        {
            yield break;
        }

        public string ToStringLevel1()
        {
            return this.Neurons.Select(x => x.ToStringLevel1()).Join(Enumerable.Range(1, 1).Select(x => " ").Join(string.Empty));
        }

        public string ToStringLevel2()
        {
            return this.Neurons.Select(x => x.ToStringLevel2()).Join(Enumerable.Range(1, 5).Select(x => " ").Join(string.Empty));
        }

        public string ToStringLevel3()
        {
            return this.Neurons.Select(x => x.ToStringLevel3()).Join("\n");
        }

        public override string ToString()
        {
            return this.ToStringLevel1();
        }
    }
}