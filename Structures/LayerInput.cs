using System.Linq;

namespace SVN.NeuralNetwork.Structures
{
    internal class LayerInput : Layer
    {
        public LayerInput(int neurons) : base(neurons)
        {
        }

        protected override void CreateNeurons(int neurons)
        {
            for (var i = 1; i <= neurons; i++)
            {
                var neuron = new NeuronInput();
                base.Neurons.Add(neuron);
            }
            var bias = new NeuronBias();
            base.Neurons.Add(bias);
        }

        public override void SetOutputValues(params double[] values)
        {
            foreach (var neuron in base.Neurons.Where(x => x is NeuronInput))
            {
                var index = base.Neurons.IndexOf(neuron);
                var value = values.ElementAt(index);
                neuron.SetOutputValue(value);
            }
        }
    }
}