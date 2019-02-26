using System.Linq;

namespace SVN.NeuralNetwork.Structures
{
    internal class LayerHidden : Layer
    {
        public LayerHidden(int neurons) : base(neurons)
        {
        }

        protected override void CreateNeurons(int neurons)
        {
            for (var i = 1; i <= neurons; i++)
            {
                var neuron = new NeuronHidden();
                base.Neurons.Add(neuron);
            }
            var bias = new NeuronBias();
            base.Neurons.Add(bias);
        }

        public override void CalculateValues()
        {
            foreach (var neuron in base.Neurons.Where(x => x is NeuronHidden))
            {
                neuron.CalculateValues();
            }
        }

        public override void CalculateGradients(params double[] values)
        {
            foreach (var neuron in base.Neurons.Where(x => x is NeuronHidden))
            {
                neuron.CalculateGradient();
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