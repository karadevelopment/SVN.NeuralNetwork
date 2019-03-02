using SVN.Math2;
using System.Linq;

namespace SVN.NeuralNetwork.Structures
{
    internal class NeuronHidden : Neuron
    {
        public NeuronHidden()
        {
        }

        public override void CalculateValues()
        {
            base.InputValue = base.Connections1.Sum(x => x.Neuron1.OutputValue * x.Weight);
            base.OutputValue = base.InputValue.Tanh();
        }

        public override void CalculateGradient(double value)
        {
            var delta = base.Connections2.Sum(x => x.Weight * x.Neuron2.Gradient);
            base.Gradient = delta * base.InputValue.TanhDerivative();
        }

        public override void UpdateWeight(double alpha, double eta)
        {
            foreach (var connection in base.Connections1)
            {
                connection.UpdateWeight(alpha, eta);
            }
        }
    }
}