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
            base.InputValue = base.Connections1.Select(x => x.Neuron1.OutputValue * x.Weight).Sum();
            base.OutputValue = base.InputValue.TransferFunction();
        }

        // TODO gradient
        public override void CalculateGradient(double value)
        {
            var delta = base.Connections2.Sum(x => x.Neuron2.Gradient * x.Weight);
            base.Gradient = delta * base.InputValue.TransferFunctionDerivative();
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