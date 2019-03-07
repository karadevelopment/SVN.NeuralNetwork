using SVN.Math2;
using System;
using System.Linq;

namespace SVN.NeuralNetwork.Structures
{
    internal class NeuronOutput : Neuron
    {
        public NeuronOutput()
        {
        }

        public override double? GetError(double value)
        {
            var delta = value - base.OutputValue;
            var result = Math.Pow(delta, 2);
            return result;
        }

        public override void CalculateValues()
        {
            base.InputValue = base.Connections1.Sum(x => x.Neuron1.OutputValue * x.Weight);
            base.OutputValue = base.InputValue.Sigmoid();
        }

        public override void CalculateGradient(double value)
        {
            var delta = value - base.OutputValue;
            base.Gradient = delta * base.InputValue.SigmoidDerivative();
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