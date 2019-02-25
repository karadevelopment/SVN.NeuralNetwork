using System;
using System.Collections.Generic;
using System.Linq;

namespace SVN.NeuralNetwork.Structures
{
    internal class Neuron
    {
        private List<Connection> Connections1 { get; } = new List<Connection>();
        private List<Connection> Connections2 { get; } = new List<Connection>();
        public double InputValue { get; set; }
        public double OutputValue { get; set; }
        public double Gradient { get; private set; }

        public Neuron()
        {
        }

        public static void Connect(Neuron neuron1, Neuron neuron2)
        {
            var connection = new Connection(neuron1, neuron2);

            neuron1.Connections2.Add(connection);
            neuron2.Connections1.Add(connection);
        }

        public void CalculateValue()
        {
            this.InputValue = this.Connections1.Select(x => x.Neuron1.OutputValue * x.Weight).Sum();
            this.OutputValue = this.InputValue.TransferFunction();
        }

        public double GetError(double value)
        {
            var delta = value - this.OutputValue;
            var result = Math.Pow(delta, 2);
            return result;
        }

        public void CalculateHiddenGradient()
        {
            var delta = this.Connections2.Sum(x => x.Neuron2.Gradient * x.Weight);
            this.Gradient = delta * this.InputValue.TransferFunctionDerivative();
        }

        public void CalculateOutputGradient(double value)
        {
            var delta = value - this.OutputValue;
            this.Gradient = delta * this.InputValue.TransferFunctionDerivative();
        }

        public void UpdateWeight(double eta, double alpha)
        {
            foreach (var connection in this.Connections1)
            {
                connection.UpdateWeight(eta, alpha);
            }
        }

        public override string ToString()
        {
            return $"{this.OutputValue.FormatValue()} / {this.Gradient.FormatValue()}";
        }
    }
}