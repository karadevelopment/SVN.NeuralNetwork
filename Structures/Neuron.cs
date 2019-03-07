using System.Collections.Generic;
using System.Linq;

namespace SVN.NeuralNetwork.Structures
{
    internal class Neuron
    {
        public List<Connection> Connections1 { get; } = new List<Connection>();
        public List<Connection> Connections2 { get; } = new List<Connection>();
        public double InputValue { get; protected set; }
        public double OutputValue { get; protected set; } = 1;
        public double Gradient { get; protected set; }

        protected Neuron()
        {
        }

        public virtual double? GetError(double value)
        {
            return null;
        }

        public virtual void SetInputValue(double value)
        {
        }

        public virtual void SetOutputValue(double value)
        {
        }

        public virtual void CalculateValues()
        {
        }

        public virtual void CalculateGradient(double value = default(double))
        {
        }

        public virtual void UpdateWeight(double alpha, double eta)
        {
        }

        public override string ToString()
        {
            var inputValue = this.InputValue.FormatValue();
            var outputValue = this.OutputValue.FormatValue();
            var gradient = this.Gradient.FormatValue();

            var weights1 = this.Connections1.Select(x => x.Weight).DefaultIfEmpty(0).Average().FormatValue();
            var weights2 = this.Connections2.Select(x => x.Weight).DefaultIfEmpty(0).Average().FormatValue();

            var neuron = $"IN {inputValue} / OUT {outputValue} / GRD {gradient}";
            var connections = $"W_IN {weights1} / W_OUT {weights2}";
            var type = this.GetType().Name;

            return $"{neuron} / {connections} | {type}";
        }
    }
}