using SVN.Core.Linq;
using System.Collections.Generic;
using System.Linq;

namespace SVN.NeuralNetwork.Structures
{
    internal class Neuron
    {
        protected List<Connection> Connections1 { get; } = new List<Connection>();
        protected List<Connection> Connections2 { get; } = new List<Connection>();
        public double InputValue { get; protected set; }
        public double OutputValue { get; protected set; }
        public double Gradient { get; protected set; }

        protected Neuron()
        {
        }

        public void Import(string data, string separator)
        {
            var items = data.Split(Enumerable.Range(1, 1).Select(x => separator).Join(string.Empty)).ToList();

            foreach (var item in items)
            {
                var index = items.IndexOf(item);
                var connection = this.Connections2.ElementAt(index);
                connection.Import(item);
            }
        }

        public string Export(string separator)
        {
            return this.Connections2.Select(x => x.Export()).Join(Enumerable.Range(1, 1).Select(x => separator).Join(string.Empty));
        }

        public static void Connect(Neuron neuron1, Neuron neuron2)
        {
            var connection = new Connection(neuron1, neuron2);

            neuron1.Connections2.Add(connection);
            neuron2.Connections1.Add(connection);
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

        public virtual double? GetOutputValue()
        {
            return null;
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