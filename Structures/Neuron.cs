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

        public void Import(string data)
        {
            var items = data.Split("\r\n").ToList();

            foreach (var item in items)
            {
                var index = items.IndexOf(item);
                var connection = this.Connections2.ElementAt(index);
                connection.Import(item);
            }
        }

        public string Export()
        {
            return this.Connections2.Select(x => x.Export()).Join("\r\n");
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

        public string ToStringLevel1()
        {
            return $"{this.InputValue.FormatValue()}/{this.OutputValue.FormatValue()}";
        }

        public string ToStringLevel2()
        {
            return $"{this.InputValue.FormatValue()}/{this.OutputValue.FormatValue()}/{this.Gradient.FormatValue()}";
        }

        public string ToStringLevel3()
        {
            return $"IN {this.InputValue.FormatValue()} / OUT {this.OutputValue.FormatValue()} / GRD {this.Gradient.FormatValue()} / W_IN {this.Connections1.Select(x => x.Weight).DefaultIfEmpty(0).Average().FormatValue()} / W_OUT {this.Connections2.Select(x => x.Weight).DefaultIfEmpty(0).Average().FormatValue()}";
        }

        public override string ToString()
        {
            return this.ToStringLevel1();
        }
    }
}