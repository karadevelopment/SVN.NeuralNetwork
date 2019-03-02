using SVN.Core.String;
using SVN.Math2;

namespace SVN.NeuralNetwork.Structures
{
    internal class Connection
    {
        public Neuron Neuron1 { get; }
        public Neuron Neuron2 { get; }
        public double Weight { get; private set; }
        public double WeightDelta { get; private set; }

        public Connection(Neuron neuron1, Neuron neuron2)
        {
            this.Neuron1 = neuron1;
            this.Neuron2 = neuron2;
            this.Weight = Network.GetRandomNumber(-5, +5);
        }

        public void Import(string data)
        {
            var weight = data.ParseDouble();
            this.Weight = weight;
        }

        public string Export()
        {
            return this.Weight.ToString();
        }

        public void UpdateWeight(double alpha, double eta)
        {
            var target = this.Neuron1.OutputValue * this.Neuron2.Gradient * eta;
            this.WeightDelta = this.WeightDelta.Approach(target, alpha);
            this.Weight += this.WeightDelta;
        }

        public override string ToString()
        {
            var weight = $"WEI {this.Weight.FormatValue()}";
            var value = $"OUT {(this.Neuron1.OutputValue * this.Weight).FormatValue()}";

            return $"{weight} / {value}";
        }
    }
}