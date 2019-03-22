using SVN.Math2;

namespace SVN.NeuralNetwork.Structures
{
    internal class Connection
    {
        public Neuron Neuron1 { get; }
        public Neuron Neuron2 { get; }
        public double Weight { get; set; }
        public double WeightDelta { get; private set; }

        public Connection(Neuron neuron1, Neuron neuron2)
        {
            this.Neuron1 = neuron1;
            this.Neuron2 = neuron2;
            this.Weight = Random.Range(-1d, +1d);
        }

        public static void Create(Neuron neuron1, Neuron neuron2)
        {
            var connection = new Connection(neuron1, neuron2);
            connection.Neuron1.Connections2.Add(connection);
            connection.Neuron2.Connections1.Add(connection);
        }

        public void UpdateWeight(double alpha, double eta)
        {
            this.WeightDelta = this.WeightDelta * alpha + this.Neuron1.OutputValue * this.Neuron2.Gradient * eta;
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