namespace SVN.NeuralNetwork.Structures
{
    internal class Connection
    {
        public Neuron Neuron1 { get; }
        public Neuron Neuron2 { get; }
        public double Weight { get; private set; }
        private double WeightDelta { get; set; } = 1;

        public Connection(Neuron neuron1, Neuron neuron2)
        {
            this.Neuron1 = neuron1;
            this.Neuron2 = neuron2;
            this.Weight = Network.GetRandomNumber(0, 1);
        }

        public void UpdateWeight(double eta, double alpha)
        {
            this.WeightDelta = eta * this.Neuron1.OutputValue * this.Neuron2.Gradient + alpha * this.WeightDelta;
            this.Weight += this.WeightDelta;
        }

        public override string ToString()
        {
            return $"{this.Neuron1.OutputValue}<>{this.Neuron2.OutputValue}";
        }
    }
}