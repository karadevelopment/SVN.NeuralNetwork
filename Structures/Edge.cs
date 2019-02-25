namespace SVN.NeuralNetwork.Structures
{
    internal class Edge
    {
        public Node Node1 { get; }
        public Node Node2 { get; }
        public double Weight { get; private set; }
        private double WeightDelta { get; set; } = 1;

        public Edge(Node node1, Node node2)
        {
            this.Node1 = node1;
            this.Node2 = node2;
            this.Weight = Network.GetRandomNumber(0, 1);
        }

        public void UpdateWeight(double eta, double alpha)
        {
            this.WeightDelta = eta * this.Node1.OutputValue * this.Node2.Gradient + alpha * this.WeightDelta;
            this.Weight += this.WeightDelta;
        }

        public override string ToString()
        {
            return $"{this.Node1.OutputValue}<>{this.Node2.OutputValue}";
        }
    }
}