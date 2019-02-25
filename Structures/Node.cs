using System;
using System.Collections.Generic;
using System.Linq;

namespace SVN.NeuralNetwork.Structures
{
    internal class Node
    {
        private List<Edge> Edges1 { get; } = new List<Edge>();
        private List<Edge> Edges2 { get; } = new List<Edge>();
        public double InputValue { get; set; }
        public double OutputValue { get; set; }
        public double Gradient { get; private set; }

        public Node()
        {
        }

        public static void Connect(Node node1, Node node2)
        {
            var edge = new Edge(node1, node2);

            node1.Edges2.Add(edge);
            node2.Edges1.Add(edge);
        }

        public void CalculateValue()
        {
            this.InputValue = this.Edges1.Select(x => x.Node1.OutputValue * x.Weight).Sum();
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
            var delta = this.Edges2.Sum(x => x.Node2.Gradient * x.Weight);
            this.Gradient = delta * this.InputValue.TransferFunctionDerivative();
        }

        public void CalculateOutputGradient(double value)
        {
            var delta = value - this.OutputValue;
            this.Gradient = delta * this.InputValue.TransferFunctionDerivative();
        }

        public void UpdateWeight()
        {
            foreach (var edge in this.Edges1)
            {
                edge.UpdateWeight();
            }
        }

        public override string ToString()
        {
            return $"{this.OutputValue.FormatValue()} / {this.Gradient.FormatValue()}";
        }
    }
}