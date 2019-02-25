using SVN.Core.Linq;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SVN.NeuralNetwork.Structures
{
    internal class Layer
    {
        private List<Node> Nodes { get; } = new List<Node>();

        public Layer(int nodes)
        {
            this.CreateNodes(nodes);
        }

        private void CreateNodes(int amount)
        {
            for (var i = 1; i <= amount; i++)
            {
                var node = new Node();
                this.Nodes.Add(node);
            }
            var bias = new Bias();
            this.Nodes.Add(bias);
        }

        public static void Connect(Layer layer1, Layer layer2)
        {
            foreach (var node1 in layer1.Nodes)
            {
                foreach (var node2 in layer2.Nodes.Where(x => !(x is Bias)))
                {
                    Node.Connect(node1, node2);
                }
            }
        }

        public void SetValues(params double[] values)
        {
            for (var i = 1; i <= values.Length; i++)
            {
                var node = this.Nodes.ElementAt(i - 1);
                var value = values.ElementAt(i - 1);
                node.InputValue = value;
                node.OutputValue = value;
            }
        }

        public void CalculateValues()
        {
            foreach (var node in this.Nodes)
            {
                node.CalculateValue();
            }
        }

        public double GetError(params double[] values)
        {
            var result = default(double);

            for (var i = 1; i <= values.Length; i++)
            {
                var node = this.Nodes.ElementAt(i - 1);
                var value = values.ElementAt(i - 1);
                result += node.GetError(value);
            }

            result /= values.Length;
            result = Math.Sqrt(result);

            return result;
        }

        public void CalculateHiddenGradients()
        {
            foreach (var node in this.Nodes)
            {
                node.CalculateHiddenGradient();
            }
        }

        public void CalculateOutputGradients(params double[] values)
        {
            for (var i = 1; i <= values.Length; i++)
            {
                var node = this.Nodes.ElementAt(i - 1);
                var value = values.ElementAt(i - 1);
                node.CalculateOutputGradient(value);
            }
        }

        public void UpdateWeights(double eta, double alpha)
        {
            foreach (var node in this.Nodes)
            {
                node.UpdateWeight(eta, alpha);
            }
        }

        public override string ToString()
        {
            return this.Nodes.Select(x => x.ToString()).Join(Enumerable.Range(1, 10).Select(x => " ").Join(string.Empty));
        }
    }
}