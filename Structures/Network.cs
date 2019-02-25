using SVN.Core.Linq;
using SVN.Core.Number;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SVN.NeuralNetwork.Structures
{
    public class Network
    {
        // eta
        // 0.0 slow learner
        // 0.2 medium learner
        // 1.0 reckless learner

        // alpha
        // 0.0 no momentum
        // 0.5 moderate momentum

        private static Random Random { get; } = new Random(DateTime.Now.Millisecond);
        private List<Layer> Layers { get; } = new List<Layer>();
        private int Steps { get; set; }
        private double Error { get; set; } = 1;
        private double ErrorApproximation { get; set; } = 1;

        private double Eta
        {
            get => Math.Pow(this.ErrorApproximation, 2);
        }

        private double Alpha
        {
            get => 1 - this.ErrorApproximation;
        }

        public bool HasLearnedEnough
        {
            get => this.ErrorApproximation < .1;
        }

        public Network(int firstLayerLength, int lastLayerLength, int hiddenLayers = 2)
        {
            this.AddLayer(firstLayerLength);
            for (var i = 1; i <= hiddenLayers; i++)
            {
                this.AddLayer(firstLayerLength * 2 / 3 + lastLayerLength);
            }
            this.AddLayer(lastLayerLength);
            this.Connect();
        }

        internal static double GetRandomNumber(double min, double max)
        {
            var diff = Math.Abs(max - min);
            var rnd = Network.Random.NextDouble();
            return min + diff * rnd;
        }

        private void AddLayer(int neurons)
        {
            var layer = new Layer(neurons);
            this.Layers.Add(layer);
        }

        private void Connect()
        {
            var layer1 = default(Layer);
            foreach (var layer2 in this.Layers)
            {
                if (layer1 != null)
                {
                    Layer.Connect(layer1, layer2);
                }
                layer1 = layer2;
            }
        }

        private void SetInput(params double[] values)
        {
            foreach (var layer in this.Layers.Take(1))
            {
                layer.SetValues(values);
            }
        }

        private void CalculateValues(params double[] values)
        {
            foreach (var layer in this.Layers.Skip(1))
            {
                layer.CalculateValues();
            }
        }

        private void CalculateError(params double[] values)
        {
            foreach (var layer in this.Layers.AsEnumerable().Reverse().Take(1))
            {
                this.Error = layer.GetError(values);
                this.ErrorApproximation = this.ErrorApproximation.Approach(this.Error);
            }
        }

        private void CalculateHiddenGradients()
        {
            foreach (var layer in this.Layers.Skip(1).AsEnumerable().Reverse().Skip(1))
            {
                layer.CalculateHiddenGradients();
            }
        }

        private void CalculateOutputGradients(params double[] values)
        {
            foreach (var layer in this.Layers.AsEnumerable().Reverse().Take(1))
            {
                layer.CalculateOutputGradients(values);
            }
        }

        private void UpdateWeights()
        {
            foreach (var layer in this.Layers.Skip(1).AsEnumerable().Reverse())
            {
                layer.UpdateWeights(this.Eta, this.Alpha);
            }
        }

        public void FeedForward(params double[] values)
        {
            this.SetInput(values);
            this.CalculateValues(values);
        }

        public void BackPropagation(params double[] values)
        {
            this.Steps++;
            this.CalculateError(values);
            this.CalculateOutputGradients(values);
            this.CalculateHiddenGradients();
            this.UpdateWeights();
        }

        public override string ToString()
        {
            return $"Steps: {this.Steps:D5}\nError: {this.Error:N5}\nErrorApproximation: {this.ErrorApproximation:N5}";
            return $"Steps: {this.Steps:D5}\n\n{this.Layers.Select(x => x.ToString()).Join("\n")}\n\nError: {this.Error:N5}\nErrorApproximation: {this.ErrorApproximation:N5}";
        }
    }
}