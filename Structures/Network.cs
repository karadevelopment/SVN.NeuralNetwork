using SVN.Core.Collections;
using SVN.Core.Linq;
using SVN.Core.Number;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SVN.NeuralNetwork.Structures
{
    public class Network
    {
        private static Random Random { get; } = new Random(DateTime.Now.Millisecond);
        private List<Layer> Layers { get; } = new List<Layer>();
        private int Steps { get; set; }
        private double Error { get; set; } = 1;
        private double ErrorApproximation { get; set; } = 1;

        private double Alpha
        {
            get => 1 - this.ErrorApproximation;
        }

        private double Eta
        {
            get => Math.Pow(this.ErrorApproximation, 2);
        }

        public bool HasLearnedEnough
        {
            get => this.ErrorApproximation < .1;
        }

        public Network(int firstLayerLength, int hiddenLayerLength, int lastLayerLength, int hiddenLayerAmount)
        {
            this.AddInputLayer(firstLayerLength);
            for (var i = 1; i <= hiddenLayerAmount; i++)
            {
                //this.AddHiddenLayer(firstLayerLength * 2 / 3 + lastLayerLength);
                this.AddHiddenLayer(hiddenLayerLength);
            }
            this.AddOutputLayer(lastLayerLength);
            this.Connect();
        }

        internal static double GetRandomNumber(double min, double max)
        {
            var diff = Math.Abs(max - min);
            var rnd = Network.Random.NextDouble();
            return min + diff * rnd;
        }

        private void AddInputLayer(int neurons)
        {
            var layer = new LayerInput(neurons);
            this.Layers.Add(layer);
        }

        private void AddHiddenLayer(int neurons)
        {
            var layer = new LayerHidden(neurons);
            this.Layers.Add(layer);
        }

        private void AddOutputLayer(int neurons)
        {
            var layer = new LayerOutput(neurons);
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

        private void SetOutputValues(params double[] values)
        {
            foreach (var layer in this.Layers.Where(x => x is LayerInput))
            {
                layer.SetOutputValues(values);
            }
        }

        private void CalculateValues(params double[] values)
        {
            foreach (var layer in this.Layers.Where(x => x is LayerHidden || x is LayerOutput))
            {
                layer.CalculateValues();
            }
        }

        private void CalculateError(params double[] values)
        {
            foreach (var layer in this.Layers.Where(x => x is LayerOutput))
            {
                this.Error = layer.GetError(values).Value;
                this.ErrorApproximation = this.ErrorApproximation.Approach(this.Error);
            }
        }

        private void CalculateGradients(params double[] values)
        {
            foreach (var layer in this.Layers.Invert())
            {
                layer.CalculateGradients(values);
            }
        }

        private void UpdateWeights()
        {
            foreach (var layer in this.Layers.Invert())
            {
                layer.UpdateWeights(this.Alpha, this.Eta);
            }
        }

        public void FeedForward(params double[] values)
        {
            this.SetOutputValues(values);
            this.CalculateValues(values);
        }

        public void BackPropagation(params double[] values)
        {
            this.Steps++;
            this.CalculateError(values);
            this.CalculateGradients(values);
            this.UpdateWeights();
        }

        public override string ToString()
        {
            return $"Steps: {this.Steps}\n\n{this.Layers.Select(x => x.ToString()).Join("\n")}\n\nAlpha: {this.Alpha:N5}\nEta: {this.Eta:N5}\nError: {this.Error:N5}\nErrorApproximation: {this.ErrorApproximation:N5}";
            return $"Steps: {this.Steps}\nAlpha: {this.Alpha:N5}\nEta: {this.Eta:N5}\nError: {this.Error:N5}\nErrorApproximation: {this.ErrorApproximation:N5}";
        }
    }
}