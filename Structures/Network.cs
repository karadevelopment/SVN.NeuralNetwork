using SVN.Core.Collections;
using SVN.Core.Linq;
using SVN.Math2;
using SVN.NeuralNetwork.Helpers;
using SVN.Tasks;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace SVN.NeuralNetwork.Structures
{
    public class Network
    {
        internal int Epoch { get; set; }
        internal double Error { get; set; } = 1;
        internal double ErrorApproximation { get; set; } = 1;
        internal List<Layer> Layers { get; } = new List<Layer>();

        private double Alpha
        {
            get => 1 - this.ErrorApproximation;
        }

        private double Eta
        {
            get => Math.Pow(this.ErrorApproximation, 2);
        }

        public double ErrorPercentage
        {
            get => this.ErrorApproximation * 100;
        }

        public bool HasLearnedEnough
        {
            get => this.ErrorPercentage < 1;
        }

        public Network()
        {
        }

        public void Initialize(TrainingData data)
        {
            this.ClearLayers();
            this.AddInputLayer(data.InputLength);
            this.AddHiddenLayer(((double)data.InputLength / (data.InputLength + 1) * 4 + data.OutputLength).RoundToInt());
            this.AddOutputLayer(data.OutputLength);
            this.Connect();
        }

        private void ClearLayers()
        {
            this.Layers.Clear();
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

        private void FeedForward(params double[] values)
        {
            this.SetOutputValues(values);
            this.CalculateValues(values);
        }

        private void BackPropagation(params double[] values)
        {
            this.CalculateError(values);
            this.CalculateGradients(values);
            this.UpdateWeights();
            this.Epoch++;
        }

        public void TrainOnce(TrainingData data)
        {
            var (input, output) = data.Random;
            this.FeedForward(input);
            this.BackPropagation(output);
        }

        public void TrainFull(TrainingData data, TimeSpan sleepTimePerEpoch = default(TimeSpan))
        {
            TaskContainer.Run(() =>
            {
                while (!this.HasLearnedEnough)
                {
                    var (input, output) = data.Random;

                    this.FeedForward(input);
                    this.BackPropagation(output);

                    if (sleepTimePerEpoch != default(TimeSpan))
                    {
                        Thread.Sleep(sleepTimePerEpoch);
                    }
                }
            });
        }

        public override string ToString()
        {
            var layers = this.Layers.Select(x => $"{x}").Join("\n\n");
            var epoch = $"Epoch: {this.Epoch}";
            var alpha = $"Alpha: {this.Alpha:N5}";
            var eta = $"Eta: {this.Eta:N5}";
            var error = $"Error: {this.Error:N5}";
            var errorApproximation = $"ErrorApproximation: {this.ErrorApproximation:N5}";

            return $"{layers}\n\n{epoch}\n{alpha}\n{eta}\n{error}\n{errorApproximation}";
        }
    }
}