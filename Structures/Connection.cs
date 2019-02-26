﻿using SVN.Core.String;

namespace SVN.NeuralNetwork.Structures
{
    internal class Connection
    {
        public Neuron Neuron1 { get; }
        public Neuron Neuron2 { get; }
        public double Weight { get; private set; }
        private double WeightDelta { get; set; }

        public Connection(Neuron neuron1, Neuron neuron2)
        {
            this.Neuron1 = neuron1;
            this.Neuron2 = neuron2;
            this.Weight = Network.GetRandomNumber(-3, +3);
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
            this.WeightDelta = this.WeightDelta * alpha + this.Neuron1.OutputValue * this.Neuron2.Gradient * eta;
            this.Weight += this.WeightDelta;
        }

        public string ToStringLevel1()
        {
            return $"{this.Weight.FormatValue()}";
        }

        public string ToStringLevel2()
        {
            return $"DEL {this.WeightDelta.FormatValue()} / WEI {this.Weight.FormatValue()} / OUT {(this.Neuron1.OutputValue * this.Weight).FormatValue()}";
        }

        public override string ToString()
        {
            return this.ToStringLevel1();
        }
    }
}