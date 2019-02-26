namespace SVN.NeuralNetwork.Structures
{
    internal class NeuronInput : Neuron
    {
        public NeuronInput()
        {
        }

        public override void SetOutputValue(double value)
        {
            base.OutputValue = value;
        }
    }
}