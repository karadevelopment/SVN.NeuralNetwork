# Documentation
comming soon

# Dependencies
- .NETFramework 4.6
- SVN.Core (>= 4.3.2)
- SVN.Debug (>= 1.0.5)
- SVN.Drawing (>= 4.1.4)
- System.Drawing.Primitives (>= 4.3.0)

# Example Usage (XOR)
```using SVN.NeuralNetwork.Enums;
using SVN.NeuralNetwork.Structures;
using System;

public static class Program
{
    public static void Main(string[] args)
    {
        var random = new Random(DateTime.Now.Millisecond);

        var network = new Network
        {
            InputLayerLength = 2,
            HiddenLayerLength = 2,
            OutputLayerLength = 1,
            HiddenLayerAmount = 2,
            Type = GuiType.Level3,
        };
        network.Initialize();

        while (true)
        {
            Console.Clear();
            Console.WriteLine(network);
            Console.ReadLine();

            var input1 = (int)Math.Round(random.NextDouble());
            var input2 = (int)Math.Round(random.NextDouble());
            var output = (input1 + input2) % 2;

            network.FeedForward(input1, input2);
            network.BackPropagation(output);
        }
    }
}
```

# NuGet
https://www.nuget.org/packages/SVN.NeuralNetwork/
