# Documentation
comming soon

# Dependencies
- .NETFramework 4.6
- SVN.Core (>= 4.3.4)
- SVN.Debug (>= 1.0.5)
- SVN.Math (>= 1.0.8)
- SVN.Tasks (>= 1.0.3)
- System.ValueTuple (>= 4.5.0)

# Example Usage (XOR)
```
using SVN.NeuralNetwork.Helpers;
using SVN.NeuralNetwork.Structures;
using System;
using System.Threading;

public static class Program
{
    public static void Main(string[] args)
    {
        var network = new Network();
        var data = new TrainingData();

        data.AddInput(0, 0);
        data.AddOutput(0);

        data.AddInput(0, 1);
        data.AddOutput(1);

        data.AddInput(1, 0);
        data.AddOutput(1);

        data.AddInput(1, 1);
        data.AddOutput(0);

        network.Initialize(data);
        network.TrainFull(data);

        while (!network.HasLearnedEnough)
        {
            Console.Clear();
            Console.WriteLine(network);
            Thread.Sleep(TimeSpan.FromSeconds(1));
        }
        while (true)
        {
            Console.Clear();
            Console.WriteLine(network);
            Console.ReadLine();
            network.TrainOnce(data);
        }
    }
}
```

# NuGet
https://www.nuget.org/packages/SVN.NeuralNetwork/
