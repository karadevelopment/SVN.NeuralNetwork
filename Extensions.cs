using SVN.Core.Linq;
using SVN.Core.String;
using SVN.NeuralNetwork.Structures;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SVN.NeuralNetwork
{
    internal static class Extensions
    {
        private const string DATA_SEPARATOR = "\r\n";

        public static string FormatValue(this double param, int decimals = 2)
        {
            var value = param;

            value = value * Math.Pow(10, decimals);
            value = Math.Round(value);
            value = value / Math.Pow(10, decimals);

            var result = value.ToString($"N{decimals:D1}");

            if (value < 0)
            {
                result = result.TrimStart(1);
            }
            if (Math.Abs(value) < 10)
            {
                result = $"0{result}";
            }
            if (value < 0)
            {
                result = $"-{result}";
            }
            else
            {
                result = $"+{result}";
            }

            return result;
        }

        public static void ImportFromFile(this Network param, string path)
        {
            if (File.Exists(path))
            {
                var data = File.ReadAllText(path);
                param.Import(data);
            }
        }

        public static void Import(this Network param, string data)
        {
            var items = data.Split(Enumerable.Range(1, 3).Select(x => Extensions.DATA_SEPARATOR).Join(string.Empty)).ToList();

            param.Epoch = items.ElementAt(0).ParseInt();
            param.Error = items.ElementAt(1).ParseDouble();
            param.ErrorApproximation = items.ElementAt(2).ParseDouble();

            foreach (var item in items.Skip(3))
            {
                var index = items.IndexOf(item) - 3;
                var layer = param.Layers.ElementAt(index);
                layer.Import(item);
            }
        }

        private static void Import(this Layer param, string data)
        {
            var items = data.Split(Enumerable.Range(1, 2).Select(x => Extensions.DATA_SEPARATOR).Join(string.Empty)).ToList();

            foreach (var item in items)
            {
                var index = items.IndexOf(item);
                var neuron = param.Neurons.ElementAt(index);
                neuron.Import(item);
            }
        }

        private static void Import(this Neuron param, string data)
        {
            var items = data.Split(Enumerable.Range(1, 1).Select(x => Extensions.DATA_SEPARATOR).Join(string.Empty)).ToList();

            foreach (var item in items)
            {
                var index = items.IndexOf(item);
                var connection = param.Connections2.ElementAt(index);
                connection.Import(item);
            }
        }

        private static void Import(this Connection param, string data)
        {
            var weight = data.ParseDouble();
            param.Weight = weight;
        }

        public static void ExportToFile(this Network param, string path)
        {
            var directory = Path.GetDirectoryName(path);

            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var data = param.Export();
            File.WriteAllText(path, data);
        }

        public static string Export(this Network param)
        {
            var data = new List<string>
            {
                param.Epoch.ToString(),
                Enumerable.Range(1, 3).Select(x => Extensions.DATA_SEPARATOR).Join(string.Empty),
                param.Error.ToString(),
                Enumerable.Range(1, 3).Select(x => Extensions.DATA_SEPARATOR).Join(string.Empty),
                param.ErrorApproximation.ToString(),
                Enumerable.Range(1, 3).Select(x => Extensions.DATA_SEPARATOR).Join(string.Empty),
                param.Layers.Select(x => x.Export()).Join(Enumerable.Range(1, 3).Select(x => Extensions.DATA_SEPARATOR).Join(string.Empty)),
            };
            return data.Join(string.Empty);
        }

        private static string Export(this Layer param)
        {
            return param.Neurons.Select(x => x.Export()).Join(Enumerable.Range(1, 2).Select(x => Extensions.DATA_SEPARATOR).Join(string.Empty));
        }

        private static string Export(this Neuron param)
        {
            return param.Connections2.Select(x => x.Export()).Join(Enumerable.Range(1, 1).Select(x => Extensions.DATA_SEPARATOR).Join(string.Empty));
        }

        private static string Export(this Connection param)
        {
            return param.Weight.ToString();
        }
    }
}