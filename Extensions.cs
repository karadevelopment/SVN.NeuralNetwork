using System;

namespace SVN.NeuralNetwork
{
    internal static class Extensions
    {
        public static double TransferFunction(this double param)
        {
            var value = 1;
            var valueMax = 1 + Math.Pow(Math.E, -param);
            return value / valueMax;
            return Math.Tanh(param);
        }

        public static double TransferFunctionDerivative(this double param)
        {
            return param.TransferFunction() * (1 - param.TransferFunction());
            return 1 - Math.Pow(Math.Tanh(param), 2);
        }

        public static string FormatValue(this double param, int decimals = 2)
        {
            var value = param;
            value = value * Math.Pow(10, decimals);
            value = Math.Round(value);
            value = value / Math.Pow(10, decimals);
            var result = value.ToString($"N{decimals:D1}");
            result = default(double) <= value ? '+' + result : result;
            return result;
        }
    }
}