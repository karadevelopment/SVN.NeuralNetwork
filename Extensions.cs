using System;

namespace SVN.NeuralNetwork
{
    internal static class Extensions
    {
        public static double TransferFunction(this double param)
        {
            return Math.Tanh(param);
            var value = 1;
            var valueMax = 1 + Math.Pow(Math.E, -param);
            return value / valueMax;
        }

        public static double TransferFunctionDerivative(this double param)
        {
            return 1 - Math.Pow(Math.Tanh(param), 2);
        }

        public static string FormatValue(this double param)
        {
            var value = param;
            value = value * Math.Pow(10, 1);
            value = Math.Round(value);
            value = value / Math.Pow(10, 1);
            var result = $"{value:N1}";
            result = default(double) <= value ? '+' + result : result;
            return result;
        }
    }
}