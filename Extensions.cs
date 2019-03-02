using SVN.Core.String;
using System;

namespace SVN.NeuralNetwork
{
    internal static class Extensions
    {
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
    }
}