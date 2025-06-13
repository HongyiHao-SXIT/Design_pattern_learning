using System;
using System.Linq.Expressions;

namespace Basics
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.Write("Please enter the first number: ");
                string strNumberA = Console.ReadLine();
                double numberA = double.Parse(strNumberA);

                Console.Write("Please enter the second number: ");
                string strNumberB = Console.ReadLine();
                double numberB = double.Parse(strNumberB);

                Console.Write("Please enter the operator (+, -, *, /): ");
                string op = Console.ReadLine();

                double result = 0;
                switch (op)
                {
                    case "+":
                        result = numberA + numberB;
                        break;
                    case "-":
                        result = numberA - numberB;
                        break;
                    case "*":
                        result = numberA * numberB;
                        break;
                    case "/":
                        if (numberB == 0)
                        {
                            Console.WriteLine("Error: Cannot divide by zero.");
                            return;
                        }
                        result = numberA / numberB;
                        break;
                    default:
                        Console.WriteLine("Error: Invalid operator. Please use +, -, *, or /.");
                        return;
                }

                Console.WriteLine($"The result is: {result}");
            }
            catch (FormatException)
            {
                Console.WriteLine("Error: Please enter valid numbers.");
            }
            catch (Exception ex)
            {
                Console.WriteLine("An error occurred: " + ex.Message);
            }
        }
    }
}