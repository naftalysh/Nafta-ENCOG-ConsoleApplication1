using System;

namespace LotoPrediction
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var programName = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName;
            System.Console.WriteLine("Format: " + programName + " LotoNumber, PastWindowSize, MaxError, blnShowConsole ");

            DateTime Execution_Start = System.DateTime.Now;

            var lotoPrediction = new LotoPrediction();
            if (args.Length == 4)
            {
                lotoPrediction.LotoNumber = Convert.ToInt32(args[0]);
                lotoPrediction.PastWindowSize = Convert.ToInt32(args[1]);
                lotoPrediction.MaxError = Convert.ToDouble(args[2]);
                lotoPrediction.blnShowConsole = Convert.ToBoolean(args[3]);
            }
            else if (args.Length == 3)  
            {
                lotoPrediction.LotoNumber = Convert.ToInt32(args[0]);
                lotoPrediction.PastWindowSize = Convert.ToInt32(args[1]);
                lotoPrediction.blnShowConsole = Convert.ToBoolean(args[2]);
            }
            else if (args.Length == 2)
            {
                lotoPrediction.LotoNumber = Convert.ToInt32(args[0]);
                lotoPrediction.blnShowConsole = Convert.ToBoolean(args[1]);
            }
            else if (args.Length == 1)
            {
                lotoPrediction.blnShowConsole = Convert.ToBoolean(args[0]);
            }

            lotoPrediction.Predict();

            DateTime Execution_End = System.DateTime.Now;

            String line = String.Format("H:M:S.MS = {0:D1}:{1:D2}:{2:D2}.{3:D3}",
                            (Execution_End - Execution_Start).Hours,
                            (Execution_End - Execution_Start).Minutes,
                            (Execution_End - Execution_Start).Seconds,
                            (Execution_End - Execution_Start).Milliseconds);

            Console.WriteLine("-- End Of execution -- \n" + line);
            //Console.ReadLine();
        }
    }
}