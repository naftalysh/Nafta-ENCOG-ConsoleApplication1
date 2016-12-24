using System;

namespace LotoPrediction
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            //
            // Commandline parameters
            //7 20 0.01 false or
            //Report
            //
            Boolean blnReport = false;
            var programName = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName;
            System.Console.WriteLine("Format: " + programName + " LotoNumber, PastWindowSize, MaxError, blnShowConsole ");
            System.Console.WriteLine("Or, Format: " + programName + " report ");

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
                if (args[0].ToUpper() == "REPORT")
                    blnReport = true;
                else
                    lotoPrediction.blnShowConsole = Convert.ToBoolean(args[0]);
            }


            if (blnReport)
                lotoPrediction.Report();
            else
            lotoPrediction.Predict();

            DateTime Execution_End = System.DateTime.Now;

            String line = String.Format("H:M:S.MS = {0:D1}:{1:D2}:{2:D2}.{3:D3}",
                            (Execution_End - Execution_Start).Hours,
                            (Execution_End - Execution_Start).Minutes,
                            (Execution_End - Execution_Start).Seconds,
                            (Execution_End - Execution_Start).Milliseconds);

            Console.WriteLine("-- End Of execution -- \n" + line);
            //Console.WriteLine("AppPath = " + Config.AppPath.ToString());

            Console.ReadLine();

        }
    }
}