using Encog.Engine.Network.Activation;
using Encog.ML.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.App.Analyst;
using Encog.App.Analyst.CSV.Normalize;
using Encog.App.Analyst.Wizard;
using Encog.ML.Data;
using Encog.Util.CSV;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;



namespace LotoPrediction
{
    class Program
    {
        static void Main(string[] args)
        {
            var programName = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName;

            if (args.Length == 0)
            {
                System.Console.WriteLine("Format: " + programName + " LotoNumber, PastWindowSize, MaxError ");
                return;
            }

           
            DateTime Execution_Start = System.DateTime.Now;

            var lotoPrediction = new LotoPrediction();
            if (args.Length == 3)
            {
                lotoPrediction.LotoNumber = Convert.ToInt32(args[0]);
                lotoPrediction.PastWindowSize = Convert.ToInt32(args[1]);
                lotoPrediction.MaxError = Convert.ToDouble(args[2]);
            }
            else if (args.Length == 2)
            {
                lotoPrediction.LotoNumber = Convert.ToInt32(args[0]);
                lotoPrediction.PastWindowSize = Convert.ToInt32(args[1]);
            }
            else if (args.Length == 1)
            {
                lotoPrediction.LotoNumber = Convert.ToInt32(args[0]);
            }

            lotoPrediction.Predict();

            DateTime Execution_End = System.DateTime.Now;

            String line = String.Format("H:M:S.MS = {0:D1}:{1:D2}:{2:D2}.{3:D3}",
                            (Execution_End - Execution_Start).Hours,
                            (Execution_End - Execution_Start).Minutes,
                            (Execution_End - Execution_Start).Seconds,
                            (Execution_End - Execution_Start).Milliseconds);


            Console.WriteLine("-- End Of execution -- \n" + line);
            Console.ReadLine();

        }
                
    }
}
