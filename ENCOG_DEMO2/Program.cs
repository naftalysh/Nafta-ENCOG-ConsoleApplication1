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
            var lotoPrediction = new LotoPrediction();
            lotoPrediction.Predict();

            Console.WriteLine("Press any key to exit..");
            Console.ReadLine();


        }
                
    }
}
