//
// Encog(tm) Core v3.2 - .Net Version
// http://www.heatonresearch.com/encog/
//
// Copyright 2008-2014 Heaton Research, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// For more information on Heaton Research copyrights, licenses
// and trademarks visit:
// http://www.heatonresearch.com/copyright
//
using ConsoleExamples.Examples;
using System;
using System.IO;

namespace Encog.Examples.Market
{
    public class MarketPredict : IExample
    {
        public static ExampleInfo Info
        {
            get
            {
                var info = new ExampleInfo(
                    typeof(MarketPredict),
                    "market",
                    "Simple Market Prediction",
                    "Use EOD data to predict direction of a stock.");
                return info;
            }
        }

        #region IExample Members

        public void Execute(IExampleInterface app)
        {
            if (app.Args.Length < 2)
            {
                Console.WriteLine(@"MarketPredict [data dir] [generate/train/prune/evaluate]");
            }
            else
            {
                var dataDir = new FileInfo(app.Args[0]);
                if (String.Compare(app.Args[1], "generate", true) == 0)
                {
                    MarketBuildTraining.Generate(dataDir);
                }
                else if (String.Compare(app.Args[1], "train", true) == 0)
                {
                    MarketTrain.Train(dataDir);
                }
                else if (String.Compare(app.Args[1], "evaluate", true) == 0)
                {
                    MarketEvaluate.Evaluate(dataDir);
                }
                else if (String.Compare(app.Args[1], "prune", true) == 0)
                {
                    {
                        MarketPrune.Incremental(dataDir);
                    }
                }
            }
        }

        #endregion IExample Members
    }
}