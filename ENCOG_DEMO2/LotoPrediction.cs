using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Data.Temporal;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Training;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Neural.Pattern;
using Encog.Util.Arrayutil;
using Encog.Util.CSV;
using Encog.Util.Simple;
using System;
using System.Collections.Generic;
using System.Linq;

namespace LotoPrediction
{
    public class LotoData
    {
        public int Id { get; set; }
        public int DrawNumber { get; set; }
        public int DayOfWeek { get; set; }
        public double NormalizeDayOfWeek { get; set; }
        public double Actual1 { get; set; }
        public double NormalizedActual1 { get; set; }
        public double Actual2 { get; set; }
        public double NormalizedActual2 { get; set; }
        public double Actual3 { get; set; }
        public double NormalizedActual3 { get; set; }
        public double Actual4 { get; set; }
        public double NormalizedActual4 { get; set; }
        public double Actual5 { get; set; }
        public double NormalizedActual5 { get; set; }
        public double Actual6 { get; set; }
        public double NormalizedActual6 { get; set; }
        public double Actual7 { get; set; }
        public double NormalizedActual7 { get; set; }
    }

    public class LotoPrediction
    {
        public int PastWindowSize = 14;
        public const int FutureWindowSize = 1;
        public const double NormalizeHigh = 1.0;
        public const double NormalizeLow = -1.0;
        public double MaxError = 0.01;
        public int LotoNumber = 1;
        public Boolean blnShowConsole = true;
        private float predictionPercent;

        private List<LotoData> data = new List<LotoData>();
        private TemporalMLDataSet trainingSet;
        private BasicNetwork network;
        private NormalizeArray norm0, norm1, norm2, norm3, norm4, norm5, norm6;

        public void Predict()
        {
            //Read Data
            ReadData();

            //Normalization
            NormalizeData();

            ////Generate Training dataset
            GenerateTemporalData();

            ////Create & Train Network
            CreateAndTrainNetwork();

            errorDiagnostic(network, trainingSet, blnShowConsole);

            ////Evaluate Network
            EvaluateNetwork(blnShowConsole);

            ////Predict next value three times
            PredictNetwork();
        }

        private void ReadData()
        {
            var csvreader = new Encog.Util.CSV.ReadCSV(Config.BaseFile.ToString(), true,
                                                        CSVFormat.English);

            int count = 0;
            while (csvreader.Next())
            {
                count++;
                data.Add(new LotoData()
                {
                    Id = count,
                    DrawNumber = csvreader.GetInt("DrawNumber"),
                    //DrawDate = csvreader.Get("DrawDate"),
                    DayOfWeek = (int)(DateTime.ParseExact(csvreader.Get("DrawDate"), "dd/MM/yy", System.Globalization.CultureInfo.InvariantCulture)).DayOfWeek,
                    Actual1 = csvreader.GetInt("N1"),
                    Actual2 = csvreader.GetInt("N2"),
                    Actual3 = csvreader.GetInt("N3"),
                    Actual4 = csvreader.GetInt("N4"),
                    Actual5 = csvreader.GetInt("N5"),
                    Actual6 = csvreader.GetInt("N6"),
                    Actual7 = csvreader.GetInt("N7")
                });
            }
        }

        private void NormalizeData()
        {
            norm0 = new NormalizeArray() { NormalizedHigh = NormalizeHigh, NormalizedLow = NormalizeLow };
            norm1 = new NormalizeArray() { NormalizedHigh = NormalizeHigh, NormalizedLow = NormalizeLow };
            norm2 = new NormalizeArray() { NormalizedHigh = NormalizeHigh, NormalizedLow = NormalizeLow };
            norm3 = new NormalizeArray() { NormalizedHigh = NormalizeHigh, NormalizedLow = NormalizeLow };
            norm4 = new NormalizeArray() { NormalizedHigh = NormalizeHigh, NormalizedLow = NormalizeLow };
            norm5 = new NormalizeArray() { NormalizedHigh = NormalizeHigh, NormalizedLow = NormalizeLow };
            norm6 = new NormalizeArray() { NormalizedHigh = NormalizeHigh, NormalizedLow = NormalizeLow };

            //var Array1 = data.Select(t => t.Actual1).ToArray();
            //var Array2 = data.Select(t => t.Actual2).ToArray();
            //var Array3 = data.Select(t => t.Actual3).ToArray();
            //var Array4 = data.Select(t => t.Actual4).ToArray();
            //var Array5 = data.Select(t => t.Actual5).ToArray();
            //var Array6 = data.Select(t => t.Actual6).ToArray();

            //var combinedArray = Array1.Concat(Array2).ToArray();
            //combinedArray = combinedArray.Concat(Array3).ToArray();
            //combinedArray = combinedArray.Concat(Array4).ToArray();
            //combinedArray = combinedArray.Concat(Array5).ToArray();
            //combinedArray = combinedArray.Concat(Array6).ToArray();

            //var normalizedArray = norm.Process(combinedArray);

            //int arrayCount = Array1.Count();
            //int combinedCount = arrayCount * 6;

            //// get the first arrayCount
            //double[] normalizedArray1 = normalizedArray.Slice(0, arrayCount);
            //double[] normalizedArray2 = normalizedArray.Slice(arrayCount, arrayCount);
            //double[] normalizedArray3 = normalizedArray.Slice(arrayCount * 2, arrayCount);
            //double[] normalizedArray4 = normalizedArray.Slice(arrayCount * 3, arrayCount);
            //double[] normalizedArray5 = normalizedArray.Slice(arrayCount * 4, arrayCount);
            //double[] normalizedArray6 = normalizedArray.Slice(arrayCount * 5, arrayCount);

            //for (int i = 0; i < normalizedArray1.Count(); i++)
            //{
            //    data[i].NormalizedActual1 = normalizedArray1[i];
            //}

            //for (int i = 0; i < normalizedArray2.Count(); i++)
            //{
            //    data[i].NormalizedActual2 = normalizedArray2[i];
            //}

            //for (int i = 0; i < normalizedArray3.Count(); i++)
            //{
            //    data[i].NormalizedActual3 = normalizedArray3[i];
            //}

            //for (int i = 0; i < normalizedArray4.Count(); i++)
            //{
            //    data[i].NormalizedActual4 = normalizedArray4[i];
            //}

            //for (int i = 0; i < normalizedArray5.Count(); i++)
            //{
            //    data[i].NormalizedActual5 = normalizedArray5[i];
            //}

            //for (int i = 0; i < normalizedArray6.Count(); i++)
            //{
            //    data[i].NormalizedActual6 = normalizedArray6[i];
            //}

            //NormalizeDayOfWeek
            var normalizedArray = norm0.Process((data.Select(t => (double)t.DayOfWeek).ToArray()));
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizeDayOfWeek = normalizedArray[i];
            }

            normalizedArray = norm1.Process(data.Select(t => t.Actual1).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedActual1 = normalizedArray[i];
            }

            normalizedArray = norm2.Process(data.Select(t => t.Actual2).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedActual2 = normalizedArray[i];
            }

            normalizedArray = norm3.Process(data.Select(t => t.Actual3).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedActual3 = normalizedArray[i];
            }

            normalizedArray = norm4.Process(data.Select(t => t.Actual4).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedActual4 = normalizedArray[i];
            }

            normalizedArray = norm5.Process(data.Select(t => t.Actual5).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedActual5 = normalizedArray[i];
            }

            normalizedArray = norm6.Process(data.Select(t => t.Actual6).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedActual6 = normalizedArray[i];
            }

            //normalizedArray = norm.Process(data.Select(t => t.Actual7).ToArray());
            //for (int i = 0; i < normalizedArray.Count(); i++)
            //{
            //    data[i].NormalizedActual7 = normalizedArray[i];
            //}
        }

        private void GenerateTemporalData()
        {
            TemporalDataDescription desc1 = null, desc2 = null, desc3 = null, desc4 = null, desc5 = null, desc6 = null;
            //Temporal dataset
            trainingSet = new TemporalMLDataSet(PastWindowSize, FutureWindowSize);

            //Description #0
            var desc0 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
            desc0.Index = 0;
            trainingSet.AddDescription(desc0);

            switch (LotoNumber)
            {
                case 1:
                    desc1 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                    desc1.Index = 1;

                    break;

                case 2:
                    desc2 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                    desc2.Index = 2;

                    break;

                case 3:
                    desc3 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                    desc3.Index = 3;

                    break;

                case 4:
                    desc4 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                    desc4.Index = 4;

                    break;

                case 5:
                    desc5 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                    desc5.Index = 5;

                    break;

                case 6:
                    desc6 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                    desc6.Index = 6;

                    break;

                default:
                    break;
            }

            //Description #1
            if (desc1 == null)
            {
                desc1 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                desc1.Index = 1;
            }

            //Description #2
            if (desc2 == null)
            {
                desc2 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                desc2.Index = 2;
            }

            if (desc3 == null)
            {
                desc3 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                desc3.Index = 3;
            }

            if (desc4 == null)
            {
                desc4 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                desc4.Index = 4;
            }

            if (desc5 == null)
            {
                desc5 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                desc5.Index = 5;
            }

            if (desc6 == null)
            {
                desc6 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                desc6.Index = 6;
            }

            trainingSet.AddDescription(desc1);
            trainingSet.AddDescription(desc2);
            trainingSet.AddDescription(desc3);
            trainingSet.AddDescription(desc4);
            trainingSet.AddDescription(desc5);
            trainingSet.AddDescription(desc6);

            ////Description #7
            //var desc7 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
            //desc7.Index = 7;
            //trainingSet.AddDescription(desc7);

            //Temporal point
            foreach (var item in data)
            {
                var point = new TemporalPoint(7); //1 values
                point.Sequence = item.Id;
                point.Data[0] = item.NormalizeDayOfWeek;
                point.Data[1] = item.NormalizedActual1;
                point.Data[2] = item.NormalizedActual2;
                point.Data[3] = item.NormalizedActual3;
                point.Data[4] = item.NormalizedActual4;
                point.Data[5] = item.NormalizedActual5;
                point.Data[6] = item.NormalizedActual6;
                trainingSet.Points.Add(point);

                //point.Data[0] = item.NormalizedActual1;
                //point.Data[1] = item.NormalizedActual2;
                //point.Data[2] = item.NormalizedActual3;
                //point.Data[3] = item.NormalizedActual4;
                //point.Data[4] = item.NormalizedActual5;
                //point.Data[5] = item.NormalizedActual6;
                //trainingSet.Points.Add(point);
            }

            trainingSet.Generate();
        }

        private void CreateAndTrainNetwork()
        {
            //Alternative patterns
            //var pattern = new FeedForwardPattern
            //var pattern = new ElmanPattern
            //var pattern = new JordanPattern

            var pattern = new FeedForwardPattern
            {
                ActivationFunction = new ActivationTANH(),
                InputNeurons = PastWindowSize,
                OutputNeurons = FutureWindowSize
            };

            pattern.AddHiddenLayer((PastWindowSize + FutureWindowSize) * 2);
            network = (BasicNetwork)pattern.Generate();

            ITrain train = new ResilientPropagation(network, trainingSet);
            EncogUtility.TrainToError(train, MaxError);

            Console.WriteLine("-- End of CreateAndTrainNetwork step --");
        }

        public static void errorDiagnostic(BasicNetwork network, TemporalMLDataSet dataSet, Boolean blnShowConsole)
        {
            int count = 0;
            double totalError = 0;

            if (!blnShowConsole)
                return;

            Console.WriteLine("Network error: " + network.CalculateError(dataSet));

            foreach (IMLDataPair pair in dataSet)
            {
                IMLData actual = network.Compute(pair.Input);
                Console.WriteLine("Evaluating element " + count + " : " + pair.Input.ToString());

                for (int i = 0; i < pair.Ideal.Count; i++)
                {
                    double delta = Math.Abs(actual[i] - pair.Ideal[i]);
                    totalError += delta * delta;
                    count++;
                    double currentError = totalError / count;
                    Console.WriteLine("\tIdeal: " + pair.Ideal[i] + ", Actual: " + actual[i] + ", Delta: " + delta + ", Current Error: " + currentError);
                }
            }
        }

        private void EvaluateNetwork(Boolean blnShowConsole)
        {
            //Console.WriteLine("Neural Network Results:");
            //foreach (IMLDataPair pair in trainingSet)
            //{
            //    INeuralData output = network.Compute(pair.Input);
            //    Console.WriteLine(pair.Input[0] + "," + pair.Input[1]
            //    + ", actual=" + output[0] + ",ideal=" + pair.Ideal[0]);
            //}

            int countPredicted, countUnPredicted;
            Boolean blnPredicted;

            //float predictionPercent;

            int evaluateStart = data.Select(t => t.Id).Min() + PastWindowSize;
            int evaluateStop = data.Select(t => t.Id).Max();

            using (var file = new System.IO.StreamWriter(Config.EvaluationResult.ToString()))
            {
                countPredicted = 0;
                countUnPredicted = 0;

                for (int currentId = evaluateStart; currentId <= evaluateStop; currentId++)
                {
                    //Calculate based on actual data
                    var input = new BasicMLData(PastWindowSize * 7);
                    for (int i = 0; i < PastWindowSize; i++)
                    {
                        input[i * 7] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                         .Select(t => t.NormalizeDayOfWeek).First();

                        input[i * 7 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t.NormalizedActual1).First();

                        input[i * 7 + 2] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t.NormalizedActual2).First();

                        input[i * 7 + 3] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t.NormalizedActual3).First();

                        input[i * 7 + 4] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t.NormalizedActual4).First();

                        input[i * 7 + 5] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t.NormalizedActual5).First();

                        input[i * 7 + 6] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t.NormalizedActual6).First();
                    }

                    var output = network.Compute(input);
                    double normalizedPredicted = output[0];
                    double predicted = 0.0;

                    switch (LotoNumber)
                    {
                        case 1:
                            predicted = Math.Round(norm1.Stats.DeNormalize(normalizedPredicted), 0);
                            break;

                        case 2:
                            predicted = Math.Round(norm2.Stats.DeNormalize(normalizedPredicted), 0);
                            break;

                        case 3:
                            predicted = Math.Round(norm3.Stats.DeNormalize(normalizedPredicted), 0);
                            break;

                        case 4:
                            predicted = Math.Round(norm4.Stats.DeNormalize(normalizedPredicted), 0);
                            break;

                        case 5:
                            predicted = Math.Round(norm5.Stats.DeNormalize(normalizedPredicted), 0);
                            break;

                        case 6:
                            predicted = Math.Round(norm6.Stats.DeNormalize(normalizedPredicted), 0);
                            break;

                        default:
                            break;
                    }

                    //double Actual1 = data.Where(t => t.Id == currentId).Select(t => t.Actual1).First();
                    double actual = 0.0;

                    switch (LotoNumber)
                    {
                        case 1:
                            actual = Math.Round(norm1.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual1).First()), 0);
                            break;

                        case 2:
                            actual = Math.Round(norm2.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual2).First()), 0);
                            break;

                        case 3:
                            actual = Math.Round(norm3.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual3).First()), 0);
                            break;

                        case 4:
                            actual = Math.Round(norm4.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual4).First()), 0);
                            break;

                        case 5:
                            actual = Math.Round(norm5.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual5).First()), 0);
                            break;

                        case 6:
                            actual = Math.Round(norm6.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual6).First()), 0);
                            break;

                        default:
                            break;
                    }

                    int DrawNumber = data.Where(t => t.Id == currentId).Select(t => t.DrawNumber).First();
                    blnPredicted = (actual == predicted) ? true : false;

                    if (blnPredicted)
                        countPredicted++;
                    else
                        countUnPredicted++;

                    string line1 = string.Format("DrawNumber: {0}; Actual: {1}; Predicted: {2}", DrawNumber, actual, predicted);
                    file.WriteLine(line1);

                    if (blnShowConsole)
                        Console.WriteLine(line1);
                }

                predictionPercent = ((float)countPredicted / ((float)countPredicted + (float)countUnPredicted)) * 100;
                string line2 = string.Format("TotalPredictions = {0}, Predicted = {1}, UnPredicted = {2}, Prediction percent = {3:0.00}%",
                                                (countPredicted + countUnPredicted), countPredicted, countUnPredicted, predictionPercent);
                file.WriteLine(line2);
                Console.WriteLine(line2);
            }
        }

        private void PredictNetwork()
        {
            int evaluateStop = data.Select(t => t.Id).Max();

            using (var file = new System.IO.StreamWriter(Config.PredictResult.ToString(), true))
            {
                //Start new
                int currentId = evaluateStop + 1;

                //Calculate based on actual data
                var input = new BasicMLData(PastWindowSize * 7);
                for (int i = 0; i < PastWindowSize; i++)
                {
                    input[i * 7] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                     .Select(t => t.NormalizeDayOfWeek).First();

                    input[i * 7 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                        .Select(t => t.NormalizedActual1).First();

                    input[i * 7 + 2] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                        .Select(t => t.NormalizedActual2).First();

                    input[i * 7 + 3] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                        .Select(t => t.NormalizedActual3).First();

                    input[i * 7 + 4] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                        .Select(t => t.NormalizedActual4).First();

                    input[i * 7 + 5] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                        .Select(t => t.NormalizedActual5).First();

                    input[i * 7 + 6] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                        .Select(t => t.NormalizedActual6).First();
                }
                //Start new - end

                var output = network.Compute(input);
                double normalizedPredicted = output[0];

                double predicted = 0.0;

                switch (LotoNumber)
                {
                    case 1:
                        predicted = Math.Round(norm1.Stats.DeNormalize(normalizedPredicted), 0);
                        break;

                    case 2:
                        predicted = Math.Round(norm2.Stats.DeNormalize(normalizedPredicted), 0);
                        break;

                    case 3:
                        predicted = Math.Round(norm3.Stats.DeNormalize(normalizedPredicted), 0);
                        break;

                    case 4:
                        predicted = Math.Round(norm4.Stats.DeNormalize(normalizedPredicted), 0);
                        break;

                    case 5:
                        predicted = Math.Round(norm5.Stats.DeNormalize(normalizedPredicted), 0);
                        break;

                    case 6:
                        predicted = Math.Round(norm6.Stats.DeNormalize(normalizedPredicted), 0);
                        break;

                    default:
                        break;
                }

                int DrawNumber = data.Where(t => t.Id == (currentId - 1)).Select(t => t.DrawNumber).First() + 1;

                string consoleLine = string.Format("DrawNumber: {0}; LotoNumber: {1}; PastWindowSize = {2}; MaxError: {3}; Predicted: {4}; Prediction percent: {5:0.00}%",
                                            DrawNumber, LotoNumber, PastWindowSize, MaxError, predicted, predictionPercent);

                string line = string.Format("{0}; {1}; {2}; {3}; {4}; {5:0.00}%",
                                            DrawNumber, LotoNumber, PastWindowSize, MaxError, predicted, predictionPercent);

                file.WriteLine(line);
                Console.WriteLine(consoleLine);
            }
        }
    }
}