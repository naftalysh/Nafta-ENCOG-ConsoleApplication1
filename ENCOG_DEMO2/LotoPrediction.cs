﻿using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Data.Temporal;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Training;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Neural.Pattern;
using Encog.Persist;
using Encog.Util.Arrayutil;
using Encog.Util.CSV;
using Encog.Util.Simple;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Configuration;
using System.Reflection;
using System.Threading.Tasks;
using System.Diagnostics;

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
        public double _closedLoopNormalizedActual1 { get; set; }
        public double NA_Actual1 { get; set; }
        public double NormalizedNA_Actual1 { get; set; }
        public double _closedLoopNA_Actual1 { get; set; }
        public double Normalized_closedLoopNA_Actual1 { get; set; }
        public double Actual2 { get; set; }
        public double NormalizedActual2 { get; set; }
        public double _closedLoopNormalizedActual2 { get; set; }
        public double NA_Actual2 { get; set; }
        public double NormalizedNA_Actual2 { get; set; }
        public double _closedLoopNA_Actual2 { get; set; }
        public double Normalized_closedLoopNA_Actual2 { get; set; }
        public double Actual3 { get; set; }
        public double NormalizedActual3 { get; set; }
        public double _closedLoopNormalizedActual3 { get; set; }
        public double NA_Actual3 { get; set; }
        public double NormalizedNA_Actual3 { get; set; }
        public double _closedLoopNA_Actual3 { get; set; }
        public double Normalized_closedLoopNA_Actual3 { get; set; }
        public double Actual4 { get; set; }
        public double NormalizedActual4 { get; set; }
        public double _closedLoopNormalizedActual4 { get; set; }
        public double NA_Actual4 { get; set; }
        public double NormalizedNA_Actual4 { get; set; }
        public double _closedLoopNA_Actual4 { get; set; }
        public double Normalized_closedLoopNA_Actual4 { get; set; }
        public double Actual5 { get; set; }
        public double NormalizedActual5 { get; set; }
        public double _closedLoopNormalizedActual5 { get; set; }
        public double NA_Actual5 { get; set; }
        public double NormalizedNA_Actual5 { get; set; }
        public double _closedLoopNA_Actual5 { get; set; }
        public double Normalized_closedLoopNA_Actual5 { get; set; }
        public double Actual6 { get; set; }
        public double NormalizedActual6 { get; set; }
        public double _closedLoopNormalizedActual6 { get; set; }
        public double NA_Actual6 { get; set; }
        public double NormalizedNA_Actual6 { get; set; }
        public double _closedLoopNA_Actual6 { get; set; }
        public double Normalized_closedLoopNA_Actual6 { get; set; }
        public double Actual7 { get; set; }
        public double NormalizedActual7 { get; set; }
        public double _closedLoopNormalizedActual7 { get; set; }
        public double NA_Actual7 { get; set; }
        public double NormalizedNA_Actual7 { get; set; }
        public double _closedLoopNA_Actual7 { get; set; }
        public double Normalized_closedLoopNA_Actual7 { get; set; }

    }

    public class LotoPrediction
    {
        public string EvaluateFileHeader = @"DrawNumber,LotoNumber,TotalPredictions,countPredicted,countUnPredicted,predictionPercent,CL_countPredicted,CL_countUnPredicted,CL_predictionPercent,countPredicted_Abs1,countUnPredicted_Abs1,predictionPercent_Abs1,CL_countPredicted_Abs1,CL_countUnPredicted_Abs1,CL_predictionPercent_Abs1";
        public string PredictFileHeader = @"DrawNumber,LotoNumber,PastWindowSize,MaxError,predicted,predictionPercent,CL_predictionPercent";

        public int trainPercent = 75;
        public int PastWindowSize = 17;
        public int MaxPastWindowSize = 45;
        public int MinPastWindowSize = 14;
        public const int FutureWindowSize = 1;
        public const double NormalizeHigh = 1.0;
        public const double NormalizeLow = -1.0;
        public double MaxError = 0.01;
        public int LotoNumber = 1;
        public Boolean blnShowConsole = true;
        private float predictionPercent;
        private float CL_predictionPercent;
        private float predictionPercent_Abs1;       // Abs(Actual-Predicted) <= 1
        private float CL_predictionPercent_Abs1;


        private float MAX_predictionPercent = 0;
        private float MAX_CL_predictionPercent = 0;
        private float MAX_predictionPercent_Abs1 = 0;       
        private float MAX_CL_predictionPercent_Abs1 = 0;

        private float MAX_predictionPercent_N1 = 0;
        private float MAX_CL_predictionPercent_N1 = 0;
        private float MAX_predictionPercent_Abs1_N1 = 0;
        private float MAX_CL_predictionPercent_Abs1_N1 = 0;

        private float MAX_predictionPercent_N2 = 0;
        private float MAX_CL_predictionPercent_N2 = 0;
        private float MAX_predictionPercent_Abs1_N2 = 0;
        private float MAX_CL_predictionPercent_Abs1_N2 = 0;

        private float MAX_predictionPercent_N3 = 0;
        private float MAX_CL_predictionPercent_N3 = 0;
        private float MAX_predictionPercent_Abs1_N3 = 0;
        private float MAX_CL_predictionPercent_Abs1_N3 = 0;

        private float MAX_predictionPercent_N4 = 0;
        private float MAX_CL_predictionPercent_N4 = 0;
        private float MAX_predictionPercent_Abs1_N4 = 0;
        private float MAX_CL_predictionPercent_Abs1_N4 = 0;

        private float MAX_predictionPercent_N5 = 0;
        private float MAX_CL_predictionPercent_N5 = 0;
        private float MAX_predictionPercent_Abs1_N5 = 0;
        private float MAX_CL_predictionPercent_Abs1_N5 = 0;

        private float MAX_predictionPercent_N6 = 0;
        private float MAX_CL_predictionPercent_N6 = 0;
        private float MAX_predictionPercent_Abs1_N6 = 0;
        private float MAX_CL_predictionPercent_Abs1_N6 = 0;

        private float MAX_predictionPercent_N7 = 0;
        private float MAX_CL_predictionPercent_N7 = 0;
        private float MAX_predictionPercent_Abs1_N7 = 0;
        private float MAX_CL_predictionPercent_Abs1_N7 = 0;

        //MIN Handling
        private float MIN_predictionPercent = 0;
        private float MIN_CL_predictionPercent = 0;
        private float MIN_predictionPercent_Abs1 = 0;
        private float MIN_CL_predictionPercent_Abs1 = 0;

        private float MIN_predictionPercent_N1 = 0;
        private float MIN_CL_predictionPercent_N1 = 0;
        private float MIN_predictionPercent_Abs1_N1 = 0;
        private float MIN_CL_predictionPercent_Abs1_N1 = 0;

        private float MIN_predictionPercent_N2 = 0;
        private float MIN_CL_predictionPercent_N2 = 0;
        private float MIN_predictionPercent_Abs1_N2 = 0;
        private float MIN_CL_predictionPercent_Abs1_N2 = 0;

        private float MIN_predictionPercent_N3 = 0;
        private float MIN_CL_predictionPercent_N3 = 0;
        private float MIN_predictionPercent_Abs1_N3 = 0;
        private float MIN_CL_predictionPercent_Abs1_N3 = 0;

        private float MIN_predictionPercent_N4 = 0;
        private float MIN_CL_predictionPercent_N4 = 0;
        private float MIN_predictionPercent_Abs1_N4 = 0;
        private float MIN_CL_predictionPercent_Abs1_N4 = 0;

        private float MIN_predictionPercent_N5 = 0;
        private float MIN_CL_predictionPercent_N5 = 0;
        private float MIN_predictionPercent_Abs1_N5 = 0;
        private float MIN_CL_predictionPercent_Abs1_N5 = 0;

        private float MIN_predictionPercent_N6 = 0;
        private float MIN_CL_predictionPercent_N6 = 0;
        private float MIN_predictionPercent_Abs1_N6 = 0;
        private float MIN_CL_predictionPercent_Abs1_N6 = 0;

        private float MIN_predictionPercent_N7 = 0;
        private float MIN_CL_predictionPercent_N7 = 0;
        private float MIN_predictionPercent_Abs1_N7 = 0;
        private float MIN_CL_predictionPercent_Abs1_N7 = 0;
        //MIN Handling

        double predicted = 0.0;
        double _closedLoop_predicted = 0.0;



        public int TrainStart;
        public int TrainEnd;
        public int EvaluateStart;
        public int EvaluateEnd;
        int TotalNumOfIterations;

        private List<LotoData> data = new List<LotoData>();
        private TemporalMLDataSet trainingSet;
        private BasicNetwork network;
        private NormalizeArray norm0, norm1, norm2, norm3, norm4, norm5, norm6, norm7, norm8, norm9, norm10, norm11, norm12, norm13, norm14;
        double[] normalizedArray;
        private NumbersAnalysis NA = new NumbersAnalysis();


        
        /// </summary>
        /// <param name="blnSave"></param>
        /// <param name="savedFilename"></param>
        public void SaveLoadNetwork(Boolean blnSave, string savedFilename)
        {
            if (blnSave) 
                EncogDirectoryPersistence.SaveObject(new FileInfo(savedFilename), network);
            else
                network = (BasicNetwork)EncogDirectoryPersistence.LoadObject(new FileInfo(savedFilename));
        }



        private static string GetSetting(string key)
        {
            string filePath = System.IO.Path.GetFullPath("settings.app.config");
            var map = new ExeConfigurationFileMap { ExeConfigFilename = filePath };
            Configuration config = ConfigurationManager.OpenMappedExeConfiguration(map, ConfigurationUserLevel.None);

            var entry = config.AppSettings.Settings[key];
            if (entry == null)
                return null;
            else
                return entry.Value;
        }

        
        
        public static void Set(string key, string value)
        {
            string filePath = System.IO.Path.GetFullPath("settings.app.config");
            var map = new ExeConfigurationFileMap { ExeConfigFilename = filePath };
            Configuration config = ConfigurationManager.OpenMappedExeConfiguration(map, ConfigurationUserLevel.None);

            var entry = config.AppSettings.Settings[key];
            if (entry == null)
                config.AppSettings.Settings.Add(key, value);
            else
                config.AppSettings.Settings[key].Value = value;

            config.Save(ConfigurationSaveMode.Modified);
            // Force a reload of a changed section.
            ConfigurationManager.RefreshSection("appSettings");
        }

        // Determines if a key exists within the App.config



        //
        // Report the predictions according to "appSettings" Max values per each Loto number.
        //
        public void Report()
        {

            //Read Data
            ReadData();

            //Normalization
            NormalizeData();



            //MIN Reporting
            //Read MIN prections found from "appSettings " file.
            string strMIN_predictionPercent = GetSetting("MIN_predictionPercent");
            string strMIN_CL_predictionPercent = GetSetting("MIN_CL_predictionPercent");
            string strMIN_predictionPercent_Abs1 = GetSetting("MIN_predictionPercent_Abs1");
            string strMIN_CL_predictionPercent_Abs1 = GetSetting("MIN_CL_predictionPercent_Abs1");

            string strMIN_predictionPercent_N1 = GetSetting("MIN_predictionPercent_N1");
            string strMIN_CL_predictionPercent_N1 = GetSetting("MIN_CL_predictionPercent_N1");
            string strMIN_predictionPercent_Abs1_N1 = GetSetting("MIN_predictionPercent_Abs1_N1");
            string strMIN_CL_predictionPercent_Abs1_N1 = GetSetting("MIN_CL_predictionPercent_Abs1_N1");

            string strMIN_predictionPercent_N2 = GetSetting("MIN_predictionPercent_N2");
            string strMIN_CL_predictionPercent_N2 = GetSetting("MIN_CL_predictionPercent_N2");
            string strMIN_predictionPercent_Abs1_N2 = GetSetting("MIN_predictionPercent_Abs1_N2");
            string strMIN_CL_predictionPercent_Abs1_N2 = GetSetting("MIN_CL_predictionPercent_Abs1_N2");

            string strMIN_predictionPercent_N3 = GetSetting("MIN_predictionPercent_N3");
            string strMIN_CL_predictionPercent_N3 = GetSetting("MIN_CL_predictionPercent_N3");
            string strMIN_predictionPercent_Abs1_N3 = GetSetting("MIN_predictionPercent_Abs1_N3");
            string strMIN_CL_predictionPercent_Abs1_N3 = GetSetting("MIN_CL_predictionPercent_Abs1_N3");

            string strMIN_predictionPercent_N4 = GetSetting("MIN_predictionPercent_N4");
            string strMIN_CL_predictionPercent_N4 = GetSetting("MIN_CL_predictionPercent_N4");
            string strMIN_predictionPercent_Abs1_N4 = GetSetting("MIN_predictionPercent_Abs1_N4");
            string strMIN_CL_predictionPercent_Abs1_N4 = GetSetting("MIN_CL_predictionPercent_Abs1_N4");

            string strMIN_predictionPercent_N5 = GetSetting("MIN_predictionPercent_N5");
            string strMIN_CL_predictionPercent_N5 = GetSetting("MIN_CL_predictionPercent_N5");
            string strMIN_predictionPercent_Abs1_N5 = GetSetting("MIN_predictionPercent_Abs1_N5");
            string strMIN_CL_predictionPercent_Abs1_N5 = GetSetting("MIN_CL_predictionPercent_Abs1_N5");

            string strMIN_predictionPercent_N6 = GetSetting("MIN_predictionPercent_N6");
            string strMIN_CL_predictionPercent_N6 = GetSetting("MIN_CL_predictionPercent_N6");
            string strMIN_predictionPercent_Abs1_N6 = GetSetting("MIN_predictionPercent_Abs1_N6");
            string strMIN_CL_predictionPercent_Abs1_N6 = GetSetting("MIN_CL_predictionPercent_Abs1_N6");

            string strMIN_predictionPercent_N7 = GetSetting("MIN_predictionPercent_N7");
            string strMIN_CL_predictionPercent_N7 = GetSetting("MIN_CL_predictionPercent_N7");
            string strMIN_predictionPercent_Abs1_N7 = GetSetting("MIN_predictionPercent_Abs1_N7");
            string strMIN_CL_predictionPercent_Abs1_N7 = GetSetting("MIN_CL_predictionPercent_Abs1_N7");

            // Convert to float numbers
            MIN_predictionPercent = float.Parse(GetSetting("MIN_predictionPercent"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent = float.Parse(GetSetting("MIN_CL_predictionPercent"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_predictionPercent_Abs1 = float.Parse(GetSetting("MIN_predictionPercent_Abs1"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent_Abs1 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1"), System.Globalization.CultureInfo.InvariantCulture);

            MIN_predictionPercent_N1 = float.Parse(GetSetting("MIN_predictionPercent_N1"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent_N1 = float.Parse(GetSetting("MIN_CL_predictionPercent_N1"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_predictionPercent_Abs1_N1 = float.Parse(GetSetting("MIN_predictionPercent_Abs1_N1"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent_Abs1_N1 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1_N1"), System.Globalization.CultureInfo.InvariantCulture);

            MIN_predictionPercent_N2 = float.Parse(GetSetting("MIN_predictionPercent_N2"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent_N2 = float.Parse(GetSetting("MIN_CL_predictionPercent_N2"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_predictionPercent_Abs1_N2 = float.Parse(GetSetting("MIN_predictionPercent_Abs1_N2"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent_Abs1_N2 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1_N2"), System.Globalization.CultureInfo.InvariantCulture);

            MIN_predictionPercent_N3 = float.Parse(GetSetting("MIN_predictionPercent_N3"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent_N3 = float.Parse(GetSetting("MIN_CL_predictionPercent_N3"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_predictionPercent_Abs1_N3 = float.Parse(GetSetting("MIN_predictionPercent_Abs1_N3"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent_Abs1_N3 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1_N3"), System.Globalization.CultureInfo.InvariantCulture);

            MIN_predictionPercent_N4 = float.Parse(GetSetting("MIN_predictionPercent_N4"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent_N4 = float.Parse(GetSetting("MIN_CL_predictionPercent_N4"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_predictionPercent_Abs1_N4 = float.Parse(GetSetting("MIN_predictionPercent_Abs1_N4"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent_Abs1_N4 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1_N4"), System.Globalization.CultureInfo.InvariantCulture);

            MIN_predictionPercent_N5 = float.Parse(GetSetting("MIN_predictionPercent_N5"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent_N5 = float.Parse(GetSetting("MIN_CL_predictionPercent_N5"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_predictionPercent_Abs1_N5 = float.Parse(GetSetting("MIN_predictionPercent_Abs1_N5"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent_Abs1_N5 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1_N5"), System.Globalization.CultureInfo.InvariantCulture);

            MIN_predictionPercent_N6 = float.Parse(GetSetting("MIN_predictionPercent_N6"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent_N6 = float.Parse(GetSetting("MIN_CL_predictionPercent_N6"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_predictionPercent_Abs1_N6 = float.Parse(GetSetting("MIN_predictionPercent_Abs1_N6"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent_Abs1_N6 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1_N6"), System.Globalization.CultureInfo.InvariantCulture);

            MIN_predictionPercent_N7 = float.Parse(GetSetting("MIN_predictionPercent_N7"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent_N7 = float.Parse(GetSetting("MIN_CL_predictionPercent_N7"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_predictionPercent_Abs1_N7 = float.Parse(GetSetting("MIN_predictionPercent_Abs1_N7"), System.Globalization.CultureInfo.InvariantCulture);
            MIN_CL_predictionPercent_Abs1_N7 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1_N7"), System.Globalization.CultureInfo.InvariantCulture);


            string N1PredictionsMIN, N2PredictionsMIN, N3PredictionsMIN, N4PredictionsMIN, N5PredictionsMIN, N6PredictionsMIN, N7PredictionsMIN;
            N1PredictionsMIN = N2PredictionsMIN = N3PredictionsMIN = N4PredictionsMIN = N5PredictionsMIN = N6PredictionsMIN = N7PredictionsMIN = "";
            //predicted = 0.0;
            //_closedLoop_predicted = 0.0;

            double flN1, flN2, flN3; //used for Abs1 PredictionsMIN
            double AbsNumber;
            List<double> AbsList = new List<double>(3) {1,1,1};

            // Iterate through all LotoNumber's
            for (int i = 1; i <= 7; i++)
            {
                LotoNumber = i;
                switch (LotoNumber)
                {
                    case 1:
                        if (MIN_CL_predictionPercent_N1 > MIN_predictionPercent_N1)
                        {
                            SaveLoadNetwork(false, Config.MIN_predictionPercentFile_N1.ToString()); //Load saved network
                            PredictNetworkNew();
                            N1PredictionsMIN = N1PredictionsMIN + "N1 Predict: " + predicted + " ;";
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MIN_CL_predictionPercentFile_N1.ToString()); //Load saved network
                            PredictNetworkNew();
                            N1PredictionsMIN = N1PredictionsMIN + "N1 Predict: " + _closedLoop_predicted + " ;";
                        }

                        if (MIN_CL_predictionPercent_Abs1_N1 > MIN_predictionPercent_Abs1_N1)
                        {
                            SaveLoadNetwork(false, Config.MIN_predictionPercent_Abs1File_N1.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = predicted;
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MIN_CL_predictionPercent_Abs1File_N1.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = _closedLoop_predicted;
                        }

                        N1PredictionsMIN = N1PredictionsMIN + "N1 Abs Predict: (";

                        flN1 = Math.Max(1.0, AbsNumber - 1.0);
                        flN2 = AbsNumber;
                        flN3 = AbsNumber + 1;

                        AbsList[0] = flN1; AbsList[1] = flN2; AbsList[2] = flN3;

                        foreach (var item in AbsList.Distinct())
                        {
                            N1PredictionsMIN = N1PredictionsMIN + item.ToString() + " ";
                        }

                        N1PredictionsMIN = N1PredictionsMIN + ")";

                        break;

                    case 2:
                        if (MIN_CL_predictionPercent_N2 > MIN_predictionPercent_N2)
                        {
                            SaveLoadNetwork(false, Config.MIN_predictionPercentFile_N2.ToString()); //Load saved network
                            PredictNetworkNew();
                            N2PredictionsMIN = N2PredictionsMIN + "N2 Predict: " + predicted + " ;";
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MIN_CL_predictionPercentFile_N2.ToString()); //Load saved network
                            PredictNetworkNew();
                            N2PredictionsMIN = N2PredictionsMIN + "N2 Predict: " + _closedLoop_predicted + " ;";
                        }

                        if (MIN_CL_predictionPercent_Abs1_N2 > MIN_predictionPercent_Abs1_N2)
                        {
                            SaveLoadNetwork(false, Config.MIN_predictionPercent_Abs1File_N2.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = predicted;
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MIN_CL_predictionPercent_Abs1File_N2.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = _closedLoop_predicted;
                        }

                        N2PredictionsMIN = N2PredictionsMIN + "N2 Abs Predict: (";

                        flN1 = Math.Max(1.0, AbsNumber - 1.0);
                        flN2 = AbsNumber;
                        flN3 = AbsNumber + 1;

                        AbsList[0] = flN1; AbsList[1] = flN2; AbsList[2] = flN3;

                        foreach (var item in AbsList.Distinct())
                        {
                            N2PredictionsMIN = N2PredictionsMIN + item.ToString() + " ";
                        }

                        N2PredictionsMIN = N2PredictionsMIN + ")";

                        break;

                    case 3:
                        if (MIN_CL_predictionPercent_N3 > MIN_predictionPercent_N3)
                        {
                            SaveLoadNetwork(false, Config.MIN_predictionPercentFile_N3.ToString()); //Load saved network
                            PredictNetworkNew();
                            N3PredictionsMIN = N3PredictionsMIN + "N3 Predict: " + predicted + " ;";
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MIN_CL_predictionPercentFile_N3.ToString()); //Load saved network
                            PredictNetworkNew();
                            N3PredictionsMIN = N3PredictionsMIN + "N3 Predict: " + _closedLoop_predicted + " ;";
                        }

                        if (MIN_CL_predictionPercent_Abs1_N3 > MIN_predictionPercent_Abs1_N3)
                        {
                            SaveLoadNetwork(false, Config.MIN_predictionPercent_Abs1File_N3.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = predicted;
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MIN_CL_predictionPercent_Abs1File_N3.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = _closedLoop_predicted;
                        }

                        N3PredictionsMIN = N3PredictionsMIN + "N3 Abs Predict: (";

                        flN1 = Math.Max(1.0, AbsNumber - 1.0);
                        flN2 = AbsNumber;
                        flN3 = AbsNumber + 1;

                        AbsList[0] = flN1; AbsList[1] = flN2; AbsList[2] = flN3;

                        foreach (var item in AbsList.Distinct())
                        {
                            N3PredictionsMIN = N3PredictionsMIN + item.ToString() + " ";
                        }

                        N3PredictionsMIN = N3PredictionsMIN + ")";

                        break;

                    case 4:
                        if (MIN_CL_predictionPercent_N4 > MIN_predictionPercent_N4)
                        {
                            SaveLoadNetwork(false, Config.MIN_predictionPercentFile_N4.ToString()); //Load saved network
                            PredictNetworkNew();
                            N4PredictionsMIN = N4PredictionsMIN + "N4 Predict: " + predicted + " ;";
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MIN_CL_predictionPercentFile_N4.ToString()); //Load saved network
                            PredictNetworkNew();
                            N4PredictionsMIN = N4PredictionsMIN + "N4 Predict: " + _closedLoop_predicted + " ;";
                        }

                        if (MIN_CL_predictionPercent_Abs1_N4 > MIN_predictionPercent_Abs1_N4)
                        {
                            SaveLoadNetwork(false, Config.MIN_predictionPercent_Abs1File_N4.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = predicted;
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MIN_CL_predictionPercent_Abs1File_N4.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = _closedLoop_predicted;
                        }

                        N4PredictionsMIN = N4PredictionsMIN + "N4 Abs Predict: (";

                        flN1 = Math.Max(1.0, AbsNumber - 1.0);
                        flN2 = AbsNumber;
                        flN3 = AbsNumber + 1;

                        AbsList[0] = flN1; AbsList[1] = flN2; AbsList[2] = flN3;

                        foreach (var item in AbsList.Distinct())
                        {
                            N4PredictionsMIN = N4PredictionsMIN + item.ToString() + " ";
                        }

                        N4PredictionsMIN = N4PredictionsMIN + ")";

                        break;

                    case 5:
                        if (MIN_CL_predictionPercent_N5 > MIN_predictionPercent_N5)
                        {
                            SaveLoadNetwork(false, Config.MIN_predictionPercentFile_N5.ToString()); //Load saved network
                            PredictNetworkNew();
                            N5PredictionsMIN = N5PredictionsMIN + "N5 Predict: " + predicted + " ;";
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MIN_CL_predictionPercentFile_N5.ToString()); //Load saved network
                            PredictNetworkNew();
                            N5PredictionsMIN = N5PredictionsMIN + "N5 Predict: " + _closedLoop_predicted + " ;";
                        }

                        if (MIN_CL_predictionPercent_Abs1_N5 > MIN_predictionPercent_Abs1_N5)
                        {
                            SaveLoadNetwork(false, Config.MIN_predictionPercent_Abs1File_N5.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = predicted;
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MIN_CL_predictionPercent_Abs1File_N5.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = _closedLoop_predicted;
                        }

                        N5PredictionsMIN = N5PredictionsMIN + "N5 Abs Predict: (";

                        flN1 = Math.Max(1.0, AbsNumber - 1.0);
                        flN2 = AbsNumber;
                        flN3 = AbsNumber + 1;

                        AbsList[0] = flN1; AbsList[1] = flN2; AbsList[2] = flN3;

                        foreach (var item in AbsList.Distinct())
                        {
                            N5PredictionsMIN = N5PredictionsMIN + item.ToString() + " ";
                        }

                        N5PredictionsMIN = N5PredictionsMIN + ")";

                        break;

                    case 6:
                        if (MIN_CL_predictionPercent_N6 > MIN_predictionPercent_N6)
                        {
                            SaveLoadNetwork(false, Config.MIN_predictionPercentFile_N6.ToString()); //Load saved network
                            PredictNetworkNew();
                            N6PredictionsMIN = N6PredictionsMIN + "N6 Predict: " + predicted + " ;";
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MIN_CL_predictionPercentFile_N6.ToString()); //Load saved network
                            PredictNetworkNew();
                            N6PredictionsMIN = N6PredictionsMIN + "N6 Predict: " + _closedLoop_predicted + " ;";
                        }

                        if (MIN_CL_predictionPercent_Abs1_N6 > MIN_predictionPercent_Abs1_N6)
                        {
                            SaveLoadNetwork(false, Config.MIN_predictionPercent_Abs1File_N6.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = predicted;
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MIN_CL_predictionPercent_Abs1File_N6.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = _closedLoop_predicted;
                        }

                        N6PredictionsMIN = N6PredictionsMIN + "N6 Abs Predict: (";

                        flN1 = Math.Max(1.0, AbsNumber - 1.0);
                        flN2 = AbsNumber;
                        flN3 = AbsNumber + 1;

                        AbsList[0] = flN1; AbsList[1] = flN2; AbsList[2] = flN3;

                        foreach (var item in AbsList.Distinct())
                        {
                            N6PredictionsMIN = N6PredictionsMIN + item.ToString() + " ";
                        }

                        N6PredictionsMIN = N6PredictionsMIN + ")";

                        break;

                    case 7:
                        if (MIN_CL_predictionPercent_N7 > MIN_predictionPercent_N7)
                        {
                            SaveLoadNetwork(false, Config.MIN_predictionPercentFile_N7.ToString()); //Load saved network
                            PredictNetworkNew();
                            N7PredictionsMIN = N7PredictionsMIN + "N7 Predict: " + predicted + " ;";
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MIN_CL_predictionPercentFile_N7.ToString()); //Load saved network
                            PredictNetworkNew();
                            N7PredictionsMIN = N7PredictionsMIN + "N7 Predict: " + _closedLoop_predicted + " ;";
                        }

                        if (MIN_CL_predictionPercent_Abs1_N7 > MIN_predictionPercent_Abs1_N7)
                        {
                            SaveLoadNetwork(false, Config.MIN_predictionPercent_Abs1File_N7.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = predicted;
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MIN_CL_predictionPercent_Abs1File_N7.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = _closedLoop_predicted;
                        }

                        N7PredictionsMIN = N7PredictionsMIN + "N7 Abs Predict: (";

                        flN1 = Math.Max(1.0, AbsNumber - 1.0);
                        flN2 = AbsNumber;
                        flN3 = AbsNumber + 1;

                        AbsList[0] = flN1; AbsList[1] = flN2; AbsList[2] = flN3;

                        foreach (var item in AbsList.Distinct())
                        {
                            N7PredictionsMIN = N7PredictionsMIN + item.ToString() + " ";
                        }

                        N7PredictionsMIN = N7PredictionsMIN + ")";

                        break;

                    default:
                        break;
                }
            }


            //MIN Reporting

            //MAX Reporting
            //Read MAX prections found from "appSettings " file.
            string strMAX_predictionPercent = GetSetting("MAX_predictionPercent");
            string strMAX_CL_predictionPercent = GetSetting("MAX_CL_predictionPercent");
            string strMAX_predictionPercent_Abs1 = GetSetting("MAX_predictionPercent_Abs1");
            string strMAX_CL_predictionPercent_Abs1 = GetSetting("MAX_CL_predictionPercent_Abs1");

            string strMAX_predictionPercent_N1 = GetSetting("MAX_predictionPercent_N1");
            string strMAX_CL_predictionPercent_N1 = GetSetting("MAX_CL_predictionPercent_N1");
            string strMAX_predictionPercent_Abs1_N1 = GetSetting("MAX_predictionPercent_Abs1_N1");
            string strMAX_CL_predictionPercent_Abs1_N1 = GetSetting("MAX_CL_predictionPercent_Abs1_N1");

            string strMAX_predictionPercent_N2 = GetSetting("MAX_predictionPercent_N2");
            string strMAX_CL_predictionPercent_N2 = GetSetting("MAX_CL_predictionPercent_N2");
            string strMAX_predictionPercent_Abs1_N2 = GetSetting("MAX_predictionPercent_Abs1_N2");
            string strMAX_CL_predictionPercent_Abs1_N2 = GetSetting("MAX_CL_predictionPercent_Abs1_N2");

            string strMAX_predictionPercent_N3 = GetSetting("MAX_predictionPercent_N3");
            string strMAX_CL_predictionPercent_N3 = GetSetting("MAX_CL_predictionPercent_N3");
            string strMAX_predictionPercent_Abs1_N3 = GetSetting("MAX_predictionPercent_Abs1_N3");
            string strMAX_CL_predictionPercent_Abs1_N3 = GetSetting("MAX_CL_predictionPercent_Abs1_N3");

            string strMAX_predictionPercent_N4 = GetSetting("MAX_predictionPercent_N4");
            string strMAX_CL_predictionPercent_N4 = GetSetting("MAX_CL_predictionPercent_N4");
            string strMAX_predictionPercent_Abs1_N4 = GetSetting("MAX_predictionPercent_Abs1_N4");
            string strMAX_CL_predictionPercent_Abs1_N4 = GetSetting("MAX_CL_predictionPercent_Abs1_N4");

            string strMAX_predictionPercent_N5 = GetSetting("MAX_predictionPercent_N5");
            string strMAX_CL_predictionPercent_N5 = GetSetting("MAX_CL_predictionPercent_N5");
            string strMAX_predictionPercent_Abs1_N5 = GetSetting("MAX_predictionPercent_Abs1_N5");
            string strMAX_CL_predictionPercent_Abs1_N5 = GetSetting("MAX_CL_predictionPercent_Abs1_N5");

            string strMAX_predictionPercent_N6 = GetSetting("MAX_predictionPercent_N6");
            string strMAX_CL_predictionPercent_N6 = GetSetting("MAX_CL_predictionPercent_N6");
            string strMAX_predictionPercent_Abs1_N6 = GetSetting("MAX_predictionPercent_Abs1_N6");
            string strMAX_CL_predictionPercent_Abs1_N6 = GetSetting("MAX_CL_predictionPercent_Abs1_N6");

            string strMAX_predictionPercent_N7 = GetSetting("MAX_predictionPercent_N7");
            string strMAX_CL_predictionPercent_N7 = GetSetting("MAX_CL_predictionPercent_N7");
            string strMAX_predictionPercent_Abs1_N7 = GetSetting("MAX_predictionPercent_Abs1_N7");
            string strMAX_CL_predictionPercent_Abs1_N7 = GetSetting("MAX_CL_predictionPercent_Abs1_N7");

            // Convert to float numbers
            MAX_predictionPercent = float.Parse(GetSetting("MAX_predictionPercent"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent = float.Parse(GetSetting("MAX_CL_predictionPercent"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_predictionPercent_Abs1 = float.Parse(GetSetting("MAX_predictionPercent_Abs1"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent_Abs1 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1"), System.Globalization.CultureInfo.InvariantCulture);

            MAX_predictionPercent_N1 = float.Parse(GetSetting("MAX_predictionPercent_N1"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent_N1 = float.Parse(GetSetting("MAX_CL_predictionPercent_N1"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_predictionPercent_Abs1_N1 = float.Parse(GetSetting("MAX_predictionPercent_Abs1_N1"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent_Abs1_N1 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1_N1"), System.Globalization.CultureInfo.InvariantCulture);

            MAX_predictionPercent_N2 = float.Parse(GetSetting("MAX_predictionPercent_N2"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent_N2 = float.Parse(GetSetting("MAX_CL_predictionPercent_N2"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_predictionPercent_Abs1_N2 = float.Parse(GetSetting("MAX_predictionPercent_Abs1_N2"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent_Abs1_N2 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1_N2"), System.Globalization.CultureInfo.InvariantCulture);

            MAX_predictionPercent_N3 = float.Parse(GetSetting("MAX_predictionPercent_N3"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent_N3 = float.Parse(GetSetting("MAX_CL_predictionPercent_N3"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_predictionPercent_Abs1_N3 = float.Parse(GetSetting("MAX_predictionPercent_Abs1_N3"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent_Abs1_N3 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1_N3"), System.Globalization.CultureInfo.InvariantCulture);

            MAX_predictionPercent_N4 = float.Parse(GetSetting("MAX_predictionPercent_N4"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent_N4 = float.Parse(GetSetting("MAX_CL_predictionPercent_N4"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_predictionPercent_Abs1_N4 = float.Parse(GetSetting("MAX_predictionPercent_Abs1_N4"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent_Abs1_N4 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1_N4"), System.Globalization.CultureInfo.InvariantCulture);

            MAX_predictionPercent_N5 = float.Parse(GetSetting("MAX_predictionPercent_N5"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent_N5 = float.Parse(GetSetting("MAX_CL_predictionPercent_N5"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_predictionPercent_Abs1_N5 = float.Parse(GetSetting("MAX_predictionPercent_Abs1_N5"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent_Abs1_N5 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1_N5"), System.Globalization.CultureInfo.InvariantCulture);

            MAX_predictionPercent_N6 = float.Parse(GetSetting("MAX_predictionPercent_N6"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent_N6 = float.Parse(GetSetting("MAX_CL_predictionPercent_N6"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_predictionPercent_Abs1_N6 = float.Parse(GetSetting("MAX_predictionPercent_Abs1_N6"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent_Abs1_N6 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1_N6"), System.Globalization.CultureInfo.InvariantCulture);

            MAX_predictionPercent_N7 = float.Parse(GetSetting("MAX_predictionPercent_N7"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent_N7 = float.Parse(GetSetting("MAX_CL_predictionPercent_N7"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_predictionPercent_Abs1_N7 = float.Parse(GetSetting("MAX_predictionPercent_Abs1_N7"), System.Globalization.CultureInfo.InvariantCulture);
            MAX_CL_predictionPercent_Abs1_N7 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1_N7"), System.Globalization.CultureInfo.InvariantCulture);


            string N1PredictionsMAX, N2PredictionsMAX, N3PredictionsMAX, N4PredictionsMAX, N5PredictionsMAX, N6PredictionsMAX, N7PredictionsMAX;
            N1PredictionsMAX = N2PredictionsMAX = N3PredictionsMAX = N4PredictionsMAX = N5PredictionsMAX = N6PredictionsMAX = N7PredictionsMAX = "";
            //predicted = 0.0;
            //_closedLoop_predicted = 0.0;


            // Iterate through all LotoNumber's
            for (int i = 1; i <= 7; i++)
            {
                LotoNumber = i;
                switch (LotoNumber)
                {
                    case 1:
                        if (MAX_CL_predictionPercent_N1 <= MAX_predictionPercent_N1)
                        {
                            SaveLoadNetwork(false, Config.MAX_predictionPercentFile_N1.ToString()); //Load saved network
                            PredictNetworkNew();
                            N1PredictionsMAX = N1PredictionsMAX + "N1 Predict: " + predicted + " ;";
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MAX_CL_predictionPercentFile_N1.ToString()); //Load saved network
                            PredictNetworkNew();
                            N1PredictionsMAX = N1PredictionsMAX + "N1 Predict: " + _closedLoop_predicted + " ;";
                        }

                        if (MAX_CL_predictionPercent_Abs1_N1 <= MAX_predictionPercent_Abs1_N1)
                        {
                            SaveLoadNetwork(false, Config.MAX_predictionPercent_Abs1File_N1.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = predicted;
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MAX_CL_predictionPercent_Abs1File_N1.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = _closedLoop_predicted;
                        }

                        N1PredictionsMAX = N1PredictionsMAX + "N1 Abs Predict: (";

                        flN1 = Math.Max(1.0, AbsNumber - 1.0);
                        flN2 = AbsNumber;
                        flN3 = AbsNumber + 1;

                        AbsList[0] = flN1; AbsList[1] = flN2; AbsList[2] = flN3;

                        foreach (var item in AbsList.Distinct())
                        {
                            N1PredictionsMAX = N1PredictionsMAX + item.ToString() + " ";
                        }

                        N1PredictionsMAX = N1PredictionsMAX + ")";

                        break;

                    case 2:
                        if (MAX_CL_predictionPercent_N2 <= MAX_predictionPercent_N2)
                        {
                            SaveLoadNetwork(false, Config.MAX_predictionPercentFile_N2.ToString()); //Load saved network
                            PredictNetworkNew();
                            N2PredictionsMAX = N2PredictionsMAX + "N2 Predict: " + predicted + " ;";
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MAX_CL_predictionPercentFile_N2.ToString()); //Load saved network
                            PredictNetworkNew();
                            N2PredictionsMAX = N2PredictionsMAX + "N2 Predict: " + _closedLoop_predicted + " ;";
                        }

                        if (MAX_CL_predictionPercent_Abs1_N2 <= MAX_predictionPercent_Abs1_N2)
                        {
                            SaveLoadNetwork(false, Config.MAX_predictionPercent_Abs1File_N2.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = predicted;
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MAX_CL_predictionPercent_Abs1File_N2.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = _closedLoop_predicted;
                        }

                        N2PredictionsMAX = N2PredictionsMAX + "N2 Abs Predict: (";

                        flN1 = Math.Max(1.0, AbsNumber - 1.0);
                        flN2 = AbsNumber;
                        flN3 = AbsNumber + 1;

                        AbsList[0] = flN1; AbsList[1] = flN2; AbsList[2] = flN3;

                        foreach (var item in AbsList.Distinct())
                        {
                            N2PredictionsMAX = N2PredictionsMAX + item.ToString() + " ";
                        }

                        N2PredictionsMAX = N2PredictionsMAX + ")";

                        break;

                    case 3:
                        if (MAX_CL_predictionPercent_N3 <= MAX_predictionPercent_N3)
                        {
                            SaveLoadNetwork(false, Config.MAX_predictionPercentFile_N3.ToString()); //Load saved network
                            PredictNetworkNew();
                            N3PredictionsMAX = N3PredictionsMAX + "N3 Predict: " + predicted + " ;";
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MAX_CL_predictionPercentFile_N3.ToString()); //Load saved network
                            PredictNetworkNew();
                            N3PredictionsMAX = N3PredictionsMAX + "N3 Predict: " + _closedLoop_predicted + " ;";
                        }

                        if (MAX_CL_predictionPercent_Abs1_N3 <= MAX_predictionPercent_Abs1_N3)
                        {
                            SaveLoadNetwork(false, Config.MAX_predictionPercent_Abs1File_N3.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = predicted;
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MAX_CL_predictionPercent_Abs1File_N3.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = _closedLoop_predicted;
                        }

                        N3PredictionsMAX = N3PredictionsMAX + "N3 Abs Predict: (";

                        flN1 = Math.Max(1.0, AbsNumber - 1.0);
                        flN2 = AbsNumber;
                        flN3 = AbsNumber + 1;

                        AbsList[0] = flN1; AbsList[1] = flN2; AbsList[2] = flN3;

                        foreach (var item in AbsList.Distinct())
                        {
                            N3PredictionsMAX = N3PredictionsMAX + item.ToString() + " ";
                        }

                        N3PredictionsMAX = N3PredictionsMAX + ")";

                        break;

                    case 4:
                        if (MAX_CL_predictionPercent_N4 <= MAX_predictionPercent_N4)
                        {
                            SaveLoadNetwork(false, Config.MAX_predictionPercentFile_N4.ToString()); //Load saved network
                            PredictNetworkNew();
                            N4PredictionsMAX = N4PredictionsMAX + "N4 Predict: " + predicted + " ;";
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MAX_CL_predictionPercentFile_N4.ToString()); //Load saved network
                            PredictNetworkNew();
                            N4PredictionsMAX = N4PredictionsMAX + "N4 Predict: " + _closedLoop_predicted + " ;";
                        }

                        if (MAX_CL_predictionPercent_Abs1_N4 <= MAX_predictionPercent_Abs1_N4)
                        {
                            SaveLoadNetwork(false, Config.MAX_predictionPercent_Abs1File_N4.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = predicted;
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MAX_CL_predictionPercent_Abs1File_N4.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = _closedLoop_predicted;
                        }

                        N4PredictionsMAX = N4PredictionsMAX + "N4 Abs Predict: (";

                        flN1 = Math.Max(1.0, AbsNumber - 1.0);
                        flN2 = AbsNumber;
                        flN3 = AbsNumber + 1;

                        AbsList[0] = flN1; AbsList[1] = flN2; AbsList[2] = flN3;

                        foreach (var item in AbsList.Distinct())
                        {
                            N4PredictionsMAX = N4PredictionsMAX + item.ToString() + " ";
                        }

                        N4PredictionsMAX = N4PredictionsMAX + ")";

                        break;

                    case 5:
                        if (MAX_CL_predictionPercent_N5 <= MAX_predictionPercent_N5)
                        {
                            SaveLoadNetwork(false, Config.MAX_predictionPercentFile_N5.ToString()); //Load saved network
                            PredictNetworkNew();
                            N5PredictionsMAX = N5PredictionsMAX + "N5 Predict: " + predicted + " ;";
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MAX_CL_predictionPercentFile_N5.ToString()); //Load saved network
                            PredictNetworkNew();
                            N5PredictionsMAX = N5PredictionsMAX + "N5 Predict: " + _closedLoop_predicted + " ;";
                        }

                        if (MAX_CL_predictionPercent_Abs1_N5 <= MAX_predictionPercent_Abs1_N5)
                        {
                            SaveLoadNetwork(false, Config.MAX_predictionPercent_Abs1File_N5.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = predicted;
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MAX_CL_predictionPercent_Abs1File_N5.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = _closedLoop_predicted;
                        }

                        N5PredictionsMAX = N5PredictionsMAX + "N5 Abs Predict: (";

                        flN1 = Math.Max(1.0, AbsNumber - 1.0);
                        flN2 = AbsNumber;
                        flN3 = AbsNumber + 1;

                        AbsList[0] = flN1; AbsList[1] = flN2; AbsList[2] = flN3;

                        foreach (var item in AbsList.Distinct())
                        {
                            N5PredictionsMAX = N5PredictionsMAX + item.ToString() + " ";
                        }

                        N5PredictionsMAX = N5PredictionsMAX + ")";

                        break;

                    case 6:
                        if (MAX_CL_predictionPercent_N6 <= MAX_predictionPercent_N6)
                        {
                            SaveLoadNetwork(false, Config.MAX_predictionPercentFile_N6.ToString()); //Load saved network
                            PredictNetworkNew();
                            N6PredictionsMAX = N6PredictionsMAX + "N6 Predict: " + predicted + " ;";
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MAX_CL_predictionPercentFile_N6.ToString()); //Load saved network
                            PredictNetworkNew();
                            N6PredictionsMAX = N6PredictionsMAX + "N6 Predict: " + _closedLoop_predicted + " ;";
                        }

                        if (MAX_CL_predictionPercent_Abs1_N6 <= MAX_predictionPercent_Abs1_N6)
                        {
                            SaveLoadNetwork(false, Config.MAX_predictionPercent_Abs1File_N6.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = predicted;
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MAX_CL_predictionPercent_Abs1File_N6.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = _closedLoop_predicted;
                        }

                        N6PredictionsMAX = N6PredictionsMAX + "N6 Abs Predict: (";

                        flN1 = Math.Max(1.0, AbsNumber - 1.0);
                        flN2 = AbsNumber;
                        flN3 = AbsNumber + 1;

                        AbsList[0] = flN1; AbsList[1] = flN2; AbsList[2] = flN3;

                        foreach (var item in AbsList.Distinct())
                        {
                            N6PredictionsMAX = N6PredictionsMAX + item.ToString() + " ";
                        }

                        N6PredictionsMAX = N6PredictionsMAX + ")";

                        break;

                    case 7:
                        if (MAX_CL_predictionPercent_N7 <= MAX_predictionPercent_N7)
                        {
                            SaveLoadNetwork(false, Config.MAX_predictionPercentFile_N7.ToString()); //Load saved network
                            PredictNetworkNew();
                            N7PredictionsMAX = N7PredictionsMAX + "N7 Predict: " + predicted + " ;";
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MAX_CL_predictionPercentFile_N7.ToString()); //Load saved network
                            PredictNetworkNew();
                            N7PredictionsMAX = N7PredictionsMAX + "N7 Predict: " + _closedLoop_predicted + " ;";
                        }

                        if (MAX_CL_predictionPercent_Abs1_N7 <= MAX_predictionPercent_Abs1_N7)
                        {
                            SaveLoadNetwork(false, Config.MAX_predictionPercent_Abs1File_N7.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = predicted;
                        }
                        else
                        {
                            SaveLoadNetwork(false, Config.MAX_CL_predictionPercent_Abs1File_N7.ToString()); //Load saved network
                            PredictNetworkNew();
                            AbsNumber = _closedLoop_predicted;
                        }

                        N7PredictionsMAX = N7PredictionsMAX + "N7 Abs Predict: (";

                        flN1 = Math.Max(1.0, AbsNumber - 1.0);
                        flN2 = AbsNumber;
                        flN3 = AbsNumber + 1;

                        AbsList[0] = flN1; AbsList[1] = flN2; AbsList[2] = flN3;

                        foreach (var item in AbsList.Distinct())
                        {
                            N7PredictionsMAX = N7PredictionsMAX + item.ToString() + " ";
                        }

                        N7PredictionsMAX = N7PredictionsMAX + ")";

                        break;

                    default:
                        break;
                }
            }


            //MAX Reporting

            Console.WriteLine("-- Predictions MAX ---\n");
            Console.WriteLine("-- N1PredictionsMAX --- " + N1PredictionsMAX + "\n");
            Console.WriteLine("-- N2PredictionsMAX --- " + N2PredictionsMAX + "\n");
            Console.WriteLine("-- N3PredictionsMAX --- " + N3PredictionsMAX + "\n");
            Console.WriteLine("-- N4PredictionsMAX --- " + N4PredictionsMAX + "\n");
            Console.WriteLine("-- N5PredictionsMAX --- " + N5PredictionsMAX + "\n");
            Console.WriteLine("-- N6PredictionsMAX --- " + N6PredictionsMAX + "\n");
            Console.WriteLine("-- N7PredictionsMAX --- " + N7PredictionsMAX + "\n");

            Console.WriteLine("-- Predictions MIN ---\n");
            Console.WriteLine("-- N1PredictionsMIN --- " + N1PredictionsMIN + "\n");
            Console.WriteLine("-- N2PredictionsMIN --- " + N2PredictionsMIN + "\n");
            Console.WriteLine("-- N3PredictionsMIN --- " + N3PredictionsMIN + "\n");
            Console.WriteLine("-- N4PredictionsMIN --- " + N4PredictionsMIN + "\n");
            Console.WriteLine("-- N5PredictionsMIN --- " + N5PredictionsMIN + "\n");
            Console.WriteLine("-- N6PredictionsMIN --- " + N6PredictionsMIN + "\n");
            Console.WriteLine("-- N7PredictionsMIN --- " + N7PredictionsMIN + "\n");


        }


        public void Predict()
        {
            //Read Data
            ReadData();

            //Normalization
            NormalizeData();

            ////Generate Training dataset
            GenerateTemporalData();

            ////Create & Train Network
            if (! CreateAndTrainNetwork(500)) {
                Console.WriteLine("-- CreateAndTrainNetwork step timeout, exiting --");
                Environment.Exit(0);
            };

            //errorDiagnostic(network, trainingSet, blnShowConsole);

            ////Evaluate Network
            EvaluateNetwork(blnShowConsole);


            ////Save the network to MAX_predictionPercentFile
            //SaveLoadNetwork(true, Config.MAX_predictionPercentFile.ToString());

            ////Predict next value
            //PredictNetwork();

            PredictNetworkNew();


            ////Load the network from MAX_predictionPercentFile
            //SaveLoadNetwork(false, Config.MAX_predictionPercentFile.ToString());

            ////Predict next value
            //PredictNetworkNew();

        }



        

        private void ReadData()
        {
            var csvreader = new Encog.Util.CSV.ReadCSV(Config.BaseArchieveFile.ToString(), true,
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

            //TotalNumOfIterations = data.Select(t => t.Id).Max() - (data.Select(t => t.Id).Min() + PastWindowSize) + 1;
            //TrainStart = 1; // data.Select(t => t.Id).Min() + PastWindowSize;
            //TrainEnd = TrainStart + TotalNumOfIterations * trainPercent / 100;
            //EvaluateStart = TrainEnd + 1;
            //EvaluateEnd = data.Count;

            //TotalNumOfIterations = data.Select(t => t.Id).Max() - (data.Select(t => t.Id).Min() + PastWindowSize) + 1;
            //TrainStart = data.Select(t => t.Id).Min() + PastWindowSize;

            TrainEnd = data.Count * trainPercent / 100;
            EvaluateStart = TrainEnd + 1;
            EvaluateEnd = data.Count;

            //var queryActual1 =
            //        from LotoData in data
            //        where LotoData.Id >= 0 && LotoData.Id < (TrainEnd - TrainStart + 1)
            //        orderby LotoData.Id ascending
            //        select LotoData.Actual1;

            //foreach (int numberA in data.Select(t => t.Actual1).ToArray()) NA.NumbersDic37_Total[numberA]++;
            //foreach (int numberA in data.Select(t => t.Actual2).ToArray()) NA.NumbersDic37_Total[numberA]++;
            //foreach (int numberA in data.Select(t => t.Actual3).ToArray()) NA.NumbersDic37_Total[numberA]++;
            //foreach (int numberA in data.Select(t => t.Actual4).ToArray()) NA.NumbersDic37_Total[numberA]++;
            //foreach (int numberA in data.Select(t => t.Actual5).ToArray()) NA.NumbersDic37_Total[numberA]++;
            //foreach (int numberA in data.Select(t => t.Actual6).ToArray()) NA.NumbersDic37_Total[numberA]++;
            //foreach (int numberA in data.Select(t => t.Actual7).ToArray()) NA.NumbersDic7_Total[numberA]++;

            // Calculate NA.NumbersDic37_PastWindow and NA.NumbersDic7_PastWindow values.


            int CurrentNumber1, CurrentNumber2, CurrentNumber3, CurrentNumber4, CurrentNumber5, CurrentNumber6, CurrentNumber7;
            int maxWindow;


            NA.NumbersDic37_PastWindow[1] = MinPastWindowSize;
            NA.NumbersDic37_PastWindow[2] = MinPastWindowSize;
            NA.NumbersDic37_PastWindow[3] = MinPastWindowSize;
            NA.NumbersDic37_PastWindow[4] = MinPastWindowSize;
            NA.NumbersDic37_PastWindow[5] = MinPastWindowSize;
            NA.NumbersDic37_PastWindow[6] = MinPastWindowSize;
            NA.NumbersDic7_PastWindow[1] = MinPastWindowSize;

            for (int i = TrainEnd; i >= TrainEnd / 90; i--)
            {
                CurrentNumber1 = (int)data[i].Actual1;
                maxWindow = 0;

                // Search for max window needed within then actual numbers for CurrentNumber
                for (int j = i - 1; j >= 0; j--)
                {
                    maxWindow++;

                    //Found a matching number after maxWindow draws.
                    if (((int)data[j].Actual1 == CurrentNumber1)||
                        ((int)data[j].Actual2 == CurrentNumber1)||
                        ((int)data[j].Actual3 == CurrentNumber1)||
                        ((int)data[j].Actual4 == CurrentNumber1)||
                        ((int)data[j].Actual5 == CurrentNumber1)||
                        ((int)data[j].Actual6 == CurrentNumber1))
                    {
                        NA.NumbersDic37_PastWindow[1] = Math.Max(NA.NumbersDic37_PastWindow[1], maxWindow);
                        NA.NumbersDic37_PastWindow[1] = Math.Min(NA.NumbersDic37_PastWindow[1], MaxPastWindowSize);
                        break;
                    }
                }

                NA.NumbersDic37_PastWindow[1] = Math.Max(NA.NumbersDic37_PastWindow[1], MinPastWindowSize);

                CurrentNumber2 = (int)data[i].Actual2;
                maxWindow = 0;

                // Search for max window needed within then actual numbers for CurrentNumber
                for (int j = i - 1; j >= 0; j--)
                {
                    maxWindow++;

                    if (((int)data[j].Actual1 == CurrentNumber2) ||
                        ((int)data[j].Actual2 == CurrentNumber2) ||
                        ((int)data[j].Actual3 == CurrentNumber2) ||
                        ((int)data[j].Actual4 == CurrentNumber2) ||
                        ((int)data[j].Actual5 == CurrentNumber2) ||
                        ((int)data[j].Actual6 == CurrentNumber2))
                    {
                        NA.NumbersDic37_PastWindow[2] = Math.Max(NA.NumbersDic37_PastWindow[2], maxWindow);
                        NA.NumbersDic37_PastWindow[2] = Math.Min(NA.NumbersDic37_PastWindow[2], MaxPastWindowSize);
                        break;
                    }
                }

                NA.NumbersDic37_PastWindow[2] = Math.Max(NA.NumbersDic37_PastWindow[2], MinPastWindowSize);

                CurrentNumber3 = (int)data[i].Actual3;
                maxWindow = 0;

                // Search for max window needed within then actual numbers for CurrentNumber
                for (int j = i - 1; j >= 0; j--)
                {
                    maxWindow++;

                    if (((int)data[j].Actual1 == CurrentNumber3) ||
                        ((int)data[j].Actual2 == CurrentNumber3) ||
                        ((int)data[j].Actual3 == CurrentNumber3) ||
                        ((int)data[j].Actual4 == CurrentNumber3) ||
                        ((int)data[j].Actual5 == CurrentNumber3) ||
                        ((int)data[j].Actual6 == CurrentNumber3))
                    {
                        NA.NumbersDic37_PastWindow[3] = Math.Max(NA.NumbersDic37_PastWindow[3], maxWindow);
                        NA.NumbersDic37_PastWindow[3] = Math.Min(NA.NumbersDic37_PastWindow[3], MaxPastWindowSize);

                        break;
                    }
                }

                NA.NumbersDic37_PastWindow[3] = Math.Max(NA.NumbersDic37_PastWindow[3], MinPastWindowSize);

                CurrentNumber4 = (int)data[i].Actual4;
                maxWindow = 0;

                // Search for max window needed within then actual numbers for CurrentNumber
                for (int j = i - 1; j >= 0; j--)
                {
                    maxWindow++;

                    if (((int)data[j].Actual1 == CurrentNumber4) ||
                        ((int)data[j].Actual2 == CurrentNumber4) ||
                        ((int)data[j].Actual3 == CurrentNumber4) ||
                        ((int)data[j].Actual4 == CurrentNumber4) ||
                        ((int)data[j].Actual5 == CurrentNumber4) ||
                        ((int)data[j].Actual6 == CurrentNumber4))
                    {
                        NA.NumbersDic37_PastWindow[4] = Math.Max(NA.NumbersDic37_PastWindow[4], maxWindow);
                        NA.NumbersDic37_PastWindow[4] = Math.Min(NA.NumbersDic37_PastWindow[4], MaxPastWindowSize);
                        break;
                    }
                }

                NA.NumbersDic37_PastWindow[4] = Math.Max(NA.NumbersDic37_PastWindow[4], MinPastWindowSize);

                CurrentNumber5 = (int)data[i].Actual5;
                maxWindow = 0;

                // Search for max window needed within then actual numbers for CurrentNumber
                for (int j = i - 1; j >= 0; j--)
                {
                    maxWindow++;

                    if (((int)data[j].Actual1 == CurrentNumber5) ||
                        ((int)data[j].Actual2 == CurrentNumber5) ||
                        ((int)data[j].Actual3 == CurrentNumber5) ||
                        ((int)data[j].Actual4 == CurrentNumber5) ||
                        ((int)data[j].Actual5 == CurrentNumber5) ||
                        ((int)data[j].Actual6 == CurrentNumber5))
                    {
                        NA.NumbersDic37_PastWindow[5] = Math.Max(NA.NumbersDic37_PastWindow[5], maxWindow);
                        NA.NumbersDic37_PastWindow[5] = Math.Min(NA.NumbersDic37_PastWindow[5], MaxPastWindowSize);

                        break;
                    }
                }

                NA.NumbersDic37_PastWindow[5] = Math.Max(NA.NumbersDic37_PastWindow[5], MinPastWindowSize);

                CurrentNumber6 = (int)data[i].Actual6;
                maxWindow = 0;

                // Search for max window needed within then actual numbers for CurrentNumber
                for (int j = i - 1; j >= 0; j--)
                {
                    maxWindow++;

                    if (((int)data[j].Actual1 == CurrentNumber6) ||
                        ((int)data[j].Actual2 == CurrentNumber6) ||
                        ((int)data[j].Actual3 == CurrentNumber6) ||
                        ((int)data[j].Actual4 == CurrentNumber6) ||
                        ((int)data[j].Actual5 == CurrentNumber6) ||
                        ((int)data[j].Actual6 == CurrentNumber6))
                    {
                        NA.NumbersDic37_PastWindow[6] = Math.Max(NA.NumbersDic37_PastWindow[6], maxWindow);
                        NA.NumbersDic37_PastWindow[6] = Math.Min(NA.NumbersDic37_PastWindow[6], MaxPastWindowSize);

                        break;
                    }
                }

                NA.NumbersDic37_PastWindow[6] = Math.Max(NA.NumbersDic37_PastWindow[6], MinPastWindowSize);

                CurrentNumber7 = (int)data[i].Actual7;
                maxWindow = 0;

                // Search for max window needed within then actual numbers for CurrentNumber
                for (int j = i - 1; j >= 0; j--)
                {
                    maxWindow++;
                    if ((int)data[j].Actual7 == CurrentNumber7)  //Found a matching number after maxWindow draws.
                    {
                        NA.NumbersDic7_PastWindow[1] = Math.Max(NA.NumbersDic7_PastWindow[1], maxWindow);
                        break;
                    }
                }

                //NA.NumbersDic7_PastWindow[1] = Math.Max(NA.NumbersDic7_PastWindow[1], MinPastWindowSize);
            }


            if (PastWindowSize == 0)    //If we need to get PastWindowSize from the "PastWindow" calculation above then ..
            {
                if (LotoNumber >= 1 && LotoNumber <= 6)
                    PastWindowSize = NA.NumbersDic37_PastWindow[LotoNumber];
                else
                    PastWindowSize = NA.NumbersDic7_PastWindow[1];
            }


            TotalNumOfIterations = data.Select(t => t.Id).Max() - (data.Select(t => t.Id).Min() + PastWindowSize) + 1;
            TrainStart = data.Select(t => t.Id).Min() + PastWindowSize;


            for (int i = TrainStart; i < EvaluateEnd; i++)
            {
                NA.NumbersDic37_Total[(int)data[i].Actual1] = 0;
                NA.NumbersDic37_Total[(int)data[i].Actual2] = 0;
                NA.NumbersDic37_Total[(int)data[i].Actual3] = 0;
                NA.NumbersDic37_Total[(int)data[i].Actual4] = 0;
                NA.NumbersDic37_Total[(int)data[i].Actual5] = 0;
                NA.NumbersDic37_Total[(int)data[i].Actual6] = 0;
                NA.NumbersDic7_Total[(int)data[i].Actual7] = 0;

                for (int j = i - PastWindowSize; j < i; j++)
                {
                    NA.NumbersDic37_Total[(int)data[j].Actual1]++;
                    NA.NumbersDic37_Total[(int)data[j].Actual2]++;
                    NA.NumbersDic37_Total[(int)data[j].Actual3]++;
                    NA.NumbersDic37_Total[(int)data[j].Actual4]++;
                    NA.NumbersDic37_Total[(int)data[j].Actual5]++;
                    NA.NumbersDic37_Total[(int)data[j].Actual6]++;
                    NA.NumbersDic7_Total[(int)data[j].Actual7]++;
                }

                data[i].NA_Actual1 = NA.NumbersDic37_Total[(int)data[i].Actual1];
                data[i].NA_Actual2 = NA.NumbersDic37_Total[(int)data[i].Actual2];
                data[i].NA_Actual3 = NA.NumbersDic37_Total[(int)data[i].Actual3];
                data[i].NA_Actual4 = NA.NumbersDic37_Total[(int)data[i].Actual4];
                data[i].NA_Actual5 = NA.NumbersDic37_Total[(int)data[i].Actual5];
                data[i].NA_Actual6 = NA.NumbersDic37_Total[(int)data[i].Actual6];
                data[i].NA_Actual7 = NA.NumbersDic7_Total[(int)data[i].Actual7];

                data[i]._closedLoopNA_Actual1 = NA.NumbersDic37_Total[(int)data[i].Actual1];
                data[i]._closedLoopNA_Actual2 = NA.NumbersDic37_Total[(int)data[i].Actual2];
                data[i]._closedLoopNA_Actual3 = NA.NumbersDic37_Total[(int)data[i].Actual3];
                data[i]._closedLoopNA_Actual4 = NA.NumbersDic37_Total[(int)data[i].Actual4];
                data[i]._closedLoopNA_Actual5 = NA.NumbersDic37_Total[(int)data[i].Actual5];
                data[i]._closedLoopNA_Actual6 = NA.NumbersDic37_Total[(int)data[i].Actual6];
                data[i]._closedLoopNA_Actual7 = NA.NumbersDic7_Total[(int)data[i].Actual7];
        }


        //Assign NA_Actual values for training set

        //for (int i = 0; i < TrainEnd; i++)
        //{
        //    NA.NumbersDic37_Total[(int)data[i].Actual1]++;
        //    NA.NumbersDic37_Total[(int)data[i].Actual2]++;
        //    NA.NumbersDic37_Total[(int)data[i].Actual3]++;
        //    NA.NumbersDic37_Total[(int)data[i].Actual4]++;
        //    NA.NumbersDic37_Total[(int)data[i].Actual5]++;
        //    NA.NumbersDic37_Total[(int)data[i].Actual6]++;
        //    NA.NumbersDic7_Total[(int)data[i].Actual7]++;
        //}

        //for (int i = 0; i < TrainEnd; i++)
        //{
        //    NA.closedLoopNumbersDic37_Total[(int)data[i].Actual1] = NA.NumbersDic37_Total[(int)data[i].Actual1];
        //    NA.closedLoopNumbersDic37_Total[(int)data[i].Actual2] = NA.NumbersDic37_Total[(int)data[i].Actual2];
        //    NA.closedLoopNumbersDic37_Total[(int)data[i].Actual3] = NA.NumbersDic37_Total[(int)data[i].Actual3];
        //    NA.closedLoopNumbersDic37_Total[(int)data[i].Actual4] = NA.NumbersDic37_Total[(int)data[i].Actual4];
        //    NA.closedLoopNumbersDic37_Total[(int)data[i].Actual5] = NA.NumbersDic37_Total[(int)data[i].Actual5];
        //    NA.closedLoopNumbersDic37_Total[(int)data[i].Actual6] = NA.NumbersDic37_Total[(int)data[i].Actual6];
        //    NA.closedLoopNumbersDic7_Total[(int)data[i].Actual7] = NA.NumbersDic7_Total[(int)data[i].Actual7];

        //    data[i].NA_Actual1 = NA.NumbersDic37_Total[(int)data[i].Actual1];
        //    data[i].NA_Actual2 = NA.NumbersDic37_Total[(int)data[i].Actual2];
        //    data[i].NA_Actual3 = NA.NumbersDic37_Total[(int)data[i].Actual3];
        //    data[i].NA_Actual4 = NA.NumbersDic37_Total[(int)data[i].Actual4];
        //    data[i].NA_Actual5 = NA.NumbersDic37_Total[(int)data[i].Actual5];
        //    data[i].NA_Actual6 = NA.NumbersDic37_Total[(int)data[i].Actual6];
        //    data[i].NA_Actual7 = NA.NumbersDic7_Total[(int)data[i].Actual7];

        //    data[i]._closedLoopNA_Actual1 = NA.closedLoopNumbersDic37_Total[(int)data[i].Actual1];
        //    data[i]._closedLoopNA_Actual2 = NA.closedLoopNumbersDic37_Total[(int)data[i].Actual2];
        //    data[i]._closedLoopNA_Actual3 = NA.closedLoopNumbersDic37_Total[(int)data[i].Actual3];
        //    data[i]._closedLoopNA_Actual4 = NA.closedLoopNumbersDic37_Total[(int)data[i].Actual4];
        //    data[i]._closedLoopNA_Actual5 = NA.closedLoopNumbersDic37_Total[(int)data[i].Actual5];
        //    data[i]._closedLoopNA_Actual6 = NA.closedLoopNumbersDic37_Total[(int)data[i].Actual6];
        //    data[i]._closedLoopNA_Actual7 = NA.closedLoopNumbersDic7_Total[(int)data[i].Actual7];
        //}


        ////Assign NA_Actual values for evaluation set
        //for (int i = EvaluateStart - 1; i < EvaluateEnd; i++)
        //{

        //    NA.NumbersDic37_Total[(int)data[i].Actual1] += 1;
        //    NA.NumbersDic37_Total[(int)data[i].Actual2] += 1;
        //    NA.NumbersDic37_Total[(int)data[i].Actual3] += 1;
        //    NA.NumbersDic37_Total[(int)data[i].Actual4] += 1;
        //    NA.NumbersDic37_Total[(int)data[i].Actual5] += 1;
        //    NA.NumbersDic37_Total[(int)data[i].Actual6] += 1;
        //    NA.NumbersDic7_Total[(int)data[i].Actual7] += 1;
        //}

        //for (int i = EvaluateStart-1; i < EvaluateEnd; i++)
        //{

        //    data[i].NA_Actual1 = NA.NumbersDic37_Total[(int)data[i].Actual1];
        //    data[i].NA_Actual2 = NA.NumbersDic37_Total[(int)data[i].Actual2];
        //    data[i].NA_Actual3 = NA.NumbersDic37_Total[(int)data[i].Actual3];
        //    data[i].NA_Actual4 = NA.NumbersDic37_Total[(int)data[i].Actual4];
        //    data[i].NA_Actual5 = NA.NumbersDic37_Total[(int)data[i].Actual5];
        //    data[i].NA_Actual6 = NA.NumbersDic37_Total[(int)data[i].Actual6];
        //    data[i].NA_Actual7 = NA.NumbersDic7_Total[(int)data[i].Actual7];
        //}





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
            norm7 = new NormalizeArray() { NormalizedHigh = NormalizeHigh, NormalizedLow = NormalizeLow };
            norm8 = new NormalizeArray() { NormalizedHigh = NormalizeHigh, NormalizedLow = NormalizeLow };
            norm9 = new NormalizeArray() { NormalizedHigh = NormalizeHigh, NormalizedLow = NormalizeLow };
            norm10 = new NormalizeArray() { NormalizedHigh = NormalizeHigh, NormalizedLow = NormalizeLow };
            norm11 = new NormalizeArray() { NormalizedHigh = NormalizeHigh, NormalizedLow = NormalizeLow };
            norm12 = new NormalizeArray() { NormalizedHigh = NormalizeHigh, NormalizedLow = NormalizeLow };
            norm13 = new NormalizeArray() { NormalizedHigh = NormalizeHigh, NormalizedLow = NormalizeLow };
            norm14 = new NormalizeArray() { NormalizedHigh = NormalizeHigh, NormalizedLow = NormalizeLow };

                       

            //NormalizeDayOfWeek
            normalizedArray = norm0.Process((data.Select(t => (double)t.DayOfWeek).ToArray()));
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizeDayOfWeek = normalizedArray[i];
            }

            
            normalizedArray = norm1.Process(data.Select(t => t.Actual1).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedActual1 = normalizedArray[i];
            }

            var _closedLoopNormalizedArray = Encog.Util.EngineArray.ArrayCopy(normalizedArray);
            for (int i = 0; i < _closedLoopNormalizedArray.Count(); i++)
            {
                data[i]._closedLoopNormalizedActual1 = _closedLoopNormalizedArray[i];
            }

            normalizedArray = norm2.Process(data.Select(t => t.Actual2).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedActual2 = normalizedArray[i];
            }

            _closedLoopNormalizedArray = Encog.Util.EngineArray.ArrayCopy(normalizedArray);
            for (int i = 0; i < _closedLoopNormalizedArray.Count(); i++)
            {
                data[i]._closedLoopNormalizedActual2 = _closedLoopNormalizedArray[i];
            }

            normalizedArray = norm3.Process(data.Select(t => t.Actual3).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedActual3 = normalizedArray[i];
            }

            _closedLoopNormalizedArray = Encog.Util.EngineArray.ArrayCopy(normalizedArray);
            for (int i = 0; i < _closedLoopNormalizedArray.Count(); i++)
            {
                data[i]._closedLoopNormalizedActual3 = _closedLoopNormalizedArray[i];
            }


            normalizedArray = norm4.Process(data.Select(t => t.Actual4).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedActual4 = normalizedArray[i];
            }

            _closedLoopNormalizedArray = Encog.Util.EngineArray.ArrayCopy(normalizedArray);
            for (int i = 0; i < _closedLoopNormalizedArray.Count(); i++)
            {
                data[i]._closedLoopNormalizedActual4 = _closedLoopNormalizedArray[i];
            }


            normalizedArray = norm5.Process(data.Select(t => t.Actual5).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedActual5 = normalizedArray[i];
            }

            _closedLoopNormalizedArray = Encog.Util.EngineArray.ArrayCopy(normalizedArray);
            for (int i = 0; i < _closedLoopNormalizedArray.Count(); i++)
            {
                data[i]._closedLoopNormalizedActual5 = _closedLoopNormalizedArray[i];
            }


            normalizedArray = norm6.Process(data.Select(t => t.Actual6).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedActual6 = normalizedArray[i];
            }

            _closedLoopNormalizedArray = Encog.Util.EngineArray.ArrayCopy(normalizedArray);
            for (int i = 0; i < _closedLoopNormalizedArray.Count(); i++)
            {
                data[i]._closedLoopNormalizedActual6 = _closedLoopNormalizedArray[i];
            }


            normalizedArray = norm7.Process(data.Select(t => t.Actual7).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedActual7 = normalizedArray[i];
            }

            _closedLoopNormalizedArray = Encog.Util.EngineArray.ArrayCopy(normalizedArray);
            for (int i = 0; i < _closedLoopNormalizedArray.Count(); i++)
            {
                data[i]._closedLoopNormalizedActual7 = _closedLoopNormalizedArray[i];
            }


            /*
            data[i].NA_Actual1 = NA.NumbersDic37_Total[(int)data[i].Actual1];
            data[i].NA_Actual2 = NA.NumbersDic37_Total[(int)data[i].Actual2];
            data[i].NA_Actual3 = NA.NumbersDic37_Total[(int)data[i].Actual3];
            data[i].NA_Actual4 = NA.NumbersDic37_Total[(int)data[i].Actual4];
            data[i].NA_Actual5 = NA.NumbersDic37_Total[(int)data[i].Actual5];
            data[i].NA_Actual6 = NA.NumbersDic37_Total[(int)data[i].Actual6];
            data[i].NA_Actual7 = NA.NumbersDic7_Total[(int)data[i].Actual7];
            */

            normalizedArray = norm8.Process(data.Select(t => t.NA_Actual1).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedNA_Actual1 = normalizedArray[i];
            }


            normalizedArray = norm9.Process(data.Select(t => t.NA_Actual2).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedNA_Actual2 = normalizedArray[i];
            }

            normalizedArray = norm10.Process(data.Select(t => t.NA_Actual3).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedNA_Actual3 = normalizedArray[i];
            }

            normalizedArray = norm11.Process(data.Select(t => t.NA_Actual4).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedNA_Actual4 = normalizedArray[i];
            }

            normalizedArray = norm12.Process(data.Select(t => t.NA_Actual5).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedNA_Actual5 = normalizedArray[i];
            }

            normalizedArray = norm13.Process(data.Select(t => t.NA_Actual6).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedNA_Actual6 = normalizedArray[i];
            }

            normalizedArray = norm14.Process(data.Select(t => t.NA_Actual7).ToArray());
            for (int i = 0; i < normalizedArray.Count(); i++)
            {
                data[i].NormalizedNA_Actual7 = normalizedArray[i];
            }
        }


        private void GenerateTemporalDataWithDayOfWeek()
        {
            TemporalDataDescription desc1 = null, desc2 = null, desc3 = null, desc4 = null, desc5 = null, desc6 = null, desc7 = null;
            
            //Temporal dataset
            trainingSet = new TemporalMLDataSet(PastWindowSize, FutureWindowSize);

            //Description #0
            var desc0 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
            desc0.Index = 0;
            trainingSet.AddDescription(desc0);

            if (LotoNumber != 7) 
            {

                var desc8 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                desc8.Index = 7;

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

                    //case 7:
                    //    desc7 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                    //    desc7.Index = 7;

                    //    break;

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

                //if (desc7 == null)
                //{
                //    desc7 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                //    desc7.Index = 7;
                //}

                trainingSet.AddDescription(desc1);
                trainingSet.AddDescription(desc2);
                trainingSet.AddDescription(desc3);
                trainingSet.AddDescription(desc4);
                trainingSet.AddDescription(desc5);
                trainingSet.AddDescription(desc6);
                trainingSet.AddDescription(desc8);
            }
            else
            {

                var desc8 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                desc8.Index = 2;

                desc7 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                desc7.Index = 1;
                trainingSet.AddDescription(desc7);
                trainingSet.AddDescription(desc8);
            }

            ////Description #7
            //var desc7 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
            //desc7.Index = 7;
            //trainingSet.AddDescription(desc7);

            //Temporal point
            if (LotoNumber != 7)  
            {
                for (int i = 0; i < TrainEnd; i++)
                {
                    var point = new TemporalPoint(8); //1 values
                    point.Sequence = data[i].Id;
                    point.Data[0] = data[i].NormalizeDayOfWeek;
                    point.Data[1] = data[i].NormalizedActual1;
                    point.Data[2] = data[i].NormalizedActual2;
                    point.Data[3] = data[i].NormalizedActual3;
                    point.Data[4] = data[i].NormalizedActual4;
                    point.Data[5] = data[i].NormalizedActual5;
                    point.Data[6] = data[i].NormalizedActual6;

                    point.Data[7] = (LotoNumber == 1) ? data[i].NormalizedNA_Actual1 : (LotoNumber == 2) ? data[i].NormalizedNA_Actual2 : (LotoNumber == 3) ? data[i].NormalizedNA_Actual3 :
                                    (LotoNumber == 4) ? data[i].NormalizedNA_Actual4 : (LotoNumber == 5) ? data[i].NormalizedNA_Actual5 : data[i].NormalizedNA_Actual6;
                    
                    trainingSet.Points.Add(point);
                }
            }
            else
            {
                for (int i = 0; i < TrainEnd; i++)
                {
                    var point = new TemporalPoint(3); //1 values
                    point.Sequence = data[i].Id;
                    point.Data[0] = data[i].NormalizeDayOfWeek;
                    point.Data[1] = data[i].NormalizedActual7;
                    point.Data[2] = data[i].NormalizedNA_Actual7;
                    trainingSet.Points.Add(point);
                }
            }




            //if (LotoNumber != 7)
            //{
            //    foreach (var item in data)
            //    {
            //        var point = new TemporalPoint(7); //1 values
            //        point.Sequence = item.Id;
            //        point.Data[0] = item.NormalizeDayOfWeek;
            //        point.Data[1] = item.NormalizedActual1;
            //        point.Data[2] = item.NormalizedActual2;
            //        point.Data[3] = item.NormalizedActual3;
            //        point.Data[4] = item.NormalizedActual4;
            //        point.Data[5] = item.NormalizedActual5;
            //        point.Data[6] = item.NormalizedActual6;
            //        trainingSet.Points.Add(point);
            //    }
            //}
            //else {
            //    foreach (var item in data)
            //    {
            //        var point = new TemporalPoint(2); //1 values
            //        point.Sequence = item.Id;
            //        point.Data[0] = item.NormalizeDayOfWeek;
            //        point.Data[1] = item.NormalizedActual7;
            //        trainingSet.Points.Add(point);
            //    }
            //}

            trainingSet.Generate();
        }

        private void GenerateTemporalData()
        {
            TemporalDataDescription desc1 = null, desc2 = null, desc3 = null, desc4 = null, desc5 = null, desc6 = null, desc7 = null;
            
            //Temporal dataset
            trainingSet = new TemporalMLDataSet(PastWindowSize, FutureWindowSize);

            //Description #0
            //var desc0 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
            //desc0.Index = 0;
            //trainingSet.AddDescription(desc0);

            if (LotoNumber != 7) 
            {

                var desc8 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                desc8.Index = 6;

                switch (LotoNumber)
                {
                    case 1:
                        desc1 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                        desc1.Index = 0;

                        break;

                    case 2:
                        desc2 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                        desc2.Index = 1;

                        break;

                    case 3:
                        desc3 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                        desc3.Index = 2;

                        break;

                    case 4:
                        desc4 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                        desc4.Index = 3;

                        break;

                    case 5:
                        desc5 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                        desc5.Index = 4;

                        break;

                    case 6:
                        desc6 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                        desc6.Index = 5;

                        break;

                    //case 7:
                    //    desc7 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                    //    desc7.Index = 7;

                    //    break;

                    default:
                        break;
                }

                //Description #1
                if (desc1 == null)
                {
                    desc1 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                    desc1.Index = 0;
                }

                //Description #2
                if (desc2 == null)
                {
                    desc2 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                    desc2.Index = 1;
                }

                if (desc3 == null)
                {
                    desc3 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                    desc3.Index = 2;
                }

                if (desc4 == null)
                {
                    desc4 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                    desc4.Index = 3;
                }

                if (desc5 == null)
                {
                    desc5 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                    desc5.Index = 4;
                }

                if (desc6 == null)
                {
                    desc6 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                    desc6.Index = 5;
                }

               
                trainingSet.AddDescription(desc1);
                trainingSet.AddDescription(desc2);
                trainingSet.AddDescription(desc3);
                trainingSet.AddDescription(desc4);
                trainingSet.AddDescription(desc5);
                trainingSet.AddDescription(desc6);
                trainingSet.AddDescription(desc8);
            }
            else
            {

                var desc8 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, false);
                desc8.Index = 1;

                desc7 = new TemporalDataDescription(TemporalDataDescription.Type.Raw, true, true);
                desc7.Index = 0;
                trainingSet.AddDescription(desc7);
                trainingSet.AddDescription(desc8);
            }

            //Temporal point
            if (LotoNumber != 7)  
            {
                for (int i = 0; i < TrainEnd; i++)
                {
                    var point = new TemporalPoint(7); //1 values
                    point.Sequence = data[i].Id;
                    point.Data[0] = data[i].NormalizedActual1;
                    point.Data[1] = data[i].NormalizedActual2;
                    point.Data[2] = data[i].NormalizedActual3;
                    point.Data[3] = data[i].NormalizedActual4;
                    point.Data[4] = data[i].NormalizedActual5;
                    point.Data[5] = data[i].NormalizedActual6;

                    point.Data[6] = (LotoNumber == 1) ? data[i].NormalizedNA_Actual1 : (LotoNumber == 2) ? data[i].NormalizedNA_Actual2 : (LotoNumber == 3) ? data[i].NormalizedNA_Actual3 :
                                    (LotoNumber == 4) ? data[i].NormalizedNA_Actual4 : (LotoNumber == 5) ? data[i].NormalizedNA_Actual5 : data[i].NormalizedNA_Actual6;
                    
                    trainingSet.Points.Add(point);
                }
            }
            else
            {
                for (int i = 0; i < TrainEnd; i++)
                {
                    var point = new TemporalPoint(2); //1 values
                    point.Sequence = data[i].Id;
                    point.Data[0] = data[i].NormalizedActual7;
                    point.Data[1] = data[i].NormalizedNA_Actual7;
                    trainingSet.Points.Add(point);
                }
            }

            trainingSet.Generate();
        }


        private Boolean CreateAndTrainNetwork(double secTimeout)
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

            pattern.AddHiddenLayer((PastWindowSize + FutureWindowSize) * 1);
            network = (BasicNetwork)pattern.Generate();

            ITrain train = new ResilientPropagation(network, trainingSet);


            //EncogUtility.TrainToError(train, MaxError);



            var task = Task.Run(() => EncogUtility.TrainToError(train, MaxError));
            //if (task.Wait(TimeSpan.FromSeconds(secTimeout)))
            //{
            //    Console.WriteLine("-- End of CreateAndTrainNetwork step --");
            //    return true;
            //}
            //else
            //    return false;
            ////throw new Exception("Timed out");


            try
            {
                task.Wait(TimeSpan.FromSeconds(secTimeout));
            }
            catch (AggregateException ae)
            {
                Console.WriteLine("Had an inner exception while running the task for TrainToError. ({0})", ae.Message);
                return false;
            }
            finally
            {
                Console.WriteLine("-- End of CreateAndTrainNetwork step --");
            }

            return true;
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


        private void EvaluateNetworkWithDayOfWeek(Boolean blnShowConsole)
        {
            int countPredicted, countUnPredicted, CL_countPredicted, CL_countUnPredicted;
            Boolean blnPredicted, bln_closedLoop_Predicted;

            int countPredicted_Abs1, countUnPredicted_Abs1, CL_countPredicted_Abs1, CL_countUnPredicted_Abs1;
            Boolean blnPredicted_Abs1, bln_closedLoop_Predicted_Abs1;

            //float predictionPercent;

            //int evaluateStart = data.Select(t => t.Id).Min() + PastWindowSize;
            //int evaluateStop = data.Select(t => t.Id).Max();
            //int TotalNumOfIterations = evaluateStop - evaluateStart;

            IMLData output;
            IMLData _closedLoop_output;


            using (var file = new System.IO.StreamWriter(Config.EvaluationResult.ToString()))
            {
                countPredicted = countUnPredicted = 0;
                CL_countPredicted = CL_countUnPredicted = 0;
                countPredicted_Abs1 = countUnPredicted_Abs1 = 0;
                CL_countPredicted_Abs1 = CL_countUnPredicted_Abs1 = 0;

                for (int currentId = EvaluateStart; currentId < EvaluateEnd; currentId++)
                {

                    if (LotoNumber != 7)
                    {
                        //Calculate based on actual data
                        var input = new BasicMLData(PastWindowSize * 8);
                        var _closedLoop_input = new BasicMLData(PastWindowSize * 8);

                        for (int i = 0; i < PastWindowSize; i++)
                        {
                            input[i * 8] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                 .Select(t => t.NormalizeDayOfWeek).First();

                            input[i * 8 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizedActual1).First();

                            input[i * 8 + 2] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizedActual2).First();

                            input[i * 8 + 3] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizedActual3).First();

                            input[i * 8 + 4] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizedActual4).First();

                            input[i * 8 + 5] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizedActual5).First();

                            input[i * 8 + 6] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizedActual6).First();


                            input[i * 8 + 7] = (LotoNumber == 1) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual1).First() :
                                               (LotoNumber == 2) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual2).First() :
                                               (LotoNumber == 3) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual3).First() :
                                               (LotoNumber == 4) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual4).First() :
                                               (LotoNumber == 5) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual5).First() :
                                               data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual6).First();


                            _closedLoop_input[i * 8] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                             .Select(t => t.NormalizeDayOfWeek).First();

                            _closedLoop_input[i * 8 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t._closedLoopNormalizedActual1).First();

                            _closedLoop_input[i * 8 + 2] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t._closedLoopNormalizedActual2).First();

                            _closedLoop_input[i * 8 + 3] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t._closedLoopNormalizedActual3).First();

                            _closedLoop_input[i * 8 + 4] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t._closedLoopNormalizedActual4).First();

                            _closedLoop_input[i * 8 + 5] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t._closedLoopNormalizedActual5).First();

                            _closedLoop_input[i * 8 + 6] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t._closedLoopNormalizedActual6).First();

                            _closedLoop_input[i * 8 + 7] = input[i * 8 + 7];
                        }

                        output = network.Compute(input);
                        _closedLoop_output = network.Compute(_closedLoop_input);
                    }
                    else
                    {
                        //Calculate based on actual data
                        var input = new BasicMLData(PastWindowSize * 3);
                        var _closedLoop_input = new BasicMLData(PastWindowSize * 3);

                        for (int i = 0; i < PastWindowSize; i++)
                        {
                            input[i * 3] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                             .Select(t => t.NormalizeDayOfWeek).First();

                            input[i * 3 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizedActual7).First();

                            input[i * 3 + 2] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizedNA_Actual7).First();

                            _closedLoop_input[i * 3] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizeDayOfWeek).First();

                            _closedLoop_input[i * 3 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t._closedLoopNormalizedActual7).First();

                            _closedLoop_input[i * 3 + 2] = input[i * 3 + 2];
                        }

                        output = network.Compute(input);
                        _closedLoop_output = network.Compute(_closedLoop_input);
                    }

                    double normalizedPredicted = output[0];
                    double _closedLoop_normalizedPredicted = _closedLoop_output[0];

                    double predicted = 0.0;
                    double _closedLoop_predicted = 0.0;

                    switch (LotoNumber)
                    {
                        case 1:
                            data[currentId]._closedLoopNormalizedActual1 = _closedLoop_normalizedPredicted;
                            predicted = Math.Round(norm1.Stats.DeNormalize(normalizedPredicted), 0);
                            _closedLoop_predicted = Math.Round(norm1.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                            break;

                        case 2:
                            data[currentId]._closedLoopNormalizedActual2 = _closedLoop_normalizedPredicted;
                            predicted = Math.Round(norm2.Stats.DeNormalize(normalizedPredicted), 0);
                            _closedLoop_predicted = Math.Round(norm2.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                            break;

                        case 3:
                            data[currentId]._closedLoopNormalizedActual3 = _closedLoop_normalizedPredicted;
                            predicted = Math.Round(norm3.Stats.DeNormalize(normalizedPredicted), 0);
                            _closedLoop_predicted = Math.Round(norm3.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                            break;

                        case 4:
                            data[currentId]._closedLoopNormalizedActual4 = _closedLoop_normalizedPredicted;
                            predicted = Math.Round(norm4.Stats.DeNormalize(normalizedPredicted), 0);
                            _closedLoop_predicted = Math.Round(norm4.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                            break;

                        case 5:
                            data[currentId]._closedLoopNormalizedActual5 = _closedLoop_normalizedPredicted;
                            predicted = Math.Round(norm5.Stats.DeNormalize(normalizedPredicted), 0);
                            _closedLoop_predicted = Math.Round(norm5.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                            break;

                        case 6:
                            data[currentId]._closedLoopNormalizedActual6 = _closedLoop_normalizedPredicted;
                            predicted = Math.Round(norm6.Stats.DeNormalize(normalizedPredicted), 0);
                            _closedLoop_predicted = Math.Round(norm6.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                            break;

                        case 7:
                            data[currentId]._closedLoopNormalizedActual7 = _closedLoop_normalizedPredicted;
                            predicted = Math.Round(norm7.Stats.DeNormalize(normalizedPredicted), 0);
                            _closedLoop_predicted = Math.Round(norm7.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                            break;

                        default:
                            break;
                    }

                    //double Actual1 = data.Where(t => t.Id == currentId).Select(t => t.Actual1).First();
                    double actual1 = 0.0;
                    double actual2 = 0.0;
                    double actual3 = 0.0;
                    double actual4 = 0.0;
                    double actual5 = 0.0;
                    double actual6 = 0.0;
                    double actual7 = 0.0;

                    if (LotoNumber != 7)
                    {
                        actual1 = Math.Round(norm1.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual1).First()), 0);
                        actual2 = Math.Round(norm2.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual2).First()), 0);
                        actual3 = Math.Round(norm3.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual3).First()), 0);
                        actual4 = Math.Round(norm4.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual4).First()), 0);
                        actual5 = Math.Round(norm5.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual5).First()), 0);
                        actual6 = Math.Round(norm6.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual6).First()), 0);
                    }
                    else
                        actual7 = Math.Round(norm7.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual7).First()), 0);

                    int DrawNumber = data.Where(t => t.Id == currentId).Select(t => t.DrawNumber).First();
                    if (LotoNumber != 7) {
                        blnPredicted = (actual1 == predicted || 
                                        actual2 == predicted || 
                                        actual3 == predicted || 
                                        actual4 == predicted || 
                                        actual5 == predicted || 
                                        actual6 == predicted) ? true : false;

                        bln_closedLoop_Predicted = (actual1 == _closedLoop_predicted ||
                                                    actual2 == _closedLoop_predicted ||
                                                    actual3 == _closedLoop_predicted ||
                                                    actual4 == _closedLoop_predicted ||
                                                    actual5 == _closedLoop_predicted ||
                                                    actual6 == _closedLoop_predicted) ? true : false;

                        blnPredicted_Abs1 = (Math.Abs(actual1-predicted) <= 1 ||
                                             Math.Abs(actual2-predicted) <= 1 ||
                                             Math.Abs(actual3-predicted) <= 1 ||
                                             Math.Abs(actual4-predicted) <= 1 ||
                                             Math.Abs(actual5-predicted) <= 1 ||
                                             Math.Abs(actual6-predicted) <= 1) ? true : false;

                        bln_closedLoop_Predicted_Abs1 = (Math.Abs(actual1 - _closedLoop_predicted) <= 1 ||
                                                         Math.Abs(actual2 - _closedLoop_predicted) <= 1 ||
                                                         Math.Abs(actual3 - _closedLoop_predicted) <= 1 ||
                                                         Math.Abs(actual4 - _closedLoop_predicted) <= 1 ||
                                                         Math.Abs(actual5 - _closedLoop_predicted) <= 1 ||
                                                         Math.Abs(actual6 - _closedLoop_predicted) <= 1) ? true : false;
                    }
                    else
                    {
                        blnPredicted = (actual7 == predicted) ? true : false;
                        bln_closedLoop_Predicted = (actual7 == _closedLoop_predicted) ? true : false;

                        blnPredicted_Abs1 = (Math.Abs(actual7-predicted) <= 1) ? true : false;
                        bln_closedLoop_Predicted_Abs1 = (Math.Abs(actual7 - _closedLoop_predicted) <= 1) ? true : false;
                    }


                    if (blnPredicted)
                        countPredicted++;
                    else
                        countUnPredicted++;

                    if (bln_closedLoop_Predicted)
                        CL_countPredicted++;
                    else
                        CL_countUnPredicted++;


                    if (blnPredicted_Abs1)
                        countPredicted_Abs1++;
                    else
                        countUnPredicted_Abs1++;

                    if (bln_closedLoop_Predicted_Abs1)
                        CL_countPredicted_Abs1++;
                    else
                        CL_countUnPredicted_Abs1++;


                    string line1 = "";
                    if (LotoNumber != 7)
                    {
                        line1 = string.Format("DrawNumber: {0}; Actual: ({1},{2},{3},{4},{5},{6}); Predicted: {7}; closedLoop_Predicted: {8}", DrawNumber, actual1, actual2, actual3, actual4, actual5, actual6, predicted, _closedLoop_predicted);
                    }
                    else
                    {
                        line1 = string.Format("DrawNumber: {0}; Actual: {1}; Predicted: {2}; closedLoop_Predicted: {3}", DrawNumber, actual7, predicted, _closedLoop_predicted);
                    }

                    file.WriteLine(line1);
                    if (blnShowConsole)
                        Console.WriteLine(line1);

                }

                predictionPercent = ((float)countPredicted / ((float)countPredicted + (float)countUnPredicted)) * 100;
                CL_predictionPercent = ((float)CL_countPredicted / ((float)CL_countPredicted + (float)CL_countUnPredicted)) * 100;

                predictionPercent_Abs1 = ((float)countPredicted_Abs1 / ((float)countPredicted_Abs1 + (float)countUnPredicted_Abs1)) * 100; 
                CL_predictionPercent_Abs1 = ((float)CL_countPredicted_Abs1 / ((float)CL_countPredicted_Abs1 + (float)CL_countUnPredicted_Abs1)) * 100;


                string line2 = string.Format(@"TotalPredictions = {0}, Predicted = {1}, UnPredicted = {2}, Prediction percent = {3:0.00}% \n 
                                               _closedLoop_TotalPredictions = {4}, _closedLoop_Predicted = {5}, _closedLoop_UnPredicted = {6}, _closedLoop_Prediction percent = {7:0.00}%",
                                                (countPredicted + countUnPredicted), countPredicted, countUnPredicted, predictionPercent,
                                                (CL_countPredicted + CL_countUnPredicted), CL_countPredicted, CL_countUnPredicted, CL_predictionPercent);

                string line3 = string.Format(@"TotalPredictions = {0}, Predicted_Abs1 = {1}, UnPredicted_Abs1 = {2}, Prediction_Abs1 percent = {3:0.00}% \n 
                                               _closedLoop_Abs1_TotalPredictions = {4}, _closedLoop_Abs1_Predicted = {5}, _closedLoop_Abs1_UnPredicted = {6}, _closedLoop_Prediction_Abs1 percent = {7:0.00}%",
                                                        (countPredicted_Abs1 + countUnPredicted_Abs1), countPredicted_Abs1, countUnPredicted_Abs1, predictionPercent_Abs1,
                                                        (CL_countPredicted_Abs1 + CL_countUnPredicted_Abs1), CL_countPredicted_Abs1, CL_countUnPredicted_Abs1, CL_predictionPercent_Abs1);

                file.WriteLine(line2);
                file.WriteLine(line3);
                Console.WriteLine(line2);
                Console.WriteLine(line3);
            }
        }



       
        private void EvaluateNetwork(Boolean blnShowConsole)
        {
            int countPredicted, countUnPredicted, CL_countPredicted, CL_countUnPredicted;
            Boolean blnPredicted, bln_closedLoop_Predicted;

            int countPredicted_Abs1, countUnPredicted_Abs1, CL_countPredicted_Abs1, CL_countUnPredicted_Abs1;
            Boolean blnPredicted_Abs1, bln_closedLoop_Predicted_Abs1;

            double actual1,actual2,actual3,actual4,actual5,actual6,actual7;
            int DrawNumber;
            double normalizedPredicted,_closedLoop_normalizedPredicted,predicted,_closedLoop_predicted;
            string lineStatus, line0, line1, line2, line3;

            //float predictionPercent;

            //int evaluateStart = data.Select(t => t.Id).Min() + PastWindowSize;
            //int evaluateStop = data.Select(t => t.Id).Max();
            //int TotalNumOfIterations = evaluateStop - evaluateStart;

            IMLData output;
            IMLData _closedLoop_output;
            lineStatus=line0=line1=line2=line3="";

            /*
                public string EvaluateFileHeader
                public string PredictFileHeader 

             */


            // if the file does not exist or empty, print an header.
            if (!File.Exists(Config.EvaluationResult.ToString()) || Config.EvaluationResult.Length == 0)
                using (System.IO.StreamWriter file = File.AppendText(Config.EvaluationResult.ToString()))
                {
                    file.WriteLine(EvaluateFileHeader);
                    file.Close();
                }

                using (System.IO.StreamWriter file = File.AppendText(Config.EvaluationResult.ToString()))
                //using (var file = new System.IO.StreamWriter(Config.EvaluationResult.ToString()))
                {
                    countPredicted = countUnPredicted = 0;
                    CL_countPredicted = CL_countUnPredicted = 0;
                    countPredicted_Abs1 = countUnPredicted_Abs1 = 0;
                    CL_countPredicted_Abs1 = CL_countUnPredicted_Abs1 = 0;

                    for (int currentId = EvaluateStart; currentId < EvaluateEnd; currentId++)
                    {

                        if (LotoNumber != 7)
                        {
                            //Calculate based on actual data
                            var input = new BasicMLData(PastWindowSize * 7);
                            var _closedLoop_input = new BasicMLData(PastWindowSize * 7);

                            for (int i = 0; i < PastWindowSize; i++)
                            {
                                input[i * 7] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t.NormalizedActual1).First();

                                input[i * 7 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t.NormalizedActual2).First();

                                input[i * 7 + 2] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t.NormalizedActual3).First();

                                input[i * 7 + 3] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t.NormalizedActual4).First();

                                input[i * 7 + 4] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t.NormalizedActual5).First();

                                input[i * 7 + 5] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t.NormalizedActual6).First();


                                input[i * 7 + 6] =
                                        (LotoNumber == 1) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual1).First() :
                                        (LotoNumber == 2) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual2).First() :
                                        (LotoNumber == 3) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual3).First() :
                                        (LotoNumber == 4) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual4).First() :
                                        (LotoNumber == 5) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual5).First() :
                                        data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual6).First();


                                _closedLoop_input[i * 7] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t._closedLoopNormalizedActual1).First();

                                _closedLoop_input[i * 7 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t._closedLoopNormalizedActual2).First();

                                _closedLoop_input[i * 7 + 2] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t._closedLoopNormalizedActual3).First();

                                _closedLoop_input[i * 7 + 3] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t._closedLoopNormalizedActual4).First();

                                _closedLoop_input[i * 7 + 4] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t._closedLoopNormalizedActual5).First();

                                _closedLoop_input[i * 7 + 5] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t._closedLoopNormalizedActual6).First();


                                _closedLoop_input[i * 7 + 6] = 
                                        (LotoNumber == 1) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.Normalized_closedLoopNA_Actual1).First() :
                                        (LotoNumber == 2) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.Normalized_closedLoopNA_Actual2).First() :
                                        (LotoNumber == 3) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.Normalized_closedLoopNA_Actual3).First() :
                                        (LotoNumber == 4) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.Normalized_closedLoopNA_Actual4).First() :
                                        (LotoNumber == 5) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.Normalized_closedLoopNA_Actual5).First() :
                                        data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.Normalized_closedLoopNA_Actual6).First();
                            }

                            output = network.Compute(input);
                            _closedLoop_output = network.Compute(_closedLoop_input);
                        }
                        else
                        {
                            //Calculate based on actual data
                            var input = new BasicMLData(PastWindowSize * 2);
                            var _closedLoop_input = new BasicMLData(PastWindowSize * 2);

                            for (int i = 0; i < PastWindowSize; i++)
                            {
                                input[i * 2] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t.NormalizedActual7).First();

                                input[i * 2 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t.NormalizedNA_Actual7).First();

                                _closedLoop_input[i * 2] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t._closedLoopNormalizedActual7).First();

                                _closedLoop_input[i * 2 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                    .Select(t => t.Normalized_closedLoopNA_Actual7).First();
                        }

                        output = network.Compute(input);
                        _closedLoop_output = network.Compute(_closedLoop_input);
                    }

                        normalizedPredicted = output[0];
                        _closedLoop_normalizedPredicted = _closedLoop_output[0];
                        predicted = 0.0;
                        _closedLoop_predicted = 0.0;

                        switch (LotoNumber)
                        {
                            case 1:
                                data[currentId]._closedLoopNormalizedActual1 = _closedLoop_normalizedPredicted;
                                predicted = Math.Round(norm1.Stats.DeNormalize(normalizedPredicted), 0);
                                _closedLoop_predicted = Math.Round(norm1.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                                NA.closedLoopNumbersDic37_Total[(int)_closedLoop_predicted]++;
                                data[currentId]._closedLoopNA_Actual1 = NA.closedLoopNumbersDic37_Total[(int)_closedLoop_predicted];

                                normalizedArray = norm8.Process(data.Select(t => t._closedLoopNA_Actual1).ToArray());
                                for (int i = 0; i < normalizedArray.Count(); i++)
                                    data[i].Normalized_closedLoopNA_Actual1 = normalizedArray[i];

                                break;

                            case 2:
                                data[currentId]._closedLoopNormalizedActual2 = _closedLoop_normalizedPredicted;
                                predicted = Math.Round(norm2.Stats.DeNormalize(normalizedPredicted), 0);
                                _closedLoop_predicted = Math.Round(norm2.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                                NA.closedLoopNumbersDic37_Total[(int)_closedLoop_predicted]++;
                                data[currentId]._closedLoopNA_Actual2 = NA.closedLoopNumbersDic37_Total[(int)_closedLoop_predicted];

                                normalizedArray = norm9.Process(data.Select(t => t._closedLoopNA_Actual2).ToArray());
                                for (int i = 0; i < normalizedArray.Count(); i++)
                                    data[i].Normalized_closedLoopNA_Actual2 = normalizedArray[i];

                                break;

                            case 3:
                                data[currentId]._closedLoopNormalizedActual3 = _closedLoop_normalizedPredicted;
                                predicted = Math.Round(norm3.Stats.DeNormalize(normalizedPredicted), 0);
                                _closedLoop_predicted = Math.Round(norm3.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                                NA.closedLoopNumbersDic37_Total[(int)_closedLoop_predicted]++;
                                data[currentId]._closedLoopNA_Actual3 = NA.closedLoopNumbersDic37_Total[(int)_closedLoop_predicted];

                                normalizedArray = norm10.Process(data.Select(t => t._closedLoopNA_Actual3).ToArray());
                                for (int i = 0; i < normalizedArray.Count(); i++)
                                    data[i].Normalized_closedLoopNA_Actual3 = normalizedArray[i];

                                break;
                            
                            case 4:
                                data[currentId]._closedLoopNormalizedActual4 = _closedLoop_normalizedPredicted;
                                predicted = Math.Round(norm4.Stats.DeNormalize(normalizedPredicted), 0);
                                _closedLoop_predicted = Math.Round(norm4.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                                NA.closedLoopNumbersDic37_Total[(int)_closedLoop_predicted]++;
                                data[currentId]._closedLoopNA_Actual4 = NA.closedLoopNumbersDic37_Total[(int)_closedLoop_predicted];

                                normalizedArray = norm11.Process(data.Select(t => t._closedLoopNA_Actual4).ToArray());
                                for (int i = 0; i < normalizedArray.Count(); i++)
                                    data[i].Normalized_closedLoopNA_Actual4 = normalizedArray[i];

                                break;

                            case 5:
                                data[currentId]._closedLoopNormalizedActual5 = _closedLoop_normalizedPredicted;
                                predicted = Math.Round(norm5.Stats.DeNormalize(normalizedPredicted), 0);
                                _closedLoop_predicted = Math.Round(norm5.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                                NA.closedLoopNumbersDic37_Total[(int)_closedLoop_predicted]++;
                                data[currentId]._closedLoopNA_Actual5 = NA.closedLoopNumbersDic37_Total[(int)_closedLoop_predicted];

                                normalizedArray = norm12.Process(data.Select(t => t._closedLoopNA_Actual5).ToArray());
                                for (int i = 0; i < normalizedArray.Count(); i++)
                                    data[i].Normalized_closedLoopNA_Actual5 = normalizedArray[i];

                                break;

                            case 6:
                                data[currentId]._closedLoopNormalizedActual6 = _closedLoop_normalizedPredicted;
                                predicted = Math.Round(norm6.Stats.DeNormalize(normalizedPredicted), 0);
                                _closedLoop_predicted = Math.Round(norm6.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                                NA.closedLoopNumbersDic37_Total[(int)_closedLoop_predicted]++;
                                data[currentId]._closedLoopNA_Actual6 = NA.closedLoopNumbersDic37_Total[(int)_closedLoop_predicted];

                                normalizedArray = norm13.Process(data.Select(t => t._closedLoopNA_Actual6).ToArray());
                                for (int i = 0; i < normalizedArray.Count(); i++)
                                    data[i].Normalized_closedLoopNA_Actual6 = normalizedArray[i];

                                break;

                            case 7:
                                data[currentId]._closedLoopNormalizedActual7 = _closedLoop_normalizedPredicted;
                                predicted = Math.Round(norm7.Stats.DeNormalize(normalizedPredicted), 0);
                                _closedLoop_predicted = Math.Round(norm7.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                                NA.closedLoopNumbersDic7_Total[(int)_closedLoop_predicted]++;
                                data[currentId]._closedLoopNA_Actual7 = NA.closedLoopNumbersDic7_Total[(int)_closedLoop_predicted];

                                normalizedArray = norm14.Process(data.Select(t => t._closedLoopNA_Actual7).ToArray());
                                for (int i = 0; i < normalizedArray.Count(); i++)
                                    data[i].Normalized_closedLoopNA_Actual7 = normalizedArray[i];

                                break;

                            default:
                                break;
                        }



                        //double Actual1 = data.Where(t => t.Id == currentId).Select(t => t.Actual1).First();
                        actual1 = actual2 = actual3 = actual4 = actual5 = actual6 = actual7 = 0.0;

                        if (LotoNumber != 7)
                        {
                            actual1 = Math.Round(norm1.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual1).First()), 0);
                            actual2 = Math.Round(norm2.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual2).First()), 0);
                            actual3 = Math.Round(norm3.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual3).First()), 0);
                            actual4 = Math.Round(norm4.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual4).First()), 0);
                            actual5 = Math.Round(norm5.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual5).First()), 0);
                            actual6 = Math.Round(norm6.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual6).First()), 0);
                        }
                        else
                            actual7 = Math.Round(norm7.Stats.DeNormalize(data.Where(t => t.Id == currentId).Select(t => t.NormalizedActual7).First()), 0);

                        DrawNumber = data.Where(t => t.Id == currentId).Select(t => t.DrawNumber).First();
                        if (LotoNumber != 7)
                        {
                            blnPredicted = (actual1 == predicted ||
                                            actual2 == predicted ||
                                            actual3 == predicted ||
                                            actual4 == predicted ||
                                            actual5 == predicted ||
                                            actual6 == predicted) ? true : false;

                            bln_closedLoop_Predicted = (actual1 == _closedLoop_predicted ||
                                                        actual2 == _closedLoop_predicted ||
                                                        actual3 == _closedLoop_predicted ||
                                                        actual4 == _closedLoop_predicted ||
                                                        actual5 == _closedLoop_predicted ||
                                                        actual6 == _closedLoop_predicted) ? true : false;

                            blnPredicted_Abs1 = (Math.Abs(actual1 - predicted) <= 1 ||
                                                 Math.Abs(actual2 - predicted) <= 1 ||
                                                 Math.Abs(actual3 - predicted) <= 1 ||
                                                 Math.Abs(actual4 - predicted) <= 1 ||
                                                 Math.Abs(actual5 - predicted) <= 1 ||
                                                 Math.Abs(actual6 - predicted) <= 1) ? true : false;

                            bln_closedLoop_Predicted_Abs1 = (Math.Abs(actual1 - _closedLoop_predicted) <= 1 ||
                                                             Math.Abs(actual2 - _closedLoop_predicted) <= 1 ||
                                                             Math.Abs(actual3 - _closedLoop_predicted) <= 1 ||
                                                             Math.Abs(actual4 - _closedLoop_predicted) <= 1 ||
                                                             Math.Abs(actual5 - _closedLoop_predicted) <= 1 ||
                                                             Math.Abs(actual6 - _closedLoop_predicted) <= 1) ? true : false;
                        }
                        else
                        {
                            blnPredicted = (actual7 == predicted) ? true : false;
                            bln_closedLoop_Predicted = (actual7 == _closedLoop_predicted) ? true : false;

                            blnPredicted_Abs1 = (Math.Abs(actual7 - predicted) <= 1) ? true : false;
                            bln_closedLoop_Predicted_Abs1 = (Math.Abs(actual7 - _closedLoop_predicted) <= 1) ? true : false;
                        }


                        if (blnPredicted)
                            countPredicted++;
                        else
                            countUnPredicted++;

                        if (bln_closedLoop_Predicted)
                            CL_countPredicted++;
                        else
                            CL_countUnPredicted++;


                        if (blnPredicted_Abs1)
                            countPredicted_Abs1++;
                        else
                            countUnPredicted_Abs1++;

                        if (bln_closedLoop_Predicted_Abs1)
                            CL_countPredicted_Abs1++;
                        else
                            CL_countUnPredicted_Abs1++;



                        if (LotoNumber != 7)
                        {
                            line1 = string.Format("DrawNumber: {0}; Actual: ({1},{2},{3},{4},{5},{6}); Predicted: {7}; closedLoop_Predicted: {8}", DrawNumber, actual1, actual2, actual3, actual4, actual5, actual6, predicted, _closedLoop_predicted);
                            line0 = string.Format("{0};{1}", DrawNumber, LotoNumber);
                        }
                        else
                        {
                            line1 = string.Format("DrawNumber: {0}; Actual: {1}; Predicted: {2}; closedLoop_Predicted: {3}", DrawNumber, actual7, predicted, _closedLoop_predicted);
                            line0 = string.Format("{0};{1}", DrawNumber, LotoNumber);
                        }

                        //file.WriteLine(line1);
                        if (blnShowConsole)
                            Console.WriteLine(line1);

                    }

                    predictionPercent = ((float)countPredicted / ((float)countPredicted + (float)countUnPredicted)) * 100;
                    CL_predictionPercent = ((float)CL_countPredicted / ((float)CL_countPredicted + (float)CL_countUnPredicted)) * 100;

                    predictionPercent_Abs1 = ((float)countPredicted_Abs1 / ((float)countPredicted_Abs1 + (float)countUnPredicted_Abs1)) * 100;
                    CL_predictionPercent_Abs1 = ((float)CL_countPredicted_Abs1 / ((float)CL_countPredicted_Abs1 + (float)CL_countUnPredicted_Abs1)) * 100;


                    line2 = string.Format(@"TotalPredictions = {0}, Predicted = {1}, UnPredicted = {2}, Prediction percent = {3:0.00}% \n 
                                               _closedLoop_TotalPredictions = {4}, _closedLoop_Predicted = {5}, _closedLoop_UnPredicted = {6}, _closedLoop_Prediction percent = {7:0.00}%",
                                                    (countPredicted + countUnPredicted), countPredicted, countUnPredicted, predictionPercent,
                                                    (CL_countPredicted + CL_countUnPredicted), CL_countPredicted, CL_countUnPredicted, CL_predictionPercent);

                    line3 = string.Format(@"TotalPredictions = {0}, Predicted_Abs1 = {1}, UnPredicted_Abs1 = {2}, Prediction_Abs1 percent = {3:0.00}% \n 
                                               _closedLoop_Abs1_TotalPredictions = {4}, _closedLoop_Abs1_Predicted = {5}, _closedLoop_Abs1_UnPredicted = {6}, _closedLoop_Prediction_Abs1 percent = {7:0.00}%",
                                                            (countPredicted_Abs1 + countUnPredicted_Abs1), countPredicted_Abs1, countUnPredicted_Abs1, predictionPercent_Abs1,
                                                            (CL_countPredicted_Abs1 + CL_countUnPredicted_Abs1), CL_countPredicted_Abs1, CL_countUnPredicted_Abs1, CL_predictionPercent_Abs1);



                    lineStatus =
                    string.Format(@"{0};{1};{2};{3};{4:0.00}%;{5};{6};{7:0.00}%;{8};{9};{10:0.00}%;{11};{12};{13:0.00}%",
                                    line0,
                                    (countPredicted + countUnPredicted),
                                    countPredicted,
                                    countUnPredicted,
                                    predictionPercent,
                                    CL_countPredicted,
                                    CL_countUnPredicted,
                                    CL_predictionPercent,
                                    countPredicted_Abs1,
                                    countUnPredicted_Abs1,
                                    predictionPercent_Abs1,
                                    CL_countPredicted_Abs1,
                                    CL_countUnPredicted_Abs1,
                                    CL_predictionPercent_Abs1);

                
                    //file.WriteLine(line2);
                    //file.WriteLine(line3);
                    file.WriteLine(lineStatus);
                    file.Close();

                    //
                    //Removed for faster execution...
                    //Console.WriteLine(line2);
                    //Console.WriteLine(line3);
                    //Console.WriteLine(lineStatus);
                    //


                    ////Save best performant network per each indicator


                    //Set("MAX_predictionPercent", "0");
                    //Set("MAX_CL_predictionPercent", "0");
                    //Set("MAX_predictionPercent_Abs1", "0");
                    //Set("MAX_CL_predictionPercent_Abs1", "0");

                    //f.ToString("R");
                    //12345.678901.ToString("0.0000");


                    string strMAX_predictionPercent = GetSetting("MAX_predictionPercent");
                    string strMAX_CL_predictionPercent = GetSetting("MAX_CL_predictionPercent");
                    string strMAX_predictionPercent_Abs1 = GetSetting("MAX_predictionPercent_Abs1");
                    string strMAX_CL_predictionPercent_Abs1 = GetSetting("MAX_CL_predictionPercent_Abs1");

                    string strMAX_predictionPercent_N1 = GetSetting("MAX_predictionPercent_N1");
                    string strMAX_CL_predictionPercent_N1 = GetSetting("MAX_CL_predictionPercent_N1");
                    string strMAX_predictionPercent_Abs1_N1 = GetSetting("MAX_predictionPercent_Abs1_N1");
                    string strMAX_CL_predictionPercent_Abs1_N1 = GetSetting("MAX_CL_predictionPercent_Abs1_N1");

                    string strMAX_predictionPercent_N2 = GetSetting("MAX_predictionPercent_N2");
                    string strMAX_CL_predictionPercent_N2 = GetSetting("MAX_CL_predictionPercent_N2");
                    string strMAX_predictionPercent_Abs1_N2 = GetSetting("MAX_predictionPercent_Abs1_N2");
                    string strMAX_CL_predictionPercent_Abs1_N2 = GetSetting("MAX_CL_predictionPercent_Abs1_N2");

                    string strMAX_predictionPercent_N3 = GetSetting("MAX_predictionPercent_N3");
                    string strMAX_CL_predictionPercent_N3 = GetSetting("MAX_CL_predictionPercent_N3");
                    string strMAX_predictionPercent_Abs1_N3 = GetSetting("MAX_predictionPercent_Abs1_N3");
                    string strMAX_CL_predictionPercent_Abs1_N3 = GetSetting("MAX_CL_predictionPercent_Abs1_N3");

                    string strMAX_predictionPercent_N4 = GetSetting("MAX_predictionPercent_N4");
                    string strMAX_CL_predictionPercent_N4 = GetSetting("MAX_CL_predictionPercent_N4");
                    string strMAX_predictionPercent_Abs1_N4 = GetSetting("MAX_predictionPercent_Abs1_N4");
                    string strMAX_CL_predictionPercent_Abs1_N4 = GetSetting("MAX_CL_predictionPercent_Abs1_N4");

                    string strMAX_predictionPercent_N5 = GetSetting("MAX_predictionPercent_N5");
                    string strMAX_CL_predictionPercent_N5 = GetSetting("MAX_CL_predictionPercent_N5");
                    string strMAX_predictionPercent_Abs1_N5 = GetSetting("MAX_predictionPercent_Abs1_N5");
                    string strMAX_CL_predictionPercent_Abs1_N5 = GetSetting("MAX_CL_predictionPercent_Abs1_N5");

                    string strMAX_predictionPercent_N6 = GetSetting("MAX_predictionPercent_N6");
                    string strMAX_CL_predictionPercent_N6 = GetSetting("MAX_CL_predictionPercent_N6");
                    string strMAX_predictionPercent_Abs1_N6 = GetSetting("MAX_predictionPercent_Abs1_N6");
                    string strMAX_CL_predictionPercent_Abs1_N6 = GetSetting("MAX_CL_predictionPercent_Abs1_N6");

                    string strMAX_predictionPercent_N7 = GetSetting("MAX_predictionPercent_N7");
                    string strMAX_CL_predictionPercent_N7 = GetSetting("MAX_CL_predictionPercent_N7");
                    string strMAX_predictionPercent_Abs1_N7 = GetSetting("MAX_predictionPercent_Abs1_N7");
                    string strMAX_CL_predictionPercent_Abs1_N7 = GetSetting("MAX_CL_predictionPercent_Abs1_N7");

                    if (strMAX_predictionPercent == null) Set("MAX_predictionPercent", "0");
                    if (strMAX_CL_predictionPercent == null) Set("MAX_CL_predictionPercent", "0");
                    if (strMAX_predictionPercent_Abs1 == null) Set("MAX_predictionPercent_Abs1", "0");
                    if (strMAX_CL_predictionPercent_Abs1 == null) Set("MAX_CL_predictionPercent_Abs1", "0");

                    if (strMAX_predictionPercent_N1 == null) Set("MAX_predictionPercent_N1", "0");
                    if (strMAX_CL_predictionPercent_N1 == null) Set("MAX_CL_predictionPercent_N1", "0");
                    if (strMAX_predictionPercent_Abs1_N1 == null) Set("MAX_predictionPercent_Abs1_N1", "0");
                    if (strMAX_CL_predictionPercent_Abs1_N1 == null) Set("MAX_CL_predictionPercent_Abs1_N1", "0");

                    if (strMAX_predictionPercent_N2 == null) Set("MAX_predictionPercent_N2", "0");
                    if (strMAX_CL_predictionPercent_N2 == null) Set("MAX_CL_predictionPercent_N2", "0");
                    if (strMAX_predictionPercent_Abs1_N2 == null) Set("MAX_predictionPercent_Abs1_N2", "0");
                    if (strMAX_CL_predictionPercent_Abs1_N2 == null) Set("MAX_CL_predictionPercent_Abs1_N2", "0");

                    if (strMAX_predictionPercent_N3 == null) Set("MAX_predictionPercent_N3", "0");
                    if (strMAX_CL_predictionPercent_N3 == null) Set("MAX_CL_predictionPercent_N3", "0");
                    if (strMAX_predictionPercent_Abs1_N3 == null) Set("MAX_predictionPercent_Abs1_N3", "0");
                    if (strMAX_CL_predictionPercent_Abs1_N3 == null) Set("MAX_CL_predictionPercent_Abs1_N3", "0");

                    if (strMAX_predictionPercent_N4 == null) Set("MAX_predictionPercent_N4", "0");
                    if (strMAX_CL_predictionPercent_N4 == null) Set("MAX_CL_predictionPercent_N4", "0");
                    if (strMAX_predictionPercent_Abs1_N4 == null) Set("MAX_predictionPercent_Abs1_N4", "0");
                    if (strMAX_CL_predictionPercent_Abs1_N4 == null) Set("MAX_CL_predictionPercent_Abs1_N4", "0");

                    if (strMAX_predictionPercent_N5 == null) Set("MAX_predictionPercent_N5", "0");
                    if (strMAX_CL_predictionPercent_N5 == null) Set("MAX_CL_predictionPercent_N5", "0");
                    if (strMAX_predictionPercent_Abs1_N5 == null) Set("MAX_predictionPercent_Abs1_N5", "0");
                    if (strMAX_CL_predictionPercent_Abs1_N5 == null) Set("MAX_CL_predictionPercent_Abs1_N5", "0");

                    if (strMAX_predictionPercent_N6 == null) Set("MAX_predictionPercent_N6", "0");
                    if (strMAX_CL_predictionPercent_N6 == null) Set("MAX_CL_predictionPercent_N6", "0");
                    if (strMAX_predictionPercent_Abs1_N6 == null) Set("MAX_predictionPercent_Abs1_N6", "0");
                    if (strMAX_CL_predictionPercent_Abs1_N6 == null) Set("MAX_CL_predictionPercent_Abs1_N6", "0");

                    if (strMAX_predictionPercent_N7 == null) Set("MAX_predictionPercent_N7", "0");
                    if (strMAX_CL_predictionPercent_N7 == null) Set("MAX_CL_predictionPercent_N7", "0");
                    if (strMAX_predictionPercent_Abs1_N7 == null) Set("MAX_predictionPercent_Abs1_N7", "0");
                    if (strMAX_CL_predictionPercent_Abs1_N7 == null) Set("MAX_CL_predictionPercent_Abs1_N7", "0");



                    MAX_predictionPercent = float.Parse(GetSetting("MAX_predictionPercent"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent = float.Parse(GetSetting("MAX_CL_predictionPercent"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_predictionPercent_Abs1 = float.Parse(GetSetting("MAX_predictionPercent_Abs1"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent_Abs1 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1"), System.Globalization.CultureInfo.InvariantCulture);

                    MAX_predictionPercent_N1 = float.Parse(GetSetting("MAX_predictionPercent_N1"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent_N1 = float.Parse(GetSetting("MAX_CL_predictionPercent_N1"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_predictionPercent_Abs1_N1 = float.Parse(GetSetting("MAX_predictionPercent_Abs1_N1"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent_Abs1_N1 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1_N1"), System.Globalization.CultureInfo.InvariantCulture);

                    MAX_predictionPercent_N2 = float.Parse(GetSetting("MAX_predictionPercent_N2"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent_N2 = float.Parse(GetSetting("MAX_CL_predictionPercent_N2"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_predictionPercent_Abs1_N2 = float.Parse(GetSetting("MAX_predictionPercent_Abs1_N2"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent_Abs1_N2 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1_N2"), System.Globalization.CultureInfo.InvariantCulture);

                    MAX_predictionPercent_N3 = float.Parse(GetSetting("MAX_predictionPercent_N3"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent_N3 = float.Parse(GetSetting("MAX_CL_predictionPercent_N3"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_predictionPercent_Abs1_N3 = float.Parse(GetSetting("MAX_predictionPercent_Abs1_N3"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent_Abs1_N3 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1_N3"), System.Globalization.CultureInfo.InvariantCulture);

                    MAX_predictionPercent_N4 = float.Parse(GetSetting("MAX_predictionPercent_N4"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent_N4 = float.Parse(GetSetting("MAX_CL_predictionPercent_N4"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_predictionPercent_Abs1_N4 = float.Parse(GetSetting("MAX_predictionPercent_Abs1_N4"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent_Abs1_N4 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1_N4"), System.Globalization.CultureInfo.InvariantCulture);

                    MAX_predictionPercent_N5 = float.Parse(GetSetting("MAX_predictionPercent_N5"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent_N5 = float.Parse(GetSetting("MAX_CL_predictionPercent_N5"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_predictionPercent_Abs1_N5 = float.Parse(GetSetting("MAX_predictionPercent_Abs1_N5"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent_Abs1_N5 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1_N5"), System.Globalization.CultureInfo.InvariantCulture);

                    MAX_predictionPercent_N6 = float.Parse(GetSetting("MAX_predictionPercent_N6"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent_N6 = float.Parse(GetSetting("MAX_CL_predictionPercent_N6"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_predictionPercent_Abs1_N6 = float.Parse(GetSetting("MAX_predictionPercent_Abs1_N6"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent_Abs1_N6 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1_N6"), System.Globalization.CultureInfo.InvariantCulture);

                    MAX_predictionPercent_N7 = float.Parse(GetSetting("MAX_predictionPercent_N7"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent_N7 = float.Parse(GetSetting("MAX_CL_predictionPercent_N7"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_predictionPercent_Abs1_N7 = float.Parse(GetSetting("MAX_predictionPercent_Abs1_N7"), System.Globalization.CultureInfo.InvariantCulture);
                    MAX_CL_predictionPercent_Abs1_N7 = float.Parse(GetSetting("MAX_CL_predictionPercent_Abs1_N7"), System.Globalization.CultureInfo.InvariantCulture);




                    //Set("MAX_predictionPercent", MAX_predictionPercent.ToString("R"));
                    //Set("MAX_CL_predictionPercent", MAX_CL_predictionPercent.ToString("R"));
                    //Set("MAX_predictionPercent_Abs1", MAX_predictionPercent_Abs1.ToString("R"));
                    //Set("MAX_CL_predictionPercent_Abs1", MAX_CL_predictionPercent_Abs1.ToString("R"));




                    if (predictionPercent > MAX_predictionPercent)
                    {
                        MAX_predictionPercent = predictionPercent;
                        SaveLoadNetwork(true, Config.MAX_predictionPercentFile.ToString());
                        Set("MAX_predictionPercent", MAX_predictionPercent.ToString("R"));  //Persist the value
                    }

                    if (CL_predictionPercent > MAX_CL_predictionPercent)
                    {
                        MAX_CL_predictionPercent = CL_predictionPercent;
                        SaveLoadNetwork(true, Config.MAX_CL_predictionPercentFile.ToString());
                        Set("MAX_CL_predictionPercent", MAX_CL_predictionPercent.ToString("R"));
                    }

                    if (predictionPercent_Abs1 > MAX_predictionPercent_Abs1)
                    {
                        MAX_predictionPercent_Abs1 = predictionPercent_Abs1;
                        SaveLoadNetwork(true, Config.MAX_predictionPercent_Abs1File.ToString());
                        Set("MAX_predictionPercent_Abs1", MAX_predictionPercent_Abs1.ToString("R"));
                    }

                    if (CL_predictionPercent_Abs1 > MAX_CL_predictionPercent_Abs1)
                    {
                        MAX_CL_predictionPercent_Abs1 = CL_predictionPercent_Abs1;
                        SaveLoadNetwork(true, Config.MAX_CL_predictionPercent_Abs1File.ToString());
                        Set("MAX_CL_predictionPercent_Abs1", MAX_CL_predictionPercent_Abs1.ToString("R"));
                    }

                     switch (LotoNumber)
                    {
                        case 1:
                            if (predictionPercent > MAX_predictionPercent_N1)
                            {
                                MAX_predictionPercent_N1 = predictionPercent;
                                SaveLoadNetwork(true, Config.MAX_predictionPercentFile_N1.ToString());
                                Set("MAX_predictionPercent_N1", MAX_predictionPercent_N1.ToString("R"));  //Persist the value
                            }

                            if (CL_predictionPercent > MAX_CL_predictionPercent_N1)
                            {
                                MAX_CL_predictionPercent_N1 = CL_predictionPercent;
                                SaveLoadNetwork(true, Config.MAX_CL_predictionPercentFile_N1.ToString());
                                Set("MAX_CL_predictionPercent_N1", MAX_CL_predictionPercent_N1.ToString("R"));
                            }

                            if (predictionPercent_Abs1 > MAX_predictionPercent_Abs1_N1)
                            {
                                MAX_predictionPercent_Abs1_N1 = predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MAX_predictionPercent_Abs1File_N1.ToString());
                                Set("MAX_predictionPercent_Abs1_N1", MAX_predictionPercent_Abs1_N1.ToString("R"));
                            }

                            if (CL_predictionPercent_Abs1 > MAX_CL_predictionPercent_Abs1_N1)
                            {
                                MAX_CL_predictionPercent_Abs1_N1 = CL_predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MAX_CL_predictionPercent_Abs1File_N1.ToString());
                                Set("MAX_CL_predictionPercent_Abs1_N1", MAX_CL_predictionPercent_Abs1_N1.ToString("R"));
                            }


                            break;

                        case 2:
                            if (predictionPercent > MAX_predictionPercent_N2)
                            {
                                MAX_predictionPercent_N2 = predictionPercent;
                                SaveLoadNetwork(true, Config.MAX_predictionPercentFile_N2.ToString());
                                Set("MAX_predictionPercent_N2", MAX_predictionPercent_N2.ToString("R"));  //Persist the value
                            }

                            if (CL_predictionPercent > MAX_CL_predictionPercent_N2)
                            {
                                MAX_CL_predictionPercent_N2 = CL_predictionPercent;
                                SaveLoadNetwork(true, Config.MAX_CL_predictionPercentFile_N2.ToString());
                                Set("MAX_CL_predictionPercent_N2", MAX_CL_predictionPercent_N2.ToString("R"));
                            }

                            if (predictionPercent_Abs1 > MAX_predictionPercent_Abs1_N2)
                            {
                                MAX_predictionPercent_Abs1_N2 = predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MAX_predictionPercent_Abs1File_N2.ToString());
                                Set("MAX_predictionPercent_Abs1_N2", MAX_predictionPercent_Abs1_N2.ToString("R"));
                            }

                            if (CL_predictionPercent_Abs1 > MAX_CL_predictionPercent_Abs1_N2)
                            {
                                MAX_CL_predictionPercent_Abs1_N2 = CL_predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MAX_CL_predictionPercent_Abs1File_N2.ToString());
                                Set("MAX_CL_predictionPercent_Abs1_N2", MAX_CL_predictionPercent_Abs1_N2.ToString("R"));
                            }


                            break;

                        case 3:
                            if (predictionPercent > MAX_predictionPercent_N3)
                            {
                                MAX_predictionPercent_N3 = predictionPercent;
                                SaveLoadNetwork(true, Config.MAX_predictionPercentFile_N3.ToString());
                                Set("MAX_predictionPercent_N3", MAX_predictionPercent_N3.ToString("R"));  //Persist the value
                            }

                            if (CL_predictionPercent > MAX_CL_predictionPercent_N3)
                            {
                                MAX_CL_predictionPercent_N3 = CL_predictionPercent;
                                SaveLoadNetwork(true, Config.MAX_CL_predictionPercentFile_N3.ToString());
                                Set("MAX_CL_predictionPercent_N3", MAX_CL_predictionPercent_N3.ToString("R"));
                            }

                            if (predictionPercent_Abs1 > MAX_predictionPercent_Abs1_N3)
                            {
                                MAX_predictionPercent_Abs1_N3 = predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MAX_predictionPercent_Abs1File_N3.ToString());
                                Set("MAX_predictionPercent_Abs1_N3", MAX_predictionPercent_Abs1_N3.ToString("R"));
                            }

                            if (CL_predictionPercent_Abs1 > MAX_CL_predictionPercent_Abs1_N3)
                            {
                                MAX_CL_predictionPercent_Abs1_N3 = CL_predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MAX_CL_predictionPercent_Abs1File_N3.ToString());
                                Set("MAX_CL_predictionPercent_Abs1_N3", MAX_CL_predictionPercent_Abs1_N3.ToString("R"));
                            }


                            break;

                        case 4:
                            if (predictionPercent > MAX_predictionPercent_N4)
                            {
                                MAX_predictionPercent_N4 = predictionPercent;
                                SaveLoadNetwork(true, Config.MAX_predictionPercentFile_N4.ToString());
                                Set("MAX_predictionPercent_N4", MAX_predictionPercent_N4.ToString("R"));  //Persist the value
                            }

                            if (CL_predictionPercent > MAX_CL_predictionPercent_N4)
                            {
                                MAX_CL_predictionPercent_N4 = CL_predictionPercent;
                                SaveLoadNetwork(true, Config.MAX_CL_predictionPercentFile_N4.ToString());
                                Set("MAX_CL_predictionPercent_N4", MAX_CL_predictionPercent_N4.ToString("R"));
                            }

                            if (predictionPercent_Abs1 > MAX_predictionPercent_Abs1_N4)
                            {
                                MAX_predictionPercent_Abs1_N4 = predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MAX_predictionPercent_Abs1File_N4.ToString());
                                Set("MAX_predictionPercent_Abs1_N4", MAX_predictionPercent_Abs1_N4.ToString("R"));
                            }

                            if (CL_predictionPercent_Abs1 > MAX_CL_predictionPercent_Abs1_N4)
                            {
                                MAX_CL_predictionPercent_Abs1_N4 = CL_predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MAX_CL_predictionPercent_Abs1File_N4.ToString());
                                Set("MAX_CL_predictionPercent_Abs1_N4", MAX_CL_predictionPercent_Abs1_N4.ToString("R"));
                            }

                            break;

                        case 5:
                            if (predictionPercent > MAX_predictionPercent_N5)
                            {
                                MAX_predictionPercent_N5 = predictionPercent;
                                SaveLoadNetwork(true, Config.MAX_predictionPercentFile_N5.ToString());
                                Set("MAX_predictionPercent_N5", MAX_predictionPercent_N5.ToString("R"));  //Persist the value
                            }

                            if (CL_predictionPercent > MAX_CL_predictionPercent_N5)
                            {
                                MAX_CL_predictionPercent_N5 = CL_predictionPercent;
                                SaveLoadNetwork(true, Config.MAX_CL_predictionPercentFile_N5.ToString());
                                Set("MAX_CL_predictionPercent_N5", MAX_CL_predictionPercent_N5.ToString("R"));
                            }

                            if (predictionPercent_Abs1 > MAX_predictionPercent_Abs1_N5)
                            {
                                MAX_predictionPercent_Abs1_N5 = predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MAX_predictionPercent_Abs1File_N5.ToString());
                                Set("MAX_predictionPercent_Abs1_N5", MAX_predictionPercent_Abs1_N5.ToString("R"));
                            }

                            if (CL_predictionPercent_Abs1 > MAX_CL_predictionPercent_Abs1_N5)
                            {
                                MAX_CL_predictionPercent_Abs1_N5 = CL_predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MAX_CL_predictionPercent_Abs1File_N5.ToString());
                                Set("MAX_CL_predictionPercent_Abs1_N5", MAX_CL_predictionPercent_Abs1_N5.ToString("R"));
                            }

                            break;

                        case 6:
                            if (predictionPercent > MAX_predictionPercent_N6)
                            {
                                MAX_predictionPercent_N6 = predictionPercent;
                                SaveLoadNetwork(true, Config.MAX_predictionPercentFile_N6.ToString());
                                Set("MAX_predictionPercent_N6", MAX_predictionPercent_N6.ToString("R"));  //Persist the value
                            }

                            if (CL_predictionPercent > MAX_CL_predictionPercent_N6)
                            {
                                MAX_CL_predictionPercent_N6 = CL_predictionPercent;
                                SaveLoadNetwork(true, Config.MAX_CL_predictionPercentFile_N6.ToString());
                                Set("MAX_CL_predictionPercent_N6", MAX_CL_predictionPercent_N6.ToString("R"));
                            }

                            if (predictionPercent_Abs1 > MAX_predictionPercent_Abs1_N6)
                            {
                                MAX_predictionPercent_Abs1_N6 = predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MAX_predictionPercent_Abs1File_N6.ToString());
                                Set("MAX_predictionPercent_Abs1_N6", MAX_predictionPercent_Abs1_N6.ToString("R"));
                            }

                            if (CL_predictionPercent_Abs1 > MAX_CL_predictionPercent_Abs1_N6)
                            {
                                MAX_CL_predictionPercent_Abs1_N6 = CL_predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MAX_CL_predictionPercent_Abs1File_N6.ToString());
                                Set("MAX_CL_predictionPercent_Abs1_N6", MAX_CL_predictionPercent_Abs1_N6.ToString("R"));
                            }

                            break;

                        case 7:
                            if (predictionPercent > MAX_predictionPercent_N7)
                            {
                                MAX_predictionPercent_N7 = predictionPercent;
                                SaveLoadNetwork(true, Config.MAX_predictionPercentFile_N7.ToString());
                                Set("MAX_predictionPercent_N7", MAX_predictionPercent_N7.ToString("R"));  //Persist the value
                            }

                            if (CL_predictionPercent > MAX_CL_predictionPercent_N7)
                            {
                                MAX_CL_predictionPercent_N7 = CL_predictionPercent;
                                SaveLoadNetwork(true, Config.MAX_CL_predictionPercentFile_N7.ToString());
                                Set("MAX_CL_predictionPercent_N7", MAX_CL_predictionPercent_N7.ToString("R"));
                            }

                            if (predictionPercent_Abs1 > MAX_predictionPercent_Abs1_N7)
                            {
                                MAX_predictionPercent_Abs1_N7 = predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MAX_predictionPercent_Abs1File_N7.ToString());
                                Set("MAX_predictionPercent_Abs1_N7", MAX_predictionPercent_Abs1_N7.ToString("R"));
                            }

                            if (CL_predictionPercent_Abs1 > MAX_CL_predictionPercent_Abs1_N7)
                            {
                                MAX_CL_predictionPercent_Abs1_N7 = CL_predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MAX_CL_predictionPercent_Abs1File_N7.ToString());
                                Set("MAX_CL_predictionPercent_Abs1_N7", MAX_CL_predictionPercent_Abs1_N7.ToString("R"));
                            }

                            break;

                        default:
                            break;
                    }


                    //Min Handling
                    string strMIN_predictionPercent = GetSetting("MIN_predictionPercent");
                    string strMIN_CL_predictionPercent = GetSetting("MIN_CL_predictionPercent");
                    string strMIN_predictionPercent_Abs1 = GetSetting("MIN_predictionPercent_Abs1");
                    string strMIN_CL_predictionPercent_Abs1 = GetSetting("MIN_CL_predictionPercent_Abs1");

                    string strMIN_predictionPercent_N1 = GetSetting("MIN_predictionPercent_N1");
                    string strMIN_CL_predictionPercent_N1 = GetSetting("MIN_CL_predictionPercent_N1");
                    string strMIN_predictionPercent_Abs1_N1 = GetSetting("MIN_predictionPercent_Abs1_N1");
                    string strMIN_CL_predictionPercent_Abs1_N1 = GetSetting("MIN_CL_predictionPercent_Abs1_N1");

                    string strMIN_predictionPercent_N2 = GetSetting("MIN_predictionPercent_N2");
                    string strMIN_CL_predictionPercent_N2 = GetSetting("MIN_CL_predictionPercent_N2");
                    string strMIN_predictionPercent_Abs1_N2 = GetSetting("MIN_predictionPercent_Abs1_N2");
                    string strMIN_CL_predictionPercent_Abs1_N2 = GetSetting("MIN_CL_predictionPercent_Abs1_N2");

                    string strMIN_predictionPercent_N3 = GetSetting("MIN_predictionPercent_N3");
                    string strMIN_CL_predictionPercent_N3 = GetSetting("MIN_CL_predictionPercent_N3");
                    string strMIN_predictionPercent_Abs1_N3 = GetSetting("MIN_predictionPercent_Abs1_N3");
                    string strMIN_CL_predictionPercent_Abs1_N3 = GetSetting("MIN_CL_predictionPercent_Abs1_N3");

                    string strMIN_predictionPercent_N4 = GetSetting("MIN_predictionPercent_N4");
                    string strMIN_CL_predictionPercent_N4 = GetSetting("MIN_CL_predictionPercent_N4");
                    string strMIN_predictionPercent_Abs1_N4 = GetSetting("MIN_predictionPercent_Abs1_N4");
                    string strMIN_CL_predictionPercent_Abs1_N4 = GetSetting("MIN_CL_predictionPercent_Abs1_N4");

                    string strMIN_predictionPercent_N5 = GetSetting("MIN_predictionPercent_N5");
                    string strMIN_CL_predictionPercent_N5 = GetSetting("MIN_CL_predictionPercent_N5");
                    string strMIN_predictionPercent_Abs1_N5 = GetSetting("MIN_predictionPercent_Abs1_N5");
                    string strMIN_CL_predictionPercent_Abs1_N5 = GetSetting("MIN_CL_predictionPercent_Abs1_N5");

                    string strMIN_predictionPercent_N6 = GetSetting("MIN_predictionPercent_N6");
                    string strMIN_CL_predictionPercent_N6 = GetSetting("MIN_CL_predictionPercent_N6");
                    string strMIN_predictionPercent_Abs1_N6 = GetSetting("MIN_predictionPercent_Abs1_N6");
                    string strMIN_CL_predictionPercent_Abs1_N6 = GetSetting("MIN_CL_predictionPercent_Abs1_N6");

                    string strMIN_predictionPercent_N7 = GetSetting("MIN_predictionPercent_N7");
                    string strMIN_CL_predictionPercent_N7 = GetSetting("MIN_CL_predictionPercent_N7");
                    string strMIN_predictionPercent_Abs1_N7 = GetSetting("MIN_predictionPercent_Abs1_N7");
                    string strMIN_CL_predictionPercent_Abs1_N7 = GetSetting("MIN_CL_predictionPercent_Abs1_N7");

                    if (strMIN_predictionPercent == null) Set("MIN_predictionPercent", "60");
                    if (strMIN_CL_predictionPercent == null) Set("MIN_CL_predictionPercent", "60");
                    if (strMIN_predictionPercent_Abs1 == null) Set("MIN_predictionPercent_Abs1", "60");
                    if (strMIN_CL_predictionPercent_Abs1 == null) Set("MIN_CL_predictionPercent_Abs1", "60");

                    if (strMIN_predictionPercent_N1 == null) Set("MIN_predictionPercent_N1", "60");
                    if (strMIN_CL_predictionPercent_N1 == null) Set("MIN_CL_predictionPercent_N1", "60");
                    if (strMIN_predictionPercent_Abs1_N1 == null) Set("MIN_predictionPercent_Abs1_N1", "60");
                    if (strMIN_CL_predictionPercent_Abs1_N1 == null) Set("MIN_CL_predictionPercent_Abs1_N1", "60");

                    if (strMIN_predictionPercent_N2 == null) Set("MIN_predictionPercent_N2", "60");
                    if (strMIN_CL_predictionPercent_N2 == null) Set("MIN_CL_predictionPercent_N2", "60");
                    if (strMIN_predictionPercent_Abs1_N2 == null) Set("MIN_predictionPercent_Abs1_N2", "60");
                    if (strMIN_CL_predictionPercent_Abs1_N2 == null) Set("MIN_CL_predictionPercent_Abs1_N2", "60");

                    if (strMIN_predictionPercent_N3 == null) Set("MIN_predictionPercent_N3", "60");
                    if (strMIN_CL_predictionPercent_N3 == null) Set("MIN_CL_predictionPercent_N3", "60");
                    if (strMIN_predictionPercent_Abs1_N3 == null) Set("MIN_predictionPercent_Abs1_N3", "60");
                    if (strMIN_CL_predictionPercent_Abs1_N3 == null) Set("MIN_CL_predictionPercent_Abs1_N3", "60");

                    if (strMIN_predictionPercent_N4 == null) Set("MIN_predictionPercent_N4", "60");
                    if (strMIN_CL_predictionPercent_N4 == null) Set("MIN_CL_predictionPercent_N4", "60");
                    if (strMIN_predictionPercent_Abs1_N4 == null) Set("MIN_predictionPercent_Abs1_N4", "60");
                    if (strMIN_CL_predictionPercent_Abs1_N4 == null) Set("MIN_CL_predictionPercent_Abs1_N4", "60");

                    if (strMIN_predictionPercent_N5 == null) Set("MIN_predictionPercent_N5", "60");
                    if (strMIN_CL_predictionPercent_N5 == null) Set("MIN_CL_predictionPercent_N5", "60");
                    if (strMIN_predictionPercent_Abs1_N5 == null) Set("MIN_predictionPercent_Abs1_N5", "60");
                    if (strMIN_CL_predictionPercent_Abs1_N5 == null) Set("MIN_CL_predictionPercent_Abs1_N5", "60");

                    if (strMIN_predictionPercent_N6 == null) Set("MIN_predictionPercent_N6", "60");
                    if (strMIN_CL_predictionPercent_N6 == null) Set("MIN_CL_predictionPercent_N6", "60");
                    if (strMIN_predictionPercent_Abs1_N6 == null) Set("MIN_predictionPercent_Abs1_N6", "60");
                    if (strMIN_CL_predictionPercent_Abs1_N6 == null) Set("MIN_CL_predictionPercent_Abs1_N6", "60");

                    if (strMIN_predictionPercent_N7 == null) Set("MIN_predictionPercent_N7", "60");
                    if (strMIN_CL_predictionPercent_N7 == null) Set("MIN_CL_predictionPercent_N7", "60");
                    if (strMIN_predictionPercent_Abs1_N7 == null) Set("MIN_predictionPercent_Abs1_N7", "60");
                    if (strMIN_CL_predictionPercent_Abs1_N7 == null) Set("MIN_CL_predictionPercent_Abs1_N7", "60");



                    MIN_predictionPercent = float.Parse(GetSetting("MIN_predictionPercent"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent = float.Parse(GetSetting("MIN_CL_predictionPercent"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_predictionPercent_Abs1 = float.Parse(GetSetting("MIN_predictionPercent_Abs1"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent_Abs1 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1"), System.Globalization.CultureInfo.InvariantCulture);

                    MIN_predictionPercent_N1 = float.Parse(GetSetting("MIN_predictionPercent_N1"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent_N1 = float.Parse(GetSetting("MIN_CL_predictionPercent_N1"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_predictionPercent_Abs1_N1 = float.Parse(GetSetting("MIN_predictionPercent_Abs1_N1"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent_Abs1_N1 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1_N1"), System.Globalization.CultureInfo.InvariantCulture);

                    MIN_predictionPercent_N2 = float.Parse(GetSetting("MIN_predictionPercent_N2"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent_N2 = float.Parse(GetSetting("MIN_CL_predictionPercent_N2"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_predictionPercent_Abs1_N2 = float.Parse(GetSetting("MIN_predictionPercent_Abs1_N2"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent_Abs1_N2 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1_N2"), System.Globalization.CultureInfo.InvariantCulture);

                    MIN_predictionPercent_N3 = float.Parse(GetSetting("MIN_predictionPercent_N3"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent_N3 = float.Parse(GetSetting("MIN_CL_predictionPercent_N3"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_predictionPercent_Abs1_N3 = float.Parse(GetSetting("MIN_predictionPercent_Abs1_N3"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent_Abs1_N3 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1_N3"), System.Globalization.CultureInfo.InvariantCulture);

                    MIN_predictionPercent_N4 = float.Parse(GetSetting("MIN_predictionPercent_N4"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent_N4 = float.Parse(GetSetting("MIN_CL_predictionPercent_N4"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_predictionPercent_Abs1_N4 = float.Parse(GetSetting("MIN_predictionPercent_Abs1_N4"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent_Abs1_N4 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1_N4"), System.Globalization.CultureInfo.InvariantCulture);

                    MIN_predictionPercent_N5 = float.Parse(GetSetting("MIN_predictionPercent_N5"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent_N5 = float.Parse(GetSetting("MIN_CL_predictionPercent_N5"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_predictionPercent_Abs1_N5 = float.Parse(GetSetting("MIN_predictionPercent_Abs1_N5"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent_Abs1_N5 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1_N5"), System.Globalization.CultureInfo.InvariantCulture);

                    MIN_predictionPercent_N6 = float.Parse(GetSetting("MIN_predictionPercent_N6"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent_N6 = float.Parse(GetSetting("MIN_CL_predictionPercent_N6"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_predictionPercent_Abs1_N6 = float.Parse(GetSetting("MIN_predictionPercent_Abs1_N6"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent_Abs1_N6 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1_N6"), System.Globalization.CultureInfo.InvariantCulture);

                    MIN_predictionPercent_N7 = float.Parse(GetSetting("MIN_predictionPercent_N7"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent_N7 = float.Parse(GetSetting("MIN_CL_predictionPercent_N7"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_predictionPercent_Abs1_N7 = float.Parse(GetSetting("MIN_predictionPercent_Abs1_N7"), System.Globalization.CultureInfo.InvariantCulture);
                    MIN_CL_predictionPercent_Abs1_N7 = float.Parse(GetSetting("MIN_CL_predictionPercent_Abs1_N7"), System.Globalization.CultureInfo.InvariantCulture);




                    //Set("MIN_predictionPercent", MIN_predictionPercent.ToString("R"));
                    //Set("MIN_CL_predictionPercent", MIN_CL_predictionPercent.ToString("R"));
                    //Set("MIN_predictionPercent_Abs1", MIN_predictionPercent_Abs1.ToString("R"));
                    //Set("MIN_CL_predictionPercent_Abs1", MIN_CL_predictionPercent_Abs1.ToString("R"));




                    if (predictionPercent < MIN_predictionPercent)
                    {
                        MIN_predictionPercent = predictionPercent;
                        SaveLoadNetwork(true, Config.MIN_predictionPercentFile.ToString());
                        Set("MIN_predictionPercent", MIN_predictionPercent.ToString("R"));  //Persist the value
                    }

                    if (CL_predictionPercent < MIN_CL_predictionPercent)
                    {
                        MIN_CL_predictionPercent = CL_predictionPercent;
                        SaveLoadNetwork(true, Config.MIN_CL_predictionPercentFile.ToString());
                        Set("MIN_CL_predictionPercent", MIN_CL_predictionPercent.ToString("R"));
                    }

                    if (predictionPercent_Abs1 < MIN_predictionPercent_Abs1)
                    {
                        MIN_predictionPercent_Abs1 = predictionPercent_Abs1;
                        SaveLoadNetwork(true, Config.MIN_predictionPercent_Abs1File.ToString());
                        Set("MIN_predictionPercent_Abs1", MIN_predictionPercent_Abs1.ToString("R"));
                    }

                    if (CL_predictionPercent_Abs1 < MIN_CL_predictionPercent_Abs1)
                    {
                        MIN_CL_predictionPercent_Abs1 = CL_predictionPercent_Abs1;
                        SaveLoadNetwork(true, Config.MIN_CL_predictionPercent_Abs1File.ToString());
                        Set("MIN_CL_predictionPercent_Abs1", MIN_CL_predictionPercent_Abs1.ToString("R"));
                    }


                    switch (LotoNumber)
                    {
                        case 1:
                            if (predictionPercent < MIN_predictionPercent_N1)
                            {
                                MIN_predictionPercent_N1 = predictionPercent;
                                SaveLoadNetwork(true, Config.MIN_predictionPercentFile_N1.ToString());
                                Set("MIN_predictionPercent_N1", MIN_predictionPercent_N1.ToString("R"));  //Persist the value
                            }

                            if (CL_predictionPercent < MIN_CL_predictionPercent_N1)
                            {
                                MIN_CL_predictionPercent_N1 = CL_predictionPercent;
                                SaveLoadNetwork(true, Config.MIN_CL_predictionPercentFile_N1.ToString());
                                Set("MIN_CL_predictionPercent_N1", MIN_CL_predictionPercent_N1.ToString("R"));
                            }

                            if (predictionPercent_Abs1 < MIN_predictionPercent_Abs1_N1)
                            {
                                MIN_predictionPercent_Abs1_N1 = predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MIN_predictionPercent_Abs1File_N1.ToString());
                                Set("MIN_predictionPercent_Abs1_N1", MIN_predictionPercent_Abs1_N1.ToString("R"));
                            }

                            if (CL_predictionPercent_Abs1 < MIN_CL_predictionPercent_Abs1_N1)
                            {
                                MIN_CL_predictionPercent_Abs1_N1 = CL_predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MIN_CL_predictionPercent_Abs1File_N1.ToString());
                                Set("MIN_CL_predictionPercent_Abs1_N1", MIN_CL_predictionPercent_Abs1_N1.ToString("R"));
                            }


                            break;

                        case 2:
                            if (predictionPercent < MIN_predictionPercent_N2)
                            {
                                MIN_predictionPercent_N2 = predictionPercent;
                                SaveLoadNetwork(true, Config.MIN_predictionPercentFile_N2.ToString());
                                Set("MIN_predictionPercent_N2", MIN_predictionPercent_N2.ToString("R"));  //Persist the value
                            }

                            if (CL_predictionPercent < MIN_CL_predictionPercent_N2)
                            {
                                MIN_CL_predictionPercent_N2 = CL_predictionPercent;
                                SaveLoadNetwork(true, Config.MIN_CL_predictionPercentFile_N2.ToString());
                                Set("MIN_CL_predictionPercent_N2", MIN_CL_predictionPercent_N2.ToString("R"));
                            }

                            if (predictionPercent_Abs1 < MIN_predictionPercent_Abs1_N2)
                            {
                                MIN_predictionPercent_Abs1_N2 = predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MIN_predictionPercent_Abs1File_N2.ToString());
                                Set("MIN_predictionPercent_Abs1_N2", MIN_predictionPercent_Abs1_N2.ToString("R"));
                            }

                            if (CL_predictionPercent_Abs1 < MIN_CL_predictionPercent_Abs1_N2)
                            {
                                MIN_CL_predictionPercent_Abs1_N2 = CL_predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MIN_CL_predictionPercent_Abs1File_N2.ToString());
                                Set("MIN_CL_predictionPercent_Abs1_N2", MIN_CL_predictionPercent_Abs1_N2.ToString("R"));
                            }


                            break;

                        case 3:
                            if (predictionPercent < MIN_predictionPercent_N3)
                            {
                                MIN_predictionPercent_N3 = predictionPercent;
                                SaveLoadNetwork(true, Config.MIN_predictionPercentFile_N3.ToString());
                                Set("MIN_predictionPercent_N3", MIN_predictionPercent_N3.ToString("R"));  //Persist the value
                            }

                            if (CL_predictionPercent < MIN_CL_predictionPercent_N3)
                            {
                                MIN_CL_predictionPercent_N3 = CL_predictionPercent;
                                SaveLoadNetwork(true, Config.MIN_CL_predictionPercentFile_N3.ToString());
                                Set("MIN_CL_predictionPercent_N3", MIN_CL_predictionPercent_N3.ToString("R"));
                            }

                            if (predictionPercent_Abs1 < MIN_predictionPercent_Abs1_N3)
                            {
                                MIN_predictionPercent_Abs1_N3 = predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MIN_predictionPercent_Abs1File_N3.ToString());
                                Set("MIN_predictionPercent_Abs1_N3", MIN_predictionPercent_Abs1_N3.ToString("R"));
                            }

                            if (CL_predictionPercent_Abs1 < MIN_CL_predictionPercent_Abs1_N3)
                            {
                                MIN_CL_predictionPercent_Abs1_N3 = CL_predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MIN_CL_predictionPercent_Abs1File_N3.ToString());
                                Set("MIN_CL_predictionPercent_Abs1_N3", MIN_CL_predictionPercent_Abs1_N3.ToString("R"));
                            }


                            break;

                        case 4:
                            if (predictionPercent < MIN_predictionPercent_N4)
                            {
                                MIN_predictionPercent_N4 = predictionPercent;
                                SaveLoadNetwork(true, Config.MIN_predictionPercentFile_N4.ToString());
                                Set("MIN_predictionPercent_N4", MIN_predictionPercent_N4.ToString("R"));  //Persist the value
                            }

                            if (CL_predictionPercent < MIN_CL_predictionPercent_N4)
                            {
                                MIN_CL_predictionPercent_N4 = CL_predictionPercent;
                                SaveLoadNetwork(true, Config.MIN_CL_predictionPercentFile_N4.ToString());
                                Set("MIN_CL_predictionPercent_N4", MIN_CL_predictionPercent_N4.ToString("R"));
                            }

                            if (predictionPercent_Abs1 < MIN_predictionPercent_Abs1_N4)
                            {
                                MIN_predictionPercent_Abs1_N4 = predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MIN_predictionPercent_Abs1File_N4.ToString());
                                Set("MIN_predictionPercent_Abs1_N4", MIN_predictionPercent_Abs1_N4.ToString("R"));
                            }

                            if (CL_predictionPercent_Abs1 < MIN_CL_predictionPercent_Abs1_N4)
                            {
                                MIN_CL_predictionPercent_Abs1_N4 = CL_predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MIN_CL_predictionPercent_Abs1File_N4.ToString());
                                Set("MIN_CL_predictionPercent_Abs1_N4", MIN_CL_predictionPercent_Abs1_N4.ToString("R"));
                            }

                            break;

                        case 5:
                            if (predictionPercent < MIN_predictionPercent_N5)
                            {
                                MIN_predictionPercent_N5 = predictionPercent;
                                SaveLoadNetwork(true, Config.MIN_predictionPercentFile_N5.ToString());
                                Set("MIN_predictionPercent_N5", MIN_predictionPercent_N5.ToString("R"));  //Persist the value
                            }

                            if (CL_predictionPercent < MIN_CL_predictionPercent_N5)
                            {
                                MIN_CL_predictionPercent_N5 = CL_predictionPercent;
                                SaveLoadNetwork(true, Config.MIN_CL_predictionPercentFile_N5.ToString());
                                Set("MIN_CL_predictionPercent_N5", MIN_CL_predictionPercent_N5.ToString("R"));
                            }

                            if (predictionPercent_Abs1 < MIN_predictionPercent_Abs1_N5)
                            {
                                MIN_predictionPercent_Abs1_N5 = predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MIN_predictionPercent_Abs1File_N5.ToString());
                                Set("MIN_predictionPercent_Abs1_N5", MIN_predictionPercent_Abs1_N5.ToString("R"));
                            }

                            if (CL_predictionPercent_Abs1 < MIN_CL_predictionPercent_Abs1_N5)
                            {
                                MIN_CL_predictionPercent_Abs1_N5 = CL_predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MIN_CL_predictionPercent_Abs1File_N5.ToString());
                                Set("MIN_CL_predictionPercent_Abs1_N5", MIN_CL_predictionPercent_Abs1_N5.ToString("R"));
                            }

                            break;

                        case 6:
                            if (predictionPercent < MIN_predictionPercent_N6)
                            {
                                MIN_predictionPercent_N6 = predictionPercent;
                                SaveLoadNetwork(true, Config.MIN_predictionPercentFile_N6.ToString());
                                Set("MIN_predictionPercent_N6", MIN_predictionPercent_N6.ToString("R"));  //Persist the value
                            }

                            if (CL_predictionPercent < MIN_CL_predictionPercent_N6)
                            {
                                MIN_CL_predictionPercent_N6 = CL_predictionPercent;
                                SaveLoadNetwork(true, Config.MIN_CL_predictionPercentFile_N6.ToString());
                                Set("MIN_CL_predictionPercent_N6", MIN_CL_predictionPercent_N6.ToString("R"));
                            }

                            if (predictionPercent_Abs1 < MIN_predictionPercent_Abs1_N6)
                            {
                                MIN_predictionPercent_Abs1_N6 = predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MIN_predictionPercent_Abs1File_N6.ToString());
                                Set("MIN_predictionPercent_Abs1_N6", MIN_predictionPercent_Abs1_N6.ToString("R"));
                            }

                            if (CL_predictionPercent_Abs1 < MIN_CL_predictionPercent_Abs1_N6)
                            {
                                MIN_CL_predictionPercent_Abs1_N6 = CL_predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MIN_CL_predictionPercent_Abs1File_N6.ToString());
                                Set("MIN_CL_predictionPercent_Abs1_N6", MIN_CL_predictionPercent_Abs1_N6.ToString("R"));
                            }

                            break;

                        case 7:
                            if (predictionPercent < MIN_predictionPercent_N7)
                            {
                                MIN_predictionPercent_N7 = predictionPercent;
                                SaveLoadNetwork(true, Config.MIN_predictionPercentFile_N7.ToString());
                                Set("MIN_predictionPercent_N7", MIN_predictionPercent_N7.ToString("R"));  //Persist the value
                            }

                            if (CL_predictionPercent < MIN_CL_predictionPercent_N7)
                            {
                                MIN_CL_predictionPercent_N7 = CL_predictionPercent;
                                SaveLoadNetwork(true, Config.MIN_CL_predictionPercentFile_N7.ToString());
                                Set("MIN_CL_predictionPercent_N7", MIN_CL_predictionPercent_N7.ToString("R"));
                            }

                            if (predictionPercent_Abs1 < MIN_predictionPercent_Abs1_N7)
                            {
                                MIN_predictionPercent_Abs1_N7 = predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MIN_predictionPercent_Abs1File_N7.ToString());
                                Set("MIN_predictionPercent_Abs1_N7", MIN_predictionPercent_Abs1_N7.ToString("R"));
                            }

                            if (CL_predictionPercent_Abs1 < MIN_CL_predictionPercent_Abs1_N7)
                            {
                                MIN_CL_predictionPercent_Abs1_N7 = CL_predictionPercent_Abs1;
                                SaveLoadNetwork(true, Config.MIN_CL_predictionPercent_Abs1File_N7.ToString());
                                Set("MIN_CL_predictionPercent_Abs1_N7", MIN_CL_predictionPercent_Abs1_N7.ToString("R"));
                            }

                            break;

                        default:
                            break;
                    }

                //Min Handling
            }
        }



        private void PredictNetworkNewWithDayOfWeek()
        {
            IMLData output;
            int evaluateStop = data.Select(t => t.Id).Max();

            using (var file = new System.IO.StreamWriter(Config.PredictResult.ToString(), true))
            {   
                //Start new
                int currentId = evaluateStop + 1;

                    if (LotoNumber != 7)
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

                        output = network.Compute(input);
                    }
                    else
                    {
                        //Calculate based on actual data
                        var input = new BasicMLData(PastWindowSize * 2);

                        for (int i = 0; i < PastWindowSize; i++)
                        {
                            input[i * 2] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                             .Select(t => t.NormalizeDayOfWeek).First();

                            input[i * 2 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizedActual7).First();
                        }

                        output = network.Compute(input);
                    }

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

                        case 7:
                            predicted = Math.Round(norm7.Stats.DeNormalize(normalizedPredicted), 0);
                            break;

                        default:
                            break;
                    }

                int DrawNumber = data.Where(t => t.Id == (currentId - 1)).Select(t => t.DrawNumber).First() + 1;

                string consoleLine = string.Format(@"DrawNumber:{0}; LotoNumber:{1}; PastWindowSize ={2}; MaxError:{3}; Predicted:{4}; Prediction percent:{5:0.00}%; _closedLoop_Prediction percent ={6:0.00}%",
                                       DrawNumber, LotoNumber, PastWindowSize, MaxError, predicted, predictionPercent, CL_predictionPercent);

                string Line = string.Format(@"{0}; {1}; {2}; {3}; {4}; {5:0.00}%; {6:0.00}%",
                               DrawNumber, LotoNumber, PastWindowSize, MaxError, predicted, predictionPercent, CL_predictionPercent);

                file.WriteLine(Line);
                Console.WriteLine(consoleLine);
            }
        }

        private void PredictNetworkNew()
        {
            IMLData output;
            IMLData _closedLoop_output;
            int evaluateStop = data.Select(t => t.Id).Max();
            int currentId = evaluateStop + 1;


            // if the file does not exist or empty, print an header.
            if (!File.Exists(Config.PredictResult.ToString()) || Config.PredictResult.Length == 0)
                using (System.IO.StreamWriter file = File.AppendText(Config.PredictResult.ToString()))
                {
                    file.WriteLine(PredictFileHeader);
                    file.Close();
                }

            using (System.IO.StreamWriter file = File.AppendText(Config.PredictResult.ToString()))
            {
                if (LotoNumber != 7)
                {
                    //Calculate based on actual data
                    var input = new BasicMLData(PastWindowSize * 7);
                    var _closedLoop_input = new BasicMLData(PastWindowSize * 7);

                    for (int i = 0; i < PastWindowSize; i++)
                    {
                            input[i * 7] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizedActual1).First();

                            input[i * 7 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizedActual2).First();

                            input[i * 7 + 2] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizedActual3).First();

                            input[i * 7 + 3] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizedActual4).First();

                            input[i * 7 + 4] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizedActual5).First();

                            input[i * 7 + 5] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t.NormalizedActual6).First();


                            input[i * 7 + 6] = (LotoNumber == 1) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual1).First() :
                                                (LotoNumber == 2) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual2).First() :
                                                (LotoNumber == 3) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual3).First() :
                                                (LotoNumber == 4) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual4).First() :
                                                (LotoNumber == 5) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual5).First() :
                                                data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.NormalizedNA_Actual6).First();



                            _closedLoop_input[i * 7] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t._closedLoopNormalizedActual1).First();

                            _closedLoop_input[i * 7 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t._closedLoopNormalizedActual2).First();

                            _closedLoop_input[i * 7 + 2] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t._closedLoopNormalizedActual3).First();

                            _closedLoop_input[i * 7 + 3] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t._closedLoopNormalizedActual4).First();

                            _closedLoop_input[i * 7 + 4] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t._closedLoopNormalizedActual5).First();

                            _closedLoop_input[i * 7 + 5] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                .Select(t => t._closedLoopNormalizedActual6).First();


                            _closedLoop_input[i * 7 + 6] =
                                           (LotoNumber == 1) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.Normalized_closedLoopNA_Actual1).First() :
                                           (LotoNumber == 2) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.Normalized_closedLoopNA_Actual2).First() :
                                           (LotoNumber == 3) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.Normalized_closedLoopNA_Actual3).First() :
                                           (LotoNumber == 4) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.Normalized_closedLoopNA_Actual4).First() :
                                           (LotoNumber == 5) ? data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.Normalized_closedLoopNA_Actual5).First() :
                                           data.Where(t => t.Id == ((currentId - PastWindowSize + i))).Select(t => t.Normalized_closedLoopNA_Actual6).First();
                    }

                    output = network.Compute(input);
                    _closedLoop_output = network.Compute(_closedLoop_input);
                }
                else
                {
                    //Calculate based on actual data
                    var input = new BasicMLData(PastWindowSize * 2);
                    var _closedLoop_input = new BasicMLData(PastWindowSize * 2);

                    for (int i = 0; i < PastWindowSize; i++)
                    {
                        input[i * 2] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t.NormalizedActual7).First();

                        input[i * 2 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t.NormalizedNA_Actual7).First();

                        _closedLoop_input[i * 2] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t._closedLoopNormalizedActual7).First();

                        _closedLoop_input[i * 2 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                                  .Select(t => t.Normalized_closedLoopNA_Actual7).First();
                    }

                    output = network.Compute(input);
                    _closedLoop_output = network.Compute(_closedLoop_input);
                }

                double normalizedPredicted = output[0];
                double _closedLoop_normalizedPredicted = _closedLoop_output[0];
                predicted = 0.0;
                _closedLoop_predicted = 0.0;
                
                switch (LotoNumber)
                {
                    case 1:
                        predicted = Math.Round(norm1.Stats.DeNormalize(normalizedPredicted), 0);
                        _closedLoop_predicted = Math.Round(norm1.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                        break;

                    case 2:
                        predicted = Math.Round(norm2.Stats.DeNormalize(normalizedPredicted), 0);
                        _closedLoop_predicted = Math.Round(norm2.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                        break;

                    case 3:
                        predicted = Math.Round(norm3.Stats.DeNormalize(normalizedPredicted), 0);
                        _closedLoop_predicted = Math.Round(norm3.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                        break;

                    case 4:
                        predicted = Math.Round(norm4.Stats.DeNormalize(normalizedPredicted), 0);
                        _closedLoop_predicted = Math.Round(norm4.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                        break;

                    case 5:
                        predicted = Math.Round(norm5.Stats.DeNormalize(normalizedPredicted), 0);
                        _closedLoop_predicted = Math.Round(norm5.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                        break;

                    case 6:
                        predicted = Math.Round(norm6.Stats.DeNormalize(normalizedPredicted), 0);
                        _closedLoop_predicted = Math.Round(norm6.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                        break;

                    case 7:
                        predicted = Math.Round(norm7.Stats.DeNormalize(normalizedPredicted), 0);
                        _closedLoop_predicted = Math.Round(norm7.Stats.DeNormalize(_closedLoop_normalizedPredicted), 0);
                        break;

                    default:
                        break;
                }

                int DrawNumber = data.Where(t => t.Id == (currentId - 1)).Select(t => t.DrawNumber).First() + 1;

                string consoleLine = string.Format(@"DrawNumber:{0}; LotoNumber:{1}; PastWindowSize ={2}; MaxError:{3}; Predicted:{4}; CL_Predicted:{5}; Prediction percent:{6:0.00}%; _closedLoop_Prediction percent ={7:0.00}%",
                                       DrawNumber, LotoNumber, PastWindowSize, MaxError, predicted, _closedLoop_predicted, predictionPercent, CL_predictionPercent);

                string Line = string.Format(@"{0}; {1}; {2}; {3}; {4}; {5}; {6:0.00}%; {7:0.00}%",
                               DrawNumber, LotoNumber, PastWindowSize, MaxError, predicted, _closedLoop_predicted, predictionPercent, CL_predictionPercent);

                file.WriteLine(Line);
                file.Close();
                //Console.WriteLine(consoleLine);
            }
        }

        private void PredictNetwork()
        {
            IMLData output;
            int evaluateStop = data.Select(t => t.Id).Max();

            /*
                public string EvaluateFileHeader
                public string PredictFileHeader 

             */


            // if the file does not exist or empty, print an header.
            if (!File.Exists(Config.PredictResult.ToString()) || Config.PredictResult.Length == 0)
                using (System.IO.StreamWriter file = File.AppendText(Config.PredictResult.ToString()))
                {
                    file.WriteLine(PredictFileHeader);
                    file.Close();
                }

            using (var file = new System.IO.StreamWriter(Config.PredictResult.ToString(), true))
            {
                //Start new
                int currentId = evaluateStop + 1;

                if (LotoNumber != 7)
                {
                    //Calculate based on actual data
                    var input = new BasicMLData(PastWindowSize * 6);

                    for (int i = 0; i < PastWindowSize; i++)
                    {
                        input[i * 6] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t.NormalizedActual1).First();

                        input[i * 6 + 1] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t.NormalizedActual2).First();

                        input[i * 6 + 2] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t.NormalizedActual3).First();

                        input[i * 6 + 3] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t.NormalizedActual4).First();

                        input[i * 6 + 4] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t.NormalizedActual5).First();

                        input[i * 6 + 5] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t.NormalizedActual6).First();
                    }

                    output = network.Compute(input);
                }
                else
                {
                    //Calculate based on actual data
                    var input = new BasicMLData(PastWindowSize);
                    for (int i = 0; i < PastWindowSize; i++)
                    {
                        input[i] = data.Where(t => t.Id == ((currentId - PastWindowSize + i)))
                            .Select(t => t.NormalizedActual7).First();
                    }

                    output = network.Compute(input);
                }
                //Start new - end

                
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

                    case 7:
                        predicted = Math.Round(norm7.Stats.DeNormalize(normalizedPredicted), 0);
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
                file.Close();
                Console.WriteLine(consoleLine);
            }
        }
    }
}