using Encog.Util.File;
using System;
using System.IO;

namespace LotoPrediction
{
    public static class Config
    {
        public static FileInfo AppPath = new FileInfo(AppDomain.CurrentDomain.BaseDirectory); 
        public static FileInfo BaseArchievePath = new FileInfo(@"C:\Projects\ENCOG\L-Archieve\");
        public static FileInfo BaseArchieveFile = FileUtil.CombinePath(AppPath, "L-Archieve.csv");
        public static FileInfo EvaluationResult = FileUtil.CombinePath(AppPath, "LotoData_Evaluate.csv");
        public static FileInfo PredictResult = FileUtil.CombinePath(AppPath, "LotoData_Predict.csv");

        public static FileInfo MAX_predictionPercentFile = FileUtil.CombinePath(AppPath, "MAX_predictionPercentFile.dat");
        public static FileInfo MAX_CL_predictionPercentFile = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercentFile.dat");
        public static FileInfo MAX_predictionPercent_Abs1File = FileUtil.CombinePath(AppPath, "MAX_predictionPercent_Abs1File.dat");
        public static FileInfo MAX_CL_predictionPercent_Abs1File = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercent_Abs1File.dat");


        //Config.MAX_predictionPercentFile.ToString()
        //Config.MAX_CL_predictionPercentFile.ToString()
        //Config.MAX_predictionPercent_Abs1File.ToString()
        //Config.MAX_CL_predictionPercent_Abs1File.ToString()
        #region Evaluation

        

        #endregion Evaluation
    }
}