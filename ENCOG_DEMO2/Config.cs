using Encog.Util.File;
using System.IO;

namespace LotoPrediction
{
    public static class Config
    {
        //public static FileInfo BasePath = new FileInfo(@"..\..\Archieve\");
        public static FileInfo BasePath = new FileInfo(@"C:\Projects\ENCOG\L-Archieve\");
        public static FileInfo BaseFile = FileUtil.CombinePath(BasePath, "L-Archieve.csv");
        public static FileInfo MAX_predictionPercentFile = FileUtil.CombinePath(BasePath, "MAX_predictionPercentFile.dat");
        public static FileInfo MAX_CL_predictionPercentFile = FileUtil.CombinePath(BasePath, "MAX_CL_predictionPercentFile.dat");
        public static FileInfo MAX_predictionPercent_Abs1File = FileUtil.CombinePath(BasePath, "MAX_predictionPercent_Abs1File.dat");
        public static FileInfo MAX_CL_predictionPercent_Abs1File = FileUtil.CombinePath(BasePath, "MAX_CL_predictionPercent_Abs1File.dat");


        //Config.MAX_predictionPercentFile.ToString()
        //Config.MAX_CL_predictionPercentFile.ToString()
        //Config.MAX_predictionPercent_Abs1File.ToString()
        //Config.MAX_CL_predictionPercent_Abs1File.ToString()
        #region Evaluation

        public static FileInfo EvaluationResult = FileUtil.CombinePath(BasePath, "LotoData_Evaluate.csv");
        public static FileInfo PredictResult = FileUtil.CombinePath(BasePath, "LotoData_Predict.csv");

        #endregion Evaluation
    }
}