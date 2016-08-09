using Encog.Util.File;
using System.IO;

namespace LotoPrediction
{
    public static class Config
    {
        public static FileInfo BasePath = new FileInfo(@"..\Archieve\");
        public static FileInfo BaseFile = FileUtil.CombinePath(BasePath, "L-Archieve.csv");

        #region Evaluation

        public static FileInfo EvaluationResult = FileUtil.CombinePath(BasePath, "LotoData_Evaluate.csv");
        public static FileInfo PredictResult = FileUtil.CombinePath(BasePath, "LotoData_Predict.csv");

        #endregion Evaluation
    }
}