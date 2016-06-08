using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encog.ML.Data;
using Encog.Util.File;



namespace LotoPrediction
{
    public static class Config
    {
        public static FileInfo BasePath = new FileInfo(@"C:\Projects\ENCOG\L-Archieve\");
        public static FileInfo BaseFile = FileUtil.CombinePath(BasePath, "L-Archieve.csv");

        #region Evaluation

        public static FileInfo EvaluationResult = FileUtil.CombinePath(BasePath, "LotoData_Evaluate.csv");
        public static FileInfo PredictResult = FileUtil.CombinePath(BasePath, "LotoData_Predict.csv");

        #endregion
    }
}
