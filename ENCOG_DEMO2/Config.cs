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

        public static FileInfo MAX_predictionPercentFile_N1 = FileUtil.CombinePath(AppPath, "MAX_predictionPercentFile_N1.dat");
        public static FileInfo MAX_CL_predictionPercentFile_N1 = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercentFile_N1.dat");
        public static FileInfo MAX_predictionPercent_Abs1File_N1 = FileUtil.CombinePath(AppPath, "MAX_predictionPercent_Abs1File_N1.dat");
        public static FileInfo MAX_CL_predictionPercent_Abs1File_N1 = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercent_Abs1File_N1.dat");

        public static FileInfo MAX_predictionPercentFile_N2 = FileUtil.CombinePath(AppPath, "MAX_predictionPercentFile_N2.dat");
        public static FileInfo MAX_CL_predictionPercentFile_N2 = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercentFile_N2.dat");
        public static FileInfo MAX_predictionPercent_Abs1File_N2 = FileUtil.CombinePath(AppPath, "MAX_predictionPercent_Abs1File_N2.dat");
        public static FileInfo MAX_CL_predictionPercent_Abs1File_N2 = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercent_Abs1File_N2.dat");

        public static FileInfo MAX_predictionPercentFile_N3 = FileUtil.CombinePath(AppPath, "MAX_predictionPercentFile_N3.dat");
        public static FileInfo MAX_CL_predictionPercentFile_N3 = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercentFile_N3.dat");
        public static FileInfo MAX_predictionPercent_Abs1File_N3 = FileUtil.CombinePath(AppPath, "MAX_predictionPercent_Abs1File_N3.dat");
        public static FileInfo MAX_CL_predictionPercent_Abs1File_N3 = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercent_Abs1File_N3.dat");

        public static FileInfo MAX_predictionPercentFile_N4 = FileUtil.CombinePath(AppPath, "MAX_predictionPercentFile_N4.dat");
        public static FileInfo MAX_CL_predictionPercentFile_N4 = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercentFile_N4.dat");
        public static FileInfo MAX_predictionPercent_Abs1File_N4 = FileUtil.CombinePath(AppPath, "MAX_predictionPercent_Abs1File_N4.dat");
        public static FileInfo MAX_CL_predictionPercent_Abs1File_N4 = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercent_Abs1File_N4.dat");

        public static FileInfo MAX_predictionPercentFile_N5 = FileUtil.CombinePath(AppPath, "MAX_predictionPercentFile_N5.dat");
        public static FileInfo MAX_CL_predictionPercentFile_N5 = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercentFile_N5.dat");
        public static FileInfo MAX_predictionPercent_Abs1File_N5 = FileUtil.CombinePath(AppPath, "MAX_predictionPercent_Abs1File_N5.dat");
        public static FileInfo MAX_CL_predictionPercent_Abs1File_N5 = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercent_Abs1File_N5.dat");

        public static FileInfo MAX_predictionPercentFile_N6 = FileUtil.CombinePath(AppPath, "MAX_predictionPercentFile_N6.dat");
        public static FileInfo MAX_CL_predictionPercentFile_N6 = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercentFile_N6.dat");
        public static FileInfo MAX_predictionPercent_Abs1File_N6 = FileUtil.CombinePath(AppPath, "MAX_predictionPercent_Abs1File_N6.dat");
        public static FileInfo MAX_CL_predictionPercent_Abs1File_N6 = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercent_Abs1File_N6.dat");

        public static FileInfo MAX_predictionPercentFile_N7 = FileUtil.CombinePath(AppPath, "MAX_predictionPercentFile_N7.dat");
        public static FileInfo MAX_CL_predictionPercentFile_N7 = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercentFile_N7.dat");
        public static FileInfo MAX_predictionPercent_Abs1File_N7 = FileUtil.CombinePath(AppPath, "MAX_predictionPercent_Abs1File_N7.dat");
        public static FileInfo MAX_CL_predictionPercent_Abs1File_N7 = FileUtil.CombinePath(AppPath, "MAX_CL_predictionPercent_Abs1File_N7.dat");


        //MIN Handling
        public static FileInfo MIN_predictionPercentFile = FileUtil.CombinePath(AppPath, "MIN_predictionPercentFile.dat");
        public static FileInfo MIN_CL_predictionPercentFile = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercentFile.dat");
        public static FileInfo MIN_predictionPercent_Abs1File = FileUtil.CombinePath(AppPath, "MIN_predictionPercent_Abs1File.dat");
        public static FileInfo MIN_CL_predictionPercent_Abs1File = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercent_Abs1File.dat");

        public static FileInfo MIN_predictionPercentFile_N1 = FileUtil.CombinePath(AppPath, "MIN_predictionPercentFile_N1.dat");
        public static FileInfo MIN_CL_predictionPercentFile_N1 = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercentFile_N1.dat");
        public static FileInfo MIN_predictionPercent_Abs1File_N1 = FileUtil.CombinePath(AppPath, "MIN_predictionPercent_Abs1File_N1.dat");
        public static FileInfo MIN_CL_predictionPercent_Abs1File_N1 = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercent_Abs1File_N1.dat");

        public static FileInfo MIN_predictionPercentFile_N2 = FileUtil.CombinePath(AppPath, "MIN_predictionPercentFile_N2.dat");
        public static FileInfo MIN_CL_predictionPercentFile_N2 = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercentFile_N2.dat");
        public static FileInfo MIN_predictionPercent_Abs1File_N2 = FileUtil.CombinePath(AppPath, "MIN_predictionPercent_Abs1File_N2.dat");
        public static FileInfo MIN_CL_predictionPercent_Abs1File_N2 = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercent_Abs1File_N2.dat");

        public static FileInfo MIN_predictionPercentFile_N3 = FileUtil.CombinePath(AppPath, "MIN_predictionPercentFile_N3.dat");
        public static FileInfo MIN_CL_predictionPercentFile_N3 = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercentFile_N3.dat");
        public static FileInfo MIN_predictionPercent_Abs1File_N3 = FileUtil.CombinePath(AppPath, "MIN_predictionPercent_Abs1File_N3.dat");
        public static FileInfo MIN_CL_predictionPercent_Abs1File_N3 = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercent_Abs1File_N3.dat");

        public static FileInfo MIN_predictionPercentFile_N4 = FileUtil.CombinePath(AppPath, "MIN_predictionPercentFile_N4.dat");
        public static FileInfo MIN_CL_predictionPercentFile_N4 = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercentFile_N4.dat");
        public static FileInfo MIN_predictionPercent_Abs1File_N4 = FileUtil.CombinePath(AppPath, "MIN_predictionPercent_Abs1File_N4.dat");
        public static FileInfo MIN_CL_predictionPercent_Abs1File_N4 = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercent_Abs1File_N4.dat");

        public static FileInfo MIN_predictionPercentFile_N5 = FileUtil.CombinePath(AppPath, "MIN_predictionPercentFile_N5.dat");
        public static FileInfo MIN_CL_predictionPercentFile_N5 = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercentFile_N5.dat");
        public static FileInfo MIN_predictionPercent_Abs1File_N5 = FileUtil.CombinePath(AppPath, "MIN_predictionPercent_Abs1File_N5.dat");
        public static FileInfo MIN_CL_predictionPercent_Abs1File_N5 = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercent_Abs1File_N5.dat");

        public static FileInfo MIN_predictionPercentFile_N6 = FileUtil.CombinePath(AppPath, "MIN_predictionPercentFile_N6.dat");
        public static FileInfo MIN_CL_predictionPercentFile_N6 = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercentFile_N6.dat");
        public static FileInfo MIN_predictionPercent_Abs1File_N6 = FileUtil.CombinePath(AppPath, "MIN_predictionPercent_Abs1File_N6.dat");
        public static FileInfo MIN_CL_predictionPercent_Abs1File_N6 = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercent_Abs1File_N6.dat");

        public static FileInfo MIN_predictionPercentFile_N7 = FileUtil.CombinePath(AppPath, "MIN_predictionPercentFile_N7.dat");
        public static FileInfo MIN_CL_predictionPercentFile_N7 = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercentFile_N7.dat");
        public static FileInfo MIN_predictionPercent_Abs1File_N7 = FileUtil.CombinePath(AppPath, "MIN_predictionPercent_Abs1File_N7.dat");
        public static FileInfo MIN_CL_predictionPercent_Abs1File_N7 = FileUtil.CombinePath(AppPath, "MIN_CL_predictionPercent_Abs1File_N7.dat");

        //MIN Handling


        //Config.MAX_predictionPercentFile.ToString()
        //Config.MAX_CL_predictionPercentFile.ToString()
        //Config.MAX_predictionPercent_Abs1File.ToString()
        //Config.MAX_CL_predictionPercent_Abs1File.ToString()
        #region Evaluation



        #endregion Evaluation
    }
}