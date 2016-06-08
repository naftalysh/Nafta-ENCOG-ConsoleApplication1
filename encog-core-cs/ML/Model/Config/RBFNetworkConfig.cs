//
// Encog(tm) Core v3.3 - .Net Version
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
using Encog.ML.Data.Versatile;
using Encog.ML.Data.Versatile.Columns;
using Encog.ML.Data.Versatile.Normalizers;
using Encog.ML.Data.Versatile.Normalizers.Strategy;
using Encog.ML.Factory;
using System;
using System.Text;

namespace Encog.ML.Model.Config
{
    /// <summary>
    ///     Config class for EncogModel to use a RBF neural network.
    /// </summary>
    public class RBFNetworkConfig : IMethodConfig
    {
        /// <inheritdoc />
        public String MethodName
        {
            get { return MLMethodFactory.TypeRbfnetwork; }
        }

        /// <inheritdoc />
        public String SuggestModelArchitecture(VersatileMLDataSet dataset)
        {
            int inputColumns = dataset.NormHelper.InputColumns.Count;
            int outputColumns = dataset.NormHelper.OutputColumns.Count;
            var hiddenCount = (int)((inputColumns + outputColumns) * 1.5);
            var result = new StringBuilder();

            result.Append("?->gaussian(c=");
            result.Append(hiddenCount);
            result.Append(")->?");
            return result.ToString();
        }

        /// <inheritdoc />
        public INormalizationStrategy SuggestNormalizationStrategy(VersatileMLDataSet dataset, string architecture)
        {
            int outputColumns = dataset.NormHelper.OutputColumns.Count;

            ColumnType ct = dataset.NormHelper.OutputColumns[0].DataType;

            var result = new BasicNormalizationStrategy();
            result.AssignInputNormalizer(ColumnType.Continuous, new RangeNormalizer(0, 1));
            result.AssignInputNormalizer(ColumnType.Nominal, new OneOfNNormalizer(0, 1));
            result.AssignInputNormalizer(ColumnType.Ordinal, new OneOfNNormalizer(0, 1));

            result.AssignOutputNormalizer(ColumnType.Continuous, new RangeNormalizer(0, 1));
            result.AssignOutputNormalizer(ColumnType.Nominal, new OneOfNNormalizer(0, 1));
            result.AssignOutputNormalizer(ColumnType.Ordinal, new OneOfNNormalizer(0, 1));
            return result;
        }

        /// <inheritdoc />
        public String SuggestTrainingType()
        {
            return "rprop";
        }

        /// <inheritdoc />
        public String SuggestTrainingArgs(String trainingType)
        {
            return "";
        }

        /// <inheritdoc />
        public int DetermineOutputCount(VersatileMLDataSet dataset)
        {
            return dataset.NormHelper.CalculateNormalizedOutputCount();
        }
    }
}