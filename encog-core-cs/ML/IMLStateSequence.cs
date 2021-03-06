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
using Encog.ML.Data;

namespace Encog.ML
{
    /// <summary>
    /// A state sequence ML method, for example a Hidden Markov Model.
    /// </summary>
    public interface IMLStateSequence
    {
        /// <summary>
        /// Get the sates for the given sequence.
        /// </summary>
        /// <param name="oseq">The sequence.</param>
        /// <returns>The states.</returns>
        int[] GetStatesForSequence(IMLDataSet oseq);

        /// <summary>
        /// Determine the probability of the specified sequence.
        /// </summary>
        /// <param name="oseq">The sequence.</param>
        /// <returns>The probability.</returns>
        double Probability(IMLDataSet oseq);

        /// <summary>
        /// Determine the probability for the specified sequence and states.
        /// </summary>
        /// <param name="seq">The sequence.</param>
        /// <param name="states">The states.</param>
        /// <returns>The probability.</returns>
        double Probability(IMLDataSet seq, int[] states);
    }
}