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
using Encog.MathUtil;
using System;

namespace Encog.Engine.Network.Activation
{
    /// <summary>
    /// The softmax activation function.
    /// </summary>
    [Serializable]
    public class ActivationSoftMax : IActivationFunction
    {
        /// <summary>
        /// The parameters.
        /// </summary>
        ///
        private readonly double[] _paras;

        /// <summary>
        /// Construct the soft-max activation function.
        /// </summary>
        ///
        public ActivationSoftMax()
        {
            _paras = new double[0];
        }

        /// <summary>
        /// Clone the object.
        /// </summary>
        /// <returns>The cloned object.</returns>
        public object Clone()
        {
            return new ActivationSoftMax();
        }

        /// <returns>Return false, softmax has no derivative.</returns>
        public virtual bool HasDerivative
        {
            get
            {
                return true;
            }
        }

        /// <inheritdoc />
        public virtual void ActivationFunction(double[] x, int start,
                                               int size)
        {
            double sum = 0;
            for (int i = start; i < start + size; i++)
            {
                x[i] = BoundMath.Exp(x[i]);
                sum += x[i];
            }
            for (int i = start; i < start + size; i++)
            {
                x[i] = x[i] / sum;
            }
        }

        /// <inheritdoc />
        public virtual double DerivativeFunction(double b, double a)
        {
            return 1.0d;
        }

        /// <inheritdoc />
        public virtual String[] ParamNames
        {
            get
            {
                String[] result = { };
                return result;
            }
        }

        /// <inheritdoc />
        public virtual double[] Params
        {
            get { return _paras; }
        }
    }
}