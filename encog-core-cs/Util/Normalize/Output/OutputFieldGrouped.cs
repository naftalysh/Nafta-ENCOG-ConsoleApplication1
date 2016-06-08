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
using Encog.Util.Normalize.Input;
using System;

namespace Encog.Util.Normalize.Output
{
    /// <summary>
    ///  Defines an output field that can be grouped.  Groupable classes
    /// will extend this class.
    /// </summary>
    [Serializable]
    public abstract class OutputFieldGrouped : BasicOutputField
    {
        /// <summary>
        /// The group that this field is a member of.
        /// </summary>
        private readonly IOutputFieldGroup _group;

        /// <summary>
        /// The source field, this is the input field that provides data
        /// for this output field.
        /// </summary>
        private readonly IInputField _sourceField;

        /// <summary>
        /// Default constructor, used mainly for reflection.
        /// </summary>
        protected OutputFieldGrouped()
        {
        }

        /// <summary>
        /// Construct a grouped output field.
        /// </summary>
        /// <param name="group">The group that this field belongs to.</param>
        /// <param name="sourceField">The source field for this output field.</param>
        protected OutputFieldGrouped(IOutputFieldGroup group,
                                  IInputField sourceField)
        {
            _group = group;
            _sourceField = sourceField;
            _group.GroupedFields.Add(this);
        }

        /// <summary>
        /// The group that this field belongs to.
        /// </summary>
        /// <returns></returns>
        public IOutputFieldGroup Group
        {
            get { return _group; }
        }

        /// <summary>
        /// The source field for this output field.
        /// </summary>
        /// <returns></returns>
        public IInputField SourceField
        {
            get { return _sourceField; }
        }

        /// <summary>
        /// The numebr of fields that will actually be generated by
        /// this field. For a simple field, this value is 1.
        /// </summary>
        public abstract override int SubfieldCount { get; }

        /// <summary>
        /// Init this field for a new row.
        /// </summary>
        public abstract override void RowInit();

        /// <summary>
        /// Calculate the value for this field.  Specify subfield of zero
        /// if this is a simple field.
        /// </summary>
        /// <param name="subfield"> The subfield index.</param>
        /// <returns>The calculated value for this field.</returns>
        public abstract override double Calculate(int subfield);
    }
}