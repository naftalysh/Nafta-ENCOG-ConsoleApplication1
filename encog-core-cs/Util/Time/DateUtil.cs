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
using System;

namespace Encog.Util.Time
{
    internal class DateUtil
    {
        /// <summary>
        /// January is 1.
        /// </summary>
        /// <param name="month"></param>
        /// <param name="day"></param>
        /// <param name="year"></param>
        /// <returns></returns>
        public static DateTime CreateDate(int month, int day, int year)
        {
            var result = new DateTime(year, month, day);
            return result;
        }

        /// <summary>
        /// Truncate a date, remove the time.
        /// </summary>
        /// <param name="date">The date to truncate.</param>
        /// <returns>The date without the time.</returns>
        public static DateTime TruncateDate(DateTime date)
        {
            return new DateTime(date.Year, date.Month, date.Day);
        }
    }
}