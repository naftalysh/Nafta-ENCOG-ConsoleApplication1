using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LotoPrediction
{
    public class NumbersAnalysis
    {
        private Dictionary<Int32, Int32> _NumbersDic37_Total = new Dictionary<Int32, Int32>();
        private Dictionary<Int32, Int32> _NumbersDic7_Total = new Dictionary<Int32, Int32>();
        private Dictionary<Int32, Int32> _closedLoopNumbersDic37_Total = new Dictionary<Int32, Int32>();
        private Dictionary<Int32, Int32> _closedLoopNumbersDic7_Total = new Dictionary<Int32, Int32>();
        private Dictionary<Int32, Int32> _NumbersDic37_PastWindow = new Dictionary<Int32, Int32>();
        private Dictionary<Int32, Int32> _NumbersDic7_PastWindow = new Dictionary<Int32, Int32>();


        public IDictionary<Int32, Int32> NumbersDic37_Total { get { return _NumbersDic37_Total; } }
        public IDictionary<Int32, Int32> NumbersDic7_Total { get { return _NumbersDic7_Total; } }
        public IDictionary<Int32, Int32> closedLoopNumbersDic37_Total { get { return _NumbersDic37_Total; } }
        public IDictionary<Int32, Int32> closedLoopNumbersDic7_Total { get { return _NumbersDic7_Total; } }

        public IDictionary<Int32, Int32> NumbersDic37_PastWindow { get { return _NumbersDic37_PastWindow; } }
        public IDictionary<Int32, Int32> NumbersDic7_PastWindow { get { return _NumbersDic7_PastWindow; } }



        public NumbersAnalysis()
        {
            Int32 i;

            for (i = 1; i <= 37; i++)
                _NumbersDic37_Total[i] = 0;

            for (i = 1; i <= 7; i++)
                _NumbersDic7_Total[i] = 0;

            for (i = 1; i <= 6; i++)
            {
                _NumbersDic37_PastWindow[i] = 0;
            }

            _NumbersDic7_PastWindow[1] = 0;
        }
    }
}

