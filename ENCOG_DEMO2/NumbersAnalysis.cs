using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LotoPrediction
{
    public class NumbersAnalysis
    {
        private Dictionary<int, int> _NumbersDic37_Total = new Dictionary<int, int>();
        private Dictionary<int, int> _NumbersDic7_Total = new Dictionary<int, int>();

        private Dictionary<int, int> _NumbersDic37_PastWindow = new Dictionary<int, int>();
        private Dictionary<int, int> _NumbersDic7_PastWindow = new Dictionary<int, int>();


        public IDictionary<int, int> NumbersDic37_Total { get; set; }
        public IDictionary<int, int> NumbersDic7_Total { get; set; }
        public IDictionary<int, int> NumbersDic37_PastWindow { get; set; }
        public IDictionary<int, int> NumbersDic7_PastWindow { get; set; }


        public NumbersAnalysis() {
            int i;

            for (i = 1; i <= 37; i++) _NumbersDic37_Total[i] = 0;
            for (i = 1; i <= 7; i++) _NumbersDic7_Total[i] = 0;
        }
    }

}


/*
    NumbersAnalysis NA = new NumbersAnalysis();

    NA.NumbersDic37.Add(1, 1);
    NA.NumbersDic37[2] = NA.NumbersDic[2]++;
    

    
    Console.WriteLine(NA.NumbersDic37[2]);
*/
