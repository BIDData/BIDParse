/* function [C IA IB ..] = cellstrunion(A, B, ...)   *
 *                                                   * 
 *  where A, B,... are cell arrays of strings,       *
 *  return cell array of strings C containing        *
 *  one occurence of each string from A u B...       *
 *                                                   *
 *  If LHS index matrices are given, return          *
 *  indices IX such that A = C(IA), B = C(IB)...     */

#include "mex.h"
#include "utils.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{   
  mwIndex mA, nA, mB, nB, i, j, Nstr;
  mxArray *pB, *pC, *pStr;
  double *dC;
  char * str;
/* Check for proper number of input arguments */    
  if (nrhs < 1) 
    mexErrMsgTxt("cellstrunion: at least one input arg required.");
/* Now check types */
  for (i = 0; i < nrhs; i++) { 
    if(!mxIsCell(prhs[i]))
      mexErrMsgTxt("cellstrunion: inputs must be cell arrays of strings.");
  }

  unhash unh(0);
  strhash htab(0);
  for (i = 0; i < nrhs; i++) { 
    mA  = mxGetM(prhs[i]);
    nA  = mxGetN(prhs[i]);
    if (i+1 < nlhs) {
      pC = mxCreateDoubleMatrix(mA ,nA , mxREAL);
      if (pC == NULL)
	mexErrMsgTxt("cellstrunion: Index array allocation failed");
      plhs[i+1] = pC;
      dC = mxGetPr(pC);
    } else {
      dC = NULL;
    }
    for (j = 0; j < mA*nA; j++) {
      pStr = mxGetCell(prhs[i], j);
      if(!mxIsChar(pStr))
	mexErrMsgTxt("cellstrunion: inputs must be cell arrays of strings.");
      Nstr = mxGetN(pStr);
      try {
	str = new char[Nstr+1];
	mxGetString(pStr, str, Nstr+1);
	if (htab.count(str)) {
	  if (dC) dC[j] = htab[str];
	  delete [] str;
	} else {
	  unh.push_back(str);
	  htab[str] = (int)unh.size();
	  if (dC) dC[j] = (double)unh.size();
	}
      } catch (std::bad_alloc) {
	mexErrMsgTxt("cellstrunion: internal allocation error");
      }
    }
  }
  mB = unh.size();
  pB = mxCreateCellMatrix(mB, 1);
  if (pB == NULL)
    mexErrMsgTxt("cellstrunion: Cell array allocation failed");
  plhs[0] = pB;
  for (i = 0; i < mB; i++) {
    pStr = mxCreateString(unh[i]);
    if (pStr == NULL)
      mexErrMsgTxt("cellstrunion: String allocation failed");
    mxSetCell(pB, i, pStr);
    delete [] unh[i];
    unh[i] = NULL;
  }
}
  
