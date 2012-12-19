/* ==================================================================
 * Matlab call C = cellstrmap(A, B)
 * where A and B are cell arrays of strings, 
 * return array C of integers such that
 * A(i) = B(C(i)) if A(i) occurs in B,
 * C(i) = 0 otherwise
 * ================================================================== */

#include "mex.h"
#include "utils.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{   
  mwIndex mA, nA, mB, nB, i, Nstr;
  mxArray * pStr;
  double * pC;
  char * str;
/* Check for proper number of input arguments */    
  if (nrhs != 2) 
    mexErrMsgTxt("cellstrmap: two input args required.");
/* Now check types */
  if(!mxIsCell(prhs[0]) || !mxIsCell(prhs[1]))
    mexErrMsgTxt("cellstrmap: inputs must be cell arrays of strings");
  mA  = mxGetM(prhs[0]);
  nA  = mxGetN(prhs[0]);
  mB  = mxGetM(prhs[1]);
  nB  = mxGetN(prhs[1]);
  if (min(mA, nA) > 1 || min(mB, nB) > 1)
    mexErrMsgTxt("cellstrmap: inputs must be vectors");
  nA *= mA;
  nB *= mB;
  strhash htab(0);
  unhash unh(0);
/* fill a hash map with strings from B pointing to the corresponding index of B */
  for (i = 0; i < nB; i++) {
    pStr = mxGetCell(prhs[1], i);
    if(!mxIsChar(pStr))
      mexErrMsgTxt("cellstrmap: inputs must be cell arrays of strings.");
    Nstr = mxGetN(pStr);
    try {
      str = new char[Nstr+1];
      mxGetString(pStr, str, Nstr+1);
      if (htab.count(str)) {
	delete [] str;
	mexErrMsgTxt("cellstrmap: duplicate string in second arg");
      } else {
	htab[str] = i+1;
	unh.push_back(str);
      }
    } catch (std::bad_alloc) {
      mexErrMsgTxt("cellstrmap: internal allocation error");
    }
  }
  plhs[0] = mxCreateDoubleMatrix(nA, 1, mxREAL);
  if (plhs[0] == NULL) {
    mexErrMsgTxt("cellstrmap: Output array allocation failed");
  }
  pC = mxGetPr(plhs[0]);
/* now look up strings from A to get the corresponding index of B */
  for (i = 0; i < nA; i++) {
    pStr = mxGetCell(prhs[0], i);
    if(!mxIsChar(pStr))
      mexErrMsgTxt("cellstrmap: inputs must be cell arrays of strings.");
    Nstr = mxGetN(pStr);
    try {
      str = new char[Nstr+1];
      mxGetString(pStr, str, Nstr+1);
      if (htab.count(str)) {
	pC[i] = htab[str];
      } else {
	pC[i] = 0;
      }
      delete [] str;
    } catch (std::bad_alloc) {
      mexErrMsgTxt("cellstrmap: internal allocation error");
    }
  }
  for (i = 0; i < unh.size(); i++) {
    delete [] unh[i];
    unh[i] = NULL;
  }
}
    
  
