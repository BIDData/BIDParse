/* ==================================================================
 * Matlab call [A B C] = cellstrsplit(X, d)
 * where X is a cell arrays of n strings, 
 * return sparse mxn array A of indices where A(:,i) are
 * the indices of the substrings of X{i} split by the delimiter d. 
 * B is a cell array of strings that correspond to the indices in A. 
 * C is an array of counts for each index. 
 * ================================================================== */

#include "mex.h"
#include "utils.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{   
  mwIndex mA, nA, mB, nB, i, Nstr;
  mxArray * pStr, * pC;
  char * str, * delims;
/* Check for proper number of input arguments */    
  if (nrhs != 2) 
    mexErrMsgTxt("cellstrsplit: two input args required.");
/* Now check types */
  if(!mxIsCell(prhs[0]))
    mexErrMsgTxt("cellstrsplit: first input must be cell array of strings");
  if(!mxIsChar(prhs[1]))
    mexErrMsgTxt("cellstrsplit: second input must be a string");
  mA  = mxGetM(prhs[0]);
  nA  = mxGetN(prhs[0]);
  mB  = mxGetM(prhs[1]);
  nB  = mxGetN(prhs[1]);
  if (min(mA, nA) > 1 || min(mB, nB) > 1)
    mexErrMsgTxt("cellstrsplit: inputs must be vectors");
  nA *= mA;
  nB *= mB;
  Nstr = mxGetN(prhs[1]);
  try {
    delims = new char[Nstr+1];
    str = new char[BUFSIZE];
  } catch (std::bad_alloc) {
    mexErrMsgTxt("cellstrsplit: internal allocation error");
  }
  mxGetString(prhs[1], delims, Nstr+1);
  stringIndexer si;
  imatrix im(nA);
  for (i = 0; i < nA; i++) {
    pStr = mxGetCell(prhs[0], i);
    if(!mxIsChar(pStr))
      mexErrMsgTxt("cellstrsplit: first input must be cell array of strings.");
    Nstr = mxGetN(pStr);
    mxGetString(pStr, str, Nstr+1);
    im[i] = si.checkstring(str, delims);
  }
  plhs[0] = matGetIntVecs2(im);
  if (plhs[0] == NULL) 
    mexErrMsgTxt("cellstrsplit: Output array allocation failed");
  if (nlhs > 1) {
    plhs[1] = matGetStrings(si.unh);
    if (plhs[1] == NULL) 
      mexErrMsgTxt("cellstrsplit: Output array allocation failed");
    if (nlhs > 2) {
      plhs[2] = matGetInts(si.count);
      if (plhs[2] == NULL) 
	mexErrMsgTxt("cellstrsplit: Output array allocation failed");
    }
  }
  delete [] str;
  if (delims) delete [] delims;
}
    
  
