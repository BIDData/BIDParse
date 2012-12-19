
extern __global__ void transpose(float *in, int instride, float *out, int outstride, int inrows, int incols);
extern __global__ void pcopy_transpose(float **in, float *out, int outstride, int nrows, int ncols);
extern __global__ void pcopy_setup(int *inds, float **src, float *target, int ncols, int stride);
extern __global__ void pcopy(float **src, float *dest, int nrows, int ncols, int stride);
extern void testvmax(float *vec, float *cvec, int n, int nreps, int i);

const int stride = 8192;
const int BLOCKDIM = 32;
const int INBLOCK = 4;
const int NSYMS = 640;

#define DATADIR "c:/data/Grammar/"
const char nnfname[] = DATADIR "nnbinrulesx.dat";
const char ntfname[] = DATADIR "ntbinrulesx.dat";
const char tnfname[] = DATADIR "tnbinrulesx.dat";
const char ttfname[] = DATADIR "ttbinrulesx.dat";
