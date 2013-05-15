#include <cuda_runtime.h>
#include <stdio.h>
#include "testSparse.h"

#define TREEINDX(xx,yy) ((xx) + (yy)*length - (yy)*((yy)-1)/2)

#define SHACCESS(vec,ind) vec[ind / BLOCKDIM][ind % BLOCKDIM]
#define SHSTORE(sname) sname[NSYMS/BLOCKDIM][BLOCKDIM+1]
//#define SHACCESS(vec,ind) vec[ind % BLOCKDIM][ind / BLOCKDIM]
//#define SHSTORE(sname) sname[BLOCKDIM][BLOCKDIM+1]
//#define SHACCESS(vec,ind) vec[ind]
//#define SHSTORE(sname) sname[NSYMS]
          

__device__ void findvmax(float SHSTORE(scores), int n, float *outv, int *outi) {
  __shared__ float locv[NSYMS/BLOCKDIM];
  __shared__ int loci[NSYMS/BLOCKDIM];
  float newv;
  int newi;

  float mymax = SHACCESS(scores, threadIdx.x);
  int myi = threadIdx.x;

  for (int i = threadIdx.x + blockDim.x; i < n; i += blockDim.x) {
    newv = SHACCESS(scores, i);
    if (newv > mymax) {
      mymax = newv;
      myi = i;
    }
  }

  for (int i = 1; i < 32; i *= 2) {
    newv = __shfl_down(mymax, i);
    newi = __shfl_down(myi, i);
    if (newv > mymax) {
      mymax = newv;
      myi = newi;
    }
  }

  if (threadIdx.x % BLOCKDIM == 0) {
    locv[threadIdx.x / BLOCKDIM] = mymax;
    loci[threadIdx.x / BLOCKDIM] = myi;
  }
  __syncthreads();

  if (threadIdx.x < blockDim.x/BLOCKDIM) {
    mymax = locv[threadIdx.x];
    myi = loci[threadIdx.x];
    for (int i = 1; i < blockDim.x/BLOCKDIM; i *= 2) {
      newv = __shfl_down(mymax, i);
      newi = __shfl_down(myi, i);
      if (newv > mymax) {
        mymax = newv;
        myi = newi;
      }
    }
  }
  if (threadIdx.x < 1) { 
    outv[0] = mymax;
    outi[0] = myi;
  }
  __syncthreads();
}

__device__ void findvmax2(float SHSTORE(scores), int n, float *outv, int *outi) {
  __shared__ float locv[BLOCKDIM];
  __shared__ int loci[BLOCKDIM];
  float mymax = 0;
  int myi = 0;
  if (threadIdx.x < BLOCKDIM) {
    for (int i = threadIdx.x; i < n; i += BLOCKDIM) {
      if (SHACCESS(scores,i) > mymax) {
        mymax = SHACCESS(scores,i);
        myi = i;
      }
    }
    locv[threadIdx.x] = mymax;
    loci[threadIdx.x] = myi;
  }  
  __syncthreads();

  if (threadIdx.x < BLOCKDIM) {
    mymax = locv[threadIdx.x];
    myi = loci[threadIdx.x];
    for (int i = 1; i < 32; i *= 2) {
      float newv = __shfl_down(mymax, i);
      int newi = __shfl_down(myi, i);
      if (newv > mymax) {
        mymax = newv;
        myi = newi;
      }
    }
  }
  if (threadIdx.x < 1) {
    outv[0] = mymax;
    outi[0] = myi;
  }
  __syncthreads();
}

__global__ void __testvmax2(float *vec, int n, float *pkvmax, int *pkimax, int nreps) {

	for (int irep = 0; irep < nreps; irep++) {
  float maxv = vec[threadIdx.x];
  int maxi = threadIdx.x;

  for (int i = threadIdx.x + 32; i < n; i += 32) {
	  if (vec[i] > maxv) {
		  maxv = vec[i];
		  maxi = i;
	  }
  }
  __syncthreads();

  for (int i = 1; i < 32; i *= 2) {
      float newv = __shfl_down(maxv, i);
      int newi = __shfl_down(maxi, i);
      if (newv > maxv) {
        maxv = newv;
        maxi = newi;
      }
    }

  if (threadIdx.x < 1) {
	  pkvmax[0] = maxv;
	  pkimax[0] = maxi;
  }
	}
	__syncthreads();
}

__global__ void __testvmax(float *vec, int n, float *pkvmax, int *pkimax, int nreps) {
  __shared__ float SHSTORE(vals);

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    SHACCESS(vals, i) = vec[i];
  }
  __syncthreads();

  for (int i = 0; i < nreps; i++) {
    findvmax(vals, n, pkvmax, pkimax);
  }
}

void testvmax(float *vec, float *cvec, int n, int nreps, int iter) {
  float vmax, kvmax, *pkvmax;
  int imax, kimax, *pkimax;
  cudaMalloc((void**) &pkvmax, sizeof(float));
  cudaMalloc((void**) &pkimax, sizeof(int));

//  __testvmax<<<1,320>>>(vec, n, pkvmax, pkimax, nreps);
  __testvmax<<<1,32>>>(vec, n, pkvmax, pkimax, nreps);

  cudaDeviceSynchronize();
  if (iter == 0) {
  cudaMemcpy(&kvmax, pkvmax, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&kimax, pkimax, sizeof(int), cudaMemcpyDeviceToHost);
  vmax = 0;
  for (int i = 0; i < n; i++) {
    if (cvec[i] > vmax) {
      vmax = cvec[i];
      imax = i;
    }
  }
  if (imax == kimax && vmax == kvmax) {
    printf("\nMax computed correctly\n");
  } else {
    printf("\nMax=%f(%i), kernel=%f(%i)\n", vmax, imax, kvmax, kimax);
  }
  }
  cudaFree(pkvmax);
  cudaFree(pkimax);
}

#define NVITTHREADS 128
#define MINVAL -1.0e8f

#define treestep(x,y,len) ((x) + (y)*len - (y)*((y)-1)/2)

typedef struct maxfields {
  int l;
  int r;
  int y;
  float vl;
  float vr;
  float v;
} maxfields_t;

typedef struct dcontents {
  int iv;
  float fv;
} dcontents_t;
  
__device__ void __vitunary(int isym, double *dmax, float *lscores, int nsyms,
		                   float *lmat, int tti, int *upp, int *uarr, float *uval, int nurules)
{
  double dtest;
  dcontents_t *dt = (dcontents_t *)&dtest;
  dt->iv = 0;
  dt->fv = MINVAL;
  __syncthreads();
  for (int i = threadIdx.x; i < nsyms; i += blockDim.x) {               // Load input symbol scores
    lscores[i] = lmat[tti + i];
  }
  if (threadIdx.x != 0) dmax[threadIdx.x] = dtest;                      // Clear max value tree, except dmax[0]
  __syncthreads();
  for (int t = upp[isym] + threadIdx.x; t < upp[isym+1]; t += blockDim.x) {
    int ileft = uarr[nurules + t];
    dt->iv = ileft;                                                     // Combine score and rule index into a pseudo-double
    dt->fv = uval[t] + lscores[ileft];
    dmax[threadIdx.x] = max(dmax[threadIdx.x], dtest);                  // Update the max for this thread
  }
  __syncthreads();
  for (int step = 1; step < NVITTHREADS; step *= 2) {                   // Combine max values across threads in a tree
    if (threadIdx.x % (2*step) == 0) {
      dtest = dmax[threadIdx.x+step];
    }
    __syncthreads();
    if (threadIdx.x % (2*step) == 0) {
      dmax[threadIdx.x] = max(dtest, dmax[threadIdx.x]);
    }
    __syncthreads();
  }
}

__device__ void __vitbinary(int isym, double *dmax, maxfields_t *maxf, int y, 
			    float *lscores, int nlsyms, float *rscores, int nrsyms,
			    float *lmat, int lbase, int lsymoff, float *rmat, int rbase, int rsymoff, 
			    int *pp, int *darr, float *bval, int nbrules)
{
  double dtest;
  dcontents_t *dt = (dcontents_t *)&dtest;
  __syncthreads();
  double oldmax = dmax[0];
  dt->iv = 0;
  dt->fv = MINVAL;
  for (int i = threadIdx.x; i < nlsyms; i += blockDim.x) {
    lscores[i] = lmat[lbase + i];
  }
  for (int i = threadIdx.x; i < nrsyms; i += blockDim.x) {
    rscores[i] = rmat[rbase + i];
  }
  if (threadIdx.x != 0) dmax[threadIdx.x] = dtest;
  __syncthreads();
  for (int t = pp[isym] + threadIdx.x; t < pp[isym+1]; t += blockDim.x) {
    int ileft = darr[nbrules + t];
    int iright = darr[2*nbrules + t];
    dt->iv = t;
    dt->fv = bval[t] + lscores[ileft] + rscores[iright];
    dmax[threadIdx.x] = max(dmax[threadIdx.x], dtest);
  }
  __syncthreads();
  for (int step = 1; step < NVITTHREADS; step *= 2) {
    if (threadIdx.x % (2*step) == 0) {
      dtest = dmax[threadIdx.x+step];
    }
    __syncthreads();
    if (threadIdx.x % (2*step) == 0) {
      dmax[threadIdx.x] = max(dtest, dmax[threadIdx.x]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    dtest = dmax[0];
    if (oldmax != dtest) {
      int ileft = darr[nbrules + dt->iv];
      int iright = darr[2*nbrules + dt->iv];
      maxf->l = ileft + lsymoff;
      maxf->r = iright + rsymoff;
      maxf->y = y;
      maxf->vl = lscores[ileft];
      maxf->vr = rscores[iright];
      maxf->v = dt->fv;
    }
  }
  __syncthreads();
}

__global__ void __viterbi(int nnsyms0, int ntsyms0, int nnsyms, int ntsyms, int ntrees, int isym,
			  float *wordsb4, float *wordsafter, float *treesb4, float *treesafter, 
			  int *iwordptr, int *itreeptr, int *parsetrees, float *parsevals, 
			  int *pp, int *darr, float *bval, int nbrules,
			  int *upp, int *uarr, float *uval, int nurules)
{
  __shared__ float lscores[NSYMS];
  __shared__ float rscores[NSYMS];
  __shared__ double dmax[NVITTHREADS];
  __shared__ maxfields_t maxf;

  const int prows = 3;
  int noff = nnsyms0 - 3;
  for (int bid = blockIdx.x; bid < ntrees; bid += gridDim.x) {
    int x0 = iwordptr[bid];
    int t0 = itreeptr[bid];
    int sentlen = iwordptr[bid] - x0;
    int x = 0;                                              // Word position
    int tpos = 2*x0;                                        // Parse tree position
    int pti = prows*tpos;
    dcontents_t *dm0 = (dcontents_t *)&dmax[0];
    if (threadIdx.x == 0) {
      parsetrees[pti] = sentlen-1;                            // Init height of the main tree
      parsetrees[1+pti] = isym;                               // Symbol at root of main tree
    }
    while (tpos < 2*(x0+sentlen)-1) {
      __syncthreads();
      pti = prows*tpos;
      int height = parsetrees[pti];                         // Height of the current tree
      isym = parsetrees[1+pti];                             // Symbol at root of current tree
      int tti = t0 + treestep(x,height,sentlen);
      if (threadIdx.x == 0) {                               // Clear the maximum var
        dm0->iv = 0;                                        // Low order word holds the rule number (use a double to hold value + rule number)
        dm0->fv = MINVAL;                                   // High order word holds the value
      }
      __syncthreads();		
      if (height > 0) {
        __vitunary(isym, dmax, lscores, nnsyms0, treesb4, tti*nnsyms, upp, uarr, uval, nurules); // Pull down the root symbol through unary rules
        isym = dm0->iv;
        __syncthreads();
        if (threadIdx.x == 0) {
          parsetrees[2+pti] = isym;                         // Save the result in parsetrees[2,pti]
          maxf.l = -1;                                      // maxf holds all the metadata for the best rule
          maxf.r = -1;
          maxf.y = -1;
          maxf.vl = MINVAL;
          maxf.vr = MINVAL;
          maxf.v = MINVAL;
          dm0->iv = 0;
          dm0->fv = MINVAL;
        }
        __syncthreads();
        for (int y = 0; y < height; y++) {                  // Process all splits for this node
          int ts1 = nnsyms*(t0+treestep(x, y, sentlen));    // Array addresses of children
          int ts2 = nnsyms*(t0+treestep(x+y+1, height-y-1, sentlen));
          int x1 = ntsyms*(x+x0);
          int x2 = ntsyms*(x+x0+y+1);

          __vitbinary(isym, dmax, &maxf, y, lscores, nnsyms0, rscores, nnsyms0, 
                      treesafter, ts1, 0, treesafter, ts2, 0, pp, darr, bval, nbrules);
          if (y == 0) {
            __vitbinary(isym, dmax, &maxf, y, lscores, ntsyms0, rscores, nnsyms0, 
                        wordsafter, x1, noff, treesafter, ts2, 0, pp+2*(nnsyms0+1), darr, bval, nbrules);
          }
          if (y == height-1) {
            __vitbinary(isym, dmax, &maxf, y, lscores, nnsyms0, rscores, ntsyms0, 
                        treesafter, ts1, 0, wordsafter, x2, noff, pp+(nnsyms0+1), darr, bval, nbrules);
          }
          if (height == 1) {
            __vitbinary(isym, dmax, &maxf, y, lscores, ntsyms0, rscores, ntsyms0, 
                        wordsafter, x1, noff, wordsafter, x2, noff, pp+3*(nnsyms0+1), darr, bval, nbrules);
          }
        }  			
        if (threadIdx.x == 0) {                            // Update parsetree data. 
          parsetrees[prows*(tpos+1)] = maxf.y;             // Left child (next position in tree array) height
          parsetrees[1+prows*(tpos+1)] = maxf.l;           // Left child symbol
          parsevals[tpos+1] = maxf.vl;                     // Left child score
          int nexti = tpos+2*(maxf.y+1);                   // Address of right child
          parsetrees[prows*nexti] = height - maxf.y - 1;   // Right child height
          parsetrees[1+prows*nexti] = maxf.r;              // Right child symbol
          parsevals[nexti] = maxf.vr;                      // Right child score
        }
      } else {                                             // height == 0, advance the leaf pointer
        if (isym < nnsyms0-3) {                              // max symbol was a non-terminal
          __vitunary(isym, dmax, lscores, ntsyms0, wordsb4, (x+x0)*ntsyms, upp+(nnsyms0+1), uarr, uval, nurules); // back down through unary rules
        } else {                                           // max symbol was a terminal
          __vitunary(isym-noff, dmax, lscores, ntsyms0, wordsb4, (x+x0)*ntsyms, upp+2*(nnsyms0+1), uarr, uval, nurules); 
        }
        if (threadIdx.x == 0) parsetrees[2+prows*tpos] = (dm0->iv) + noff; 
        x++;                                               // We consumed a non-terminal, so step the word ptr
      }
      __syncthreads();
      tpos++;                                              // Advance the tree pointer
    }
  }
}


int viterbi(int nnsyms0, int ntsyms0, int nnsyms, int ntsyms, int ntrees, int isym,
	    float *wordsb4, float *wordsafter, float *treesb4, float *treesafter, 
	    int *iwordptr, int *itreeptr, int *parsetrees, float *parsevals, 
	    int *pp, int *darr, float *bval, int nbrules,
	    int *upp, int *uarr, float *uval, int nurules) {

  int nblocks = min(256,ntrees);

  __viterbi<<<nblocks,NVITTHREADS>>>(nnsyms0, ntsyms0, nnsyms, ntsyms, ntrees, isym,
				    wordsb4, wordsafter, treesb4, treesafter,
				    iwordptr, itreeptr, parsetrees, parsevals,
				    pp, darr, bval, nbrules,
				    upp, uarr, uval, nurules);

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  return err;
}
