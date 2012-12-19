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

__global__ void viterbi(int *lrules, int *rrules, float *bivals, int *bip, int n, 
                        int *unirules, float *univals, int *unip,
                        float **allbitrees, float **allunitrees, int *lengths, 
                        int **parsetrees, int **nsyms, float **nscores) {
  __shared__ float maxval[1];
  __shared__ int imaxi[1];
  __shared__ int left[1];
  __shared__ float SHSTORE(lvals);
  __shared__ float SHSTORE(rvals);
  __shared__ float SHSTORE(lscores);
  __shared__ float SHSTORE(rscores);

  float *bitree = allbitrees[blockIdx.x];                              // Each block processes one tree, so gather all the data
  float *unitree = allunitrees[blockIdx.x];                            // for this tree.
  int length = lengths[blockIdx.x];
  int *parsetree = parsetrees[blockIdx.x];
  int *nsym = nsyms[blockIdx.x];
  float *nscore = nscores[blockIdx.x];

  for (int k = threadIdx.x; k < n; k += blockDim.x) {
    SHACCESS(lvals, k) = bitree[k + NSYMS*TREEINDX(0, length-1)];    // Load root node scores
  }
  __syncthreads();

  findvmax(lvals, n, maxval, imaxi);                                   // Find the best score

  if (threadIdx.x < 1) {                                               // Initialize root node props
    parsetree[0] = 2*length-1;
    nsym[0] = imaxi[0];
    nscore[0] = maxval[0];
    left[0] = 0;                                                       // Number of input symbols processed so far/ x coord
  }
  __syncthreads();

  for (int i = 0; i < 2*length-3; i++) {
    int x = left[0];                                                   // x coord of current node                       
    int y = (parsetree[i]-i-1)/2;                                      // y coord of current node
    int here = nsym[i];                                                // this node's best symbol

    float lmax = 0;                                                    // score of best rule, tallied over left children
    float rmax = 0;                                                    // score of best rule, tallied over right children
    int hmax = 0;                                                      // height of best left child
    int symleft = -1;                                                  // best left symbol
    int symright = -1;                                                 // best right symbol

    //    float thisc = nscore[i];

    for (int k = threadIdx.x; k < n; k += blockDim.x) {
      SHACCESS(lvals, k) = unitree[k + NSYMS*TREEINDX(x,y)];         // load pre-unary-rule scores for this node
      SHACCESS(lscores, k) = 0;
    }
    __syncthreads();

    for (int k = unip[here] + threadIdx.x; k < unip[here+1]; k += blockIdx.x) {  // Compute pre-unary rule scores given root symbol
      float tscore = SHACCESS(lvals,unirules[k]) * univals[k];
      SHACCESS(lscores, unirules[k]) = max(SHACCESS(lscores, unirules[k]), tscore);
    }
    __syncthreads();
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    findvmax(lscores, n, maxval, imaxi);                               // Find best pre-unary symbol
    here = imaxi[0];
    //   thisc = maxvals[0];

    for (int j = 0; j < y; j++) {                                      // Loop over splits for this node, if its not a leaf
      for (int k = threadIdx.x; k < n; k += blockDim.x) {
        SHACCESS(lvals, k) = bitree[k + NSYMS*TREEINDX(x,j)];          // Load left and right node score for this split
        SHACCESS(rvals, k) = bitree[k + NSYMS*TREEINDX(x+1+j, y-1-j)];
        SHACCESS(lscores, k) = 0;
        SHACCESS(rscores, k) = 0;
      }
      __syncthreads();

      for (int k = bip[here] + threadIdx.x; k < bip[here+1]; k += blockIdx.x) { // Compute left and right scores given root symbol
        float tscore = SHACCESS(lvals,lrules[k]) * SHACCESS(rvals,rrules[k]) * bivals[k];
        SHACCESS(lscores, lrules[k]) = max(SHACCESS(lscores, lrules[k]), tscore);
        SHACCESS(rscores, rrules[k]) = max(SHACCESS(rscores, rrules[k]), tscore);
      }
      __syncthreads();

      findvmax(lscores, n, maxval, imaxi);                             // Find the best left score and its height
      if (threadIdx.x < 1 && maxval[0] > lmax) {
        lmax = maxval[0];
        hmax = j;
        symleft = imaxi[0];
      }
      __syncthreads();

      findvmax(rscores, n, maxval, imaxi);                             // Find the best (hopefully matching) right score
      if (threadIdx.x < 1 && maxval[0] > rmax) {
        rmax = maxval[0];
        symright = imaxi[0];
      }
      __syncthreads();
    }

    if (threadIdx.x < 1) {                                             // Update data with one thread
      if (y > 0) {                                                     // If we're not at a leaf, process children
        int nexti = i+2*hmax+2;                                        // Tree is prefix order, left child is always the next node
        parsetree[i+1] = nexti;                                        // parsetree[x] points to next sibling of x, or just beyond own subtree for right siblings
        nsym[i+1] = symleft;
        nscore[i+1] = lmax;
        parsetree[nexti] = parsetree[i];                               // Right sibling points just beyond own subtree (same as beyond parent's subtree)
        nsym[nexti] = symright;
        nscore[nexti] = rmax;
      } else {                                                         // We processed a leaf, so advance our leaf pointer
        left[0] = left[0] + 1;
      }
    }
    __syncthreads();
  }
}
