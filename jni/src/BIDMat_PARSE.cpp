
#include <jni.h>
#include <cuda_runtime.h>
#include "Logger.hpp"
#include "JNIUtils.hpp"
#include "PointerUtils.hpp"
#include "nncallall.h"
#include "ntcallall.h"
#include "tncallall.h"
#include "ttcallall.h"
#include "nncallallu.h"
#include "ntcallallu.h"
#include "ttcallallu.h"
#include "BIDMat_PARSE.hpp"

extern "C" {

  JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved)
  {
    JNIEnv *env = NULL;
    if (jvm->GetEnv((void **)&env, JNI_VERSION_1_4))
      {
        return JNI_ERR;
      }

    jclass cls = NULL;

    // Initialize the JNIUtils and PointerUtils
    if (initJNIUtils(env) == JNI_ERR) return JNI_ERR;
    if (initPointerUtils(env) == JNI_ERR) return JNI_ERR;

    return JNI_VERSION_1_4;

  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_PARSE_nnrules
  (JNIEnv *env, jobject obj, jobject jP, jobject jL, jobject jR,
   jobject jScale, jint ndo, jint nthreads) 
  {
    float *P = (float*)getPointer(env, jP);
    float *L = (float*)getPointer(env, jL);
    float *R = (float*)getPointer(env, jR);
    float *scale = (float*)getPointer(env, jScale);

    return nncallall(P, L, R, scale, ndo, nthreads);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_PARSE_ntrules
  (JNIEnv *env, jobject obj, jobject jP, jobject jL, jobject jR,
   jobject jScale, jint ndo, jint nthreads) 
  {
    float *P = (float*)getPointer(env, jP);
    float *L = (float*)getPointer(env, jL);
    float *R = (float*)getPointer(env, jR);
    float *scale = (float*)getPointer(env, jScale);

    return ntcallall(P, L, R, scale, ndo, nthreads);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_PARSE_tnrules
  (JNIEnv *env, jobject obj, jobject jP, jobject jL, jobject jR,
   jobject jScale, jint ndo, jint nthreads) 
  {
    float *P = (float*)getPointer(env, jP);
    float *L = (float*)getPointer(env, jL);
    float *R = (float*)getPointer(env, jR);
    float *scale = (float*)getPointer(env, jScale);

    return tncallall(P, L, R, scale, ndo, nthreads);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_PARSE_ttrules
  (JNIEnv *env, jobject obj, jobject jP, jobject jL, jobject jR,
   jobject jScale, jint ndo, jint nthreads) 
  {
    float *P = (float*)getPointer(env, jP);
    float *L = (float*)getPointer(env, jL);
    float *R = (float*)getPointer(env, jR);
    float *scale = (float*)getPointer(env, jScale);

    return ttcallall(P, L, R, scale, ndo, nthreads); 
    }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_PARSE_nnrulesu
  (JNIEnv *env, jobject obj, jobject jP, jobject jL, jobject jR,
   jobject jScale, jint ndo, jint nthreads) 
  {
    float *P = (float*)getPointer(env, jP);
    float *L = (float*)getPointer(env, jL);
    float *scale = (float*)getPointer(env, jScale);

    return nncallallu(P, L, scale, ndo, nthreads);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_PARSE_ntrulesu
  (JNIEnv *env, jobject obj, jobject jP, jobject jL, jobject jR,
   jobject jScale, jint ndo, jint nthreads) 
  {
    float *P = (float*)getPointer(env, jP);
    float *L = (float*)getPointer(env, jL);
    float *scale = (float*)getPointer(env, jScale);

    return ntcallallu(P, L, scale, ndo, nthreads);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_PARSE_ttrulesu
  (JNIEnv *env, jobject obj, jobject jP, jobject jL, jobject jR,
   jobject jScale, jint ndo, jint nthreads) 
  {
    float *P = (float*)getPointer(env, jP);
    float *L = (float*)getPointer(env, jL);
    float *scale = (float*)getPointer(env, jScale);

    return ttcallallu(P, L, scale, ndo, nthreads); 
    }


  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_PARSE_pcopytxin
  (JNIEnv *env, jobject obj, jobject jiptrs, jobject jin, jobject jout, jint stride, jint nrows, jint ncols) 
  {
    int *iptrs = (int*)getPointer(env, jiptrs);
    float *in = (float*)getPointer(env, jin);
    float *out = (float*)getPointer(env, jout);

    return pcopy_transpose_in(iptrs, in, out, stride, nrows, ncols);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_PARSE_pcopytxout
  (JNIEnv *env, jobject obj, jobject joptrs, jobject jin, jobject jout, jint stride, jint nrows, jint ncols) 
  {
    int *optrs = (int*)getPointer(env, joptrs);
    float *in = (float*)getPointer(env, jin);
    float *out = (float*)getPointer(env, jout);

    return pcopy_transpose_out(optrs, in, out, stride, nrows, ncols);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_PARSE_viterbiGPU
  (JNIEnv *env, jobject obj, jint nnsyms0, jint ntsyms0, jint nnsyms, jint ntsyms, jint ntrees, jint isym,
   jobject j_wordsb4, jobject j_wordsafter, jobject j_treesb4, jobject j_treesafter,
   jobject j_iwordptr, jobject j_itreeptr, jobject j_parsetrees, jobject j_parsevals,
   jobject j_pp, jobject j_darr, jobject j_bval, jint ldp,
   jobject j_upp, jobject j_uarr, jobject j_uval, jint ldup)
  {
    float *wordsb4 = (float*)getPointer(env, j_wordsb4);
    float *wordsafter = (float*)getPointer(env, j_wordsafter);
    float *treesb4 = (float*)getPointer(env, j_treesb4);
    float *treesafter = (float*)getPointer(env, j_treesafter);
    int *iwordptr = (int*)getPointer(env, j_iwordptr);
    int *itreeptr = (int*)getPointer(env, j_itreeptr);
    int *parsetrees = (int*)getPointer(env, j_parsetrees);
    float *parsevals = (float*)getPointer(env, j_parsevals);
    int *pp = (int*)getPointer(env, j_pp);
    int *darr = (int*)getPointer(env, j_darr);
    float *bval = (float*)getPointer(env, j_bval);
    int *upp = (int*)getPointer(env, j_upp);
    int *uarr = (int*)getPointer(env, j_uarr);
    float *uval = (float*)getPointer(env, j_uval);

    return viterbi(nnsyms0, ntsyms0, nnsyms, ntsyms, ntrees, isym,
		   wordsb4, wordsafter, treesb4, treesafter,
		   iwordptr, itreeptr, parsetrees, parsevals,
		   pp, darr, bval, ldp,
		   upp, uarr, uval, ldup);
  }
}
