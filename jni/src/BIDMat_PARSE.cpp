
#include <jni.h>
#include <cuda_runtime.h>
#include "Logger.hpp"
#include "JNIUtils.hpp"
#include "PointerUtils.hpp"
#include "nncallall.h"
#include "ntcallall.h"
#include "tncallall.h"
#include "ttcallall.h"
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


  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_PARSE_pcopyTXin
  (JNIEnv *env, jobject obj, jobject jiptrs, jobject jin, jobject jout, jint stride, jint nrows, jint ncols) 
  {
    int *iptrs = (int*)getPointer(env, jiptrs);
    float *in = (float*)getPointer(env, jin);
    float *out = (float*)getPointer(env, jout);

    return pcopy_transpose_in(iptrs, in, out, stride, nrows, ncols);
  }

  JNIEXPORT jint JNICALL Java_edu_berkeley_bid_PARSE_pcopyTXout
  (JNIEnv *env, jobject obj, jobject joptrs, jobject jin, jobject jout, jint stride, jint nrows, jint ncols) 
  {
    int *optrs = (int*)getPointer(env, joptrs);
    float *in = (float*)getPointer(env, jin);
    float *out = (float*)getPointer(env, jout);

    return pcopy_transpose_out(optrs, in, out, stride, nrows, ncols);
  }
}
