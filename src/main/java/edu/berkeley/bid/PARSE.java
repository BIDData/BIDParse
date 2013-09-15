package edu.berkeley.bid;
import jcuda.*;
import jcuda.runtime.*;

public final class PARSE {

    private PARSE() {}

    static {
    	jcuda.LibUtils.loadLibrary("jniparse");
    }

    public static native int nnrules(Pointer P, Pointer L, Pointer R, Pointer scale, int ndo, int nthreads);

    public static native int ntrules(Pointer P, Pointer L, Pointer R, Pointer scale, int ndo, int nthreads);

    public static native int tnrules(Pointer P, Pointer L, Pointer R, Pointer scale, int ndo, int nthreads);

    public static native int ttrules(Pointer P, Pointer L, Pointer R, Pointer scale, int ndo, int nthreads);
    
    public static native int nnrulesu(Pointer P, Pointer L, Pointer R, Pointer scale, int ndo, int nthreads);
    
    public static native int ntrulesu(Pointer P, Pointer L, Pointer R, Pointer scale, int ndo, int nthreads);
    
    public static native int ttrulesu(Pointer P, Pointer L, Pointer R, Pointer scale, int ndo, int nthreads);
    
    public static native int pcopytxin(Pointer iptrs, Pointer in, Pointer out, int stride, int nrows, int ncols);
    
    public static native int pcopytxout(Pointer optrs, Pointer in, Pointer out, int stride, int nrows, int ncols);
    
    public static native int viterbiGPU(int nnsyms0, int ntsyms0, int nnsyms, int ntsyms, int ntrees, int isym,
    		Pointer wordsb4, Pointer wordsafter, Pointer treesb4, Pointer treesafter, 
    		Pointer iwordptr, Pointer itreeptr, Pointer parsetrees, Pointer parsevals, 
    		Pointer pp, Pointer darr, Pointer bval, int nbrules,
    		Pointer upp, Pointer uarr, Pointer uval, int nurules);
}
