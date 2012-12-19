package edu.berkeley.bid;
import jcuda.*;
import jcuda.runtime.*;

public final class PARSE {

    private PARSE() {}

    static {
        System.loadLibrary("jniparse");
    }

    public static native int nnrules(Pointer P, Pointer L, Pointer R, Pointer scale, int ndo, int nthreads);

    public static native int ntrules(Pointer P, Pointer L, Pointer R, Pointer scale, int ndo, int nthreads);

    public static native int tnrules(Pointer P, Pointer L, Pointer R, Pointer scale, int ndo, int nthreads);

    public static native int ttrules(Pointer P, Pointer L, Pointer R, Pointer scale, int ndo, int nthreads);
    
    public static native int pcopyTXin(Pointer iptrs, Pointer in, Pointer out, int stride, int nrows, int ncols);
    
    public static native int pcopyTXout(Pointer optrs, Pointer in, Pointer out, int stride, int nrows, int ncols);
}
