
int pcopy_transpose_in(int *iptrs, float *in, float *out, int stride, int nrows, int ncols);

int pcopy_transpose_out(int *optrs, float *in, float *out, int stride, int nrows, int ncols);

int viterbi(int nnsyms0, int ntsyms0, int nnsyms, int ntsyms, int ntrees, int isym,
	    float *wordsb4, float *wordsafter, float *treesb4, float *treesafter, 
	    int *iwordptr, int *itreeptr, int *parsetrees, float *parsevals, 
	    int *pp, int *darr, float *bval, int ldp,
	    int *upp, int *uarr, float *uval, int ldup);


