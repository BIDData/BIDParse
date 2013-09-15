package BIDParse
import BIDMat.{Mat, DMat, FMat, IMat, CMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat}
import BIDMat.SciFunctions._
import BIDMat.MatFunctions._


import jcuda._
import jcuda.runtime.JCuda
import jcuda.runtime.JCuda._
import jcuda.runtime.cudaMemcpyKind._
import jcuda.jcublas.JCublas;
import edu.berkeley.bid.CUMAT
import edu.berkeley.bid.CBLAS
import edu.berkeley.bid.PARSE._
import edu.berkeley.nlp.syntax.Tree
import edu.berkeley.nlp.syntax.Trees
import edu.berkeley.nlp.PCFGLA._
import edu.berkeley.nlp.PCFGLA.Corpus.TreeBankType
import edu.berkeley.nlp.parser.EnglishPennTreebankParseEvaluator
import edu.berkeley.nlp.io.{PTBLineLexer, PTBTokenizer}
import edu.berkeley.nlp.util.Numberer
import scala.collection.mutable.ListBuffer
import collection.mutable
import collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._
import scala.actors.Actor
import io.Source
import java.util
import java.io._


class TreeStore(val nnsyms:Int, val ntsyms:Int, val maxwords:Int, val maxnodes:Int, val maxtrees:Int, 
		            val maxlen:Int, val stride:Int, val nthreads:Int, val cdict:CSMat, val docheck:Boolean, val doGPU:Boolean, val crosswire:Boolean) {

  val usemaxsum = true                                               // Maxsum mode (on log probabilities)
  val nnsymsr = 32 * ((nnsyms+31) / 32)                              // Number of non-terminals rounded to a multiple of 32
  val ntsymsr = 32 * ((ntsyms+31) / 32)                              // Number of terminals rounded to a multiple of 32

  val minval = if (usemaxsum) -1e8f else 0f
  val floatmin = 1e-40f

  val fwordsb4     = FMat(ntsymsr, maxwords)                          // Input terminal scores
  val iwordptr     = IMat(maxwords,1)                                 // Pointer in word array to positions of start of each sentence
  val itreeptr     = IMat(maxtrees,1)                                 // Pointer in tree store to positions of start of each sentence
  
  // GPU matrices
  val wordsb4      = if (doGPU) GMat(ntsymsr, maxwords) else null     // Buffer in GPU memory for terminal scores
  val wordsafter   = if (doGPU) GMat(ntsymsr, maxwords) else null     // Terminal scores after unary rule application
  val treesb4      = if (doGPU) GMat(nnsymsr, maxnodes) else null     // Score tree storage before unary rules
  val treesafter   = if (doGPU) GMat(nnsymsr, maxnodes) else null     // Score trees after unary rule application  
  val parsetrees   = if (doGPU) GIMat(3, 2*maxwords) else null        // Parse trees
  val parsevals    = if (doGPU) GMat(1, 2*maxwords)  else null        // node scores
  
  // CPU matrices (r prefix) and copies of GPU matrices in CPU memory (f prefix)
  val rwordsafter  = if (docheck) FMat(ntsymsr, maxwords) else null    // For reference calculations on CPU
  val rtreesb4     = if (docheck) FMat(nnsymsr, maxnodes) else null
  val rtreesafter  = if (docheck) FMat(nnsymsr, maxnodes) else null
  val rparsetrees  = if (docheck) IMat(3, 2*maxwords) else null
  val rparsevals   = if (docheck) FMat(1, 2*maxwords) else null

  val fwordsafter  = if (doGPU) FMat(ntsymsr, maxwords) else null       // Copied from GPU to CPU memory
  val ftreesb4     = if (doGPU) FMat(nnsymsr, maxnodes) else null
  val ftreesafter  = if (doGPU) FMat(nnsymsr, maxnodes) else null
  val fparsetrees  = if (doGPU) IMat(3, 2*maxwords) else null
  val fparsevals   = if (doGPU) FMat(1, 2*maxwords) else null
    
  // Rule matrices.
  // for array "LRZarr", the array is kx3, columns hold respectively parent, left, right symbol ids.
  // L is the left symbol type (non-terminal "n" or terminal "t"),
  // R is the right symbol type (non-terminal "n" or terminal "t"),
  // Z is either "b" for binary rules, or "u" for unary rules.
  //
  // for array "LRZval" the array is kx1, and holds the corresponding rule score. 
  var nnbarr:IMat = null          
  var nnbval:FMat = null           
  var ntbarr:IMat = null
  var ntbval:FMat = null
  var tnbarr:IMat = null
  var tnbval:FMat = null
  var ttbarr:IMat = null
  var ttbval:FMat = null
  var nnpp:IMat = null
  var ntpp:IMat = null
  var tnpp:IMat = null
  var ttpp:IMat = null
  
  var nnuarr:IMat = null
  var nnuval:FMat = null
  var ntuarr:IMat = null
  var ntuval:FMat = null
  var ttuarr:IMat = null
  var ttuval:FMat = null
  var nnupp:IMat = null
  var ntupp:IMat = null
  var ttupp:IMat = null
  
  // These arrays have all the rules fused into binary and unary arrays (Zarr and Zval). 
  // In addition, the dpp and upp arrays are indexed by parent symbol and point to the start
  // of a block of rules indexed by that parent symbol. These are used by Viterbi search. 
  var dpp:GIMat = null
  var barr:GIMat = null
  var bval:GMat = null
  var upp:GIMat = null
  var uarr:GIMat = null
  var uval:GMat = null
  
  var allmap:CSMat = null
  var nmap:CSMat = null
  var tmap:CSMat = null
  var ssmap:CSMat = null
  
  // Buffers (transposed relative to tree store) for binary rule application
  val leftbuff     = if (doGPU) GMat(stride, ntsymsr) else null       // Left score buffer       
  val rightbuff    = if (doGPU) GMat(stride, ntsymsr) else null       // Right score buffer 
  val parbuff      = if (doGPU) GMat(stride, ntsymsr) else null       // Parent score buffer 
 
  // Arrays of indices to and from word and tree storage for copies into the above buffers 
  val gtreeptr     = if (doGPU) GIMat(maxtrees,1) else null
  val gwordptr     = if (doGPU) GIMat(maxwords,1) else null
  val leftptrs     = if (doGPU) GIMat(stride,1) else null
  val rightptrs    = if (doGPU) GIMat(stride,1) else null
  val parptrs      = if (doGPU) GIMat(stride,1) else null
 
  // State variables to track stored number of trees, words, nodes etc
  var ntrees = 0
  var nnodes = 0
  var nwords = 0
  var maxheight = 0
  var vitwork = 0.0
  var totrules = 0L
  itreeptr(0) = 0  
  iwordptr(0) = 0
	
  var parser:CoarseToFineMaxRuleParser = null
  var grammar:Grammar = null

	// This class holds internal state for a particular kernel (NN, NT, TN, or TT). When 
	// its buffers fill, it copies from the tree store into the transposed buffers, applies
	// the kernels and scales, and copies back to tree storage.
  class kState(val parr:GMat, val leftarr:GMat, val rightarr:GMat, val ppos:IMat, val lpos:IMat, val rpos:IMat, 
               val nrules:Int, fctn:(Pointer, Pointer, Pointer, Pointer, Int, Int) => Int) {
	  val ileftptrs       = IMat(stride,1)                             // IMat indices into tree storage, updated on host
	  val irightptrs      = IMat(stride,1)
	  val iparptrs        = IMat(stride,1)
	  val ileftpt         = Pointer.to(ileftptrs.data)                 // JCuda Pointers to the above
	  val irightpt        = Pointer.to(irightptrs.data)	
	  val iparpt          = Pointer.to(iparptrs.data)
	  var ndo = 0

	  // Process a block of data. Copy and transpose from tree storage, run kernels, copy back
	  def doblock(level:Int) = {
		  var err = JCuda.cudaMemcpy(parptrs.data, iparpt, ndo*Sizeof.INT, cudaMemcpyHostToDevice)             // Copy IMat data to GIMats
	    if (err != 0) throw new RuntimeException("CUDA error in doblock:cudaMemcpy1")
		  
		  err = JCuda.cudaMemcpy(leftptrs.data, ileftpt, ndo*Sizeof.INT, cudaMemcpyHostToDevice)
		  if (err != 0) throw new RuntimeException("CUDA error in doblock:cudaMemcpy2")
		  
	    err = pcopytxin(parptrs.data, parr.data, parbuff.data, stride, parr.nrows, ndo)        // Do the data copy/transpose
	    if (err != 0) throw new RuntimeException("CUDA error in doblock:pcopytxin1")
		  
	    err = pcopytxin(leftptrs.data, leftarr.data, leftbuff.data, stride, leftarr.nrows, ndo) 
	    if (err != 0) throw new RuntimeException("CUDA error in doblock:pcopytxin2")
		  
	    if (rightarr != null) {
	      err = JCuda.cudaMemcpy(rightptrs.data, irightpt, ndo*Sizeof.INT, cudaMemcpyHostToDevice)	
	      if (err != 0) throw new RuntimeException("CUDA error in doblock:cudaMemcpy3")
	      
	      err = pcopytxin(rightptrs.data, rightarr.data, rightbuff.data, stride, rightarr.nrows, ndo)
	      if (err != 0) throw new RuntimeException("CUDA error in doblock:pcopytxin3")
	    }    
	    err = fctn(parbuff.data, leftbuff.data, rightbuff.data, null, ndo, nthreads)                     // Apply binary rules
	    if (err != 0) throw new RuntimeException("CUDA error in doblock:kernel")

	    err = pcopytxout(parptrs.data, parbuff.data, parr.data, stride, parr.nrows, ndo)                 // Copy/transpose data out
	    if (err != 0) throw new RuntimeException("CUDA error in doblock:pcopytxout")
	    
	    Mat.nflops += 3L * ndo * nrules
	    totrules += 1L * ndo * nrules
	    ndo = 0
	  }
	  
	  def update(itree:Int, x:Int, y:Int, level:Int, treelen:Int) = {                              // Add the indices for one binary rule application
	  	iparptrs(ndo) = ppos(itree) + treestep(x, level, treelen)
	  	ileftptrs(ndo) = lpos(itree) + treestep(x, y, treelen);
	  	if (rpos != null) {
	  		irightptrs(ndo) = rpos(itree) + treestep(x+y+1, level-y-1, treelen)
	  	}
	  	ndo = ndo + 1
	  	if (ndo >= stride) doblock(level);
	  }
	  
  }
  
  var nnState:kState = null
  var ntState:kState = null
  var tnState:kState = null
  var ttState:kState = null
  var unnState:kState = null
  var untState:kState = null
  var uttState:kState = null
  
  def createKstates = {
	  // Create all the kernel states. Binary rules first
	  nnState = new kState(treesb4, treesafter, treesafter, itreeptr, itreeptr, itreeptr, nnbarr.nrows, nnrules _)
	  ntState = new kState(treesb4, treesafter, wordsafter, itreeptr, itreeptr, iwordptr, ntbarr.nrows, ntrules _)
	  tnState = new kState(treesb4, wordsafter, treesafter, itreeptr, iwordptr, itreeptr, tnbarr.nrows, tnrules _)
	  ttState = new kState(treesb4, wordsafter, wordsafter, itreeptr, iwordptr, iwordptr, ttbarr.nrows, ttrules _)
	  // Unary rule kernels
	  unnState = new kState(treesafter, treesb4, null, itreeptr, itreeptr, null, nnuarr.nrows, nnrulesu _)
	  untState = new kState(treesafter, wordsb4, null, itreeptr, iwordptr, null, ntuarr.nrows, ntrulesu _)
	  uttState = new kState(wordsafter, wordsb4, null, iwordptr, iwordptr, null, ttuarr.nrows, ttrulesu _)
  }
  
  // Clear the state for another parsing run
  def clearState = {
	  itreeptr(0) = 0  
	  iwordptr(0) = 0
	  ntrees = 0
	  nnodes = 0
	  nwords = 0
	  maxheight = 0
	  vitwork = 0.0
	  totrules = 0L
	  if (nnState != null) nnState.ndo = 0
	  if (ntState != null) ntState.ndo = 0
	  if (tnState != null) tnState.ndo = 0
	  if (ttState != null) ttState.ndo = 0
	  if (unnState != null) unnState.ndo = 0
	  if (untState != null) untState.ndo = 0
	  if (uttState != null) uttState.ndo = 0
	  fwordsb4.set(minval)
	  if (doGPU) {
		wordsb4.set(minval)
		wordsafter.set(minval)
		treesb4.set(minval)
		treesafter.set(minval)
		parsetrees.clear
		parsevals.set(minval)
	  }
	  if (docheck) {
	    rwordsafter.set(minval)
	    rtreesb4.set(minval)
	    rtreesafter.set(minval)
	    rparsetrees.clear
	    rparsevals.set(minval)
	  }
  }

  // columns are words, rows are symbols
  // Add a sentence and reserve the score tree space associated with it
  def addtree(syms:FMat) = {
    val len = size(syms,2)
    val newnodes = len*(len+1)/2
    ntrees = ntrees + 1
    nwords = nwords + len
    iwordptr(ntrees,0) = nwords
    nnodes = nnodes + newnodes
    itreeptr(ntrees,0) = nnodes
    maxheight = math.max(maxheight, len)
    if (usemaxsum) {
    	fwordsb4(0->size(syms,1), (nwords-len)->nwords) = ln(max(floatmin, syms))
    } else {
    	fwordsb4(0->size(syms,1), (nwords-len)->nwords) = syms
    }
  }
  
  // Map the x,y coordinates of a tree node to a linear index for that tree
  def treepos(itree:Int, x:Int, y:Int, len:Int) = {
    itreeptr(itree) + treestep(x, y, len)
  }
  
  def treestep(x:Int, y:Int, len:Int) = {
    x + y*len - y*(y-1)/2
  }
  

  
  def chkBinary(level:Int) {
	  var itree = 0
	  while (itree < ntrees) {
	    val sentlen = iwordptr(itree+1)-iwordptr(itree)
	    var x = 0;
	    while (x < sentlen - level) {
	    	var ptree = treepos(itree, x, level, sentlen)
	    	var lword = iwordptr(itree) + x
	    	var rword = iwordptr(itree) + x + level
	    	var y = 0
	    	while (y < level) {
	    	  var ltree = treepos(itree, x, y, sentlen)
	    	  var rtree = treepos(itree, x+y+1, level-y-1, sentlen)
	    	  var i = 0
	    	  while (i < size(nnbarr,1)) {    // Always do NN rules
	    	    rtreesb4(nnbarr(i,0), ptree) = math.max(rtreesb4(nnbarr(i,0), ptree),
	    	        rtreesafter(nnbarr(i,1), ltree) + rtreesafter(nnbarr(i,2), rtree) + nnbval(i,0))   	    
	    	    i += 1
	    	  }
	    	  Mat.nflops += 3L*size(nnbarr,1)
	    	  if (y == level-1) {             // Do NT rules if y at max level
	    	  	i = 0
	    	  	while (i < size(ntbarr,1)) {
	    	  		rtreesb4(ntbarr(i,0), ptree) = math.max(rtreesb4(ntbarr(i,0), ptree),
	    	  		    rtreesafter(ntbarr(i,1), ltree) + rwordsafter(ntbarr(i,2), rword) + ntbval(i,0))  	    
	    	  		i += 1
	    	  	}
	    	  	Mat.nflops += 3L*size(ntbarr,1)
	    	  }
	    	  if (y == 0) {                   // Do TN rules if y == 0
	    	  	i = 0
	    	  	while (i < size(tnbarr,1)) {
	    	  		rtreesb4(tnbarr(i,0), ptree) = math.max(rtreesb4(tnbarr(i,0), ptree),
	    	  		    rwordsafter(tnbarr(i,1), lword) + rtreesafter(tnbarr(i,2), rtree) + tnbval(i,0))  	    
	    	  		i += 1
	    	  	}
	    	  	Mat.nflops += 3L*size(tnbarr,1)
	    	  }
	    	  if (level == 1) {               // Do TT rules if at level 1
	    	  	i = 0
	    	  	while (i < size(ttbarr,1)) {
	    	  		rtreesb4(ttbarr(i,0), ptree) = math.max(rtreesb4(ttbarr(i,0), ptree),
	    	  		    rwordsafter(ttbarr(i,1), lword) + rwordsafter(ttbarr(i,2), rword) + ttbval(i,0))  	    
	    	  		i += 1
	    	  	}
	    	  	Mat.nflops += 3L*size(ttbarr,1)
	    	  }
	    	  y += 1
	    	}
	    	x += 1
	    }
	    itree += 1
	  }
  }
  
  def chkUnary(level:Int) {   
    var itree = 0 
    while (itree < ntrees) {
    	val sentlen = iwordptr(itree+1)-iwordptr(itree)
    	val ptree = treepos(itree, 0, level, sentlen)
    	val pword = iwordptr(itree)
    	var x = 0
    	while (x < sentlen - level) {
    		var i = 0
    		if (level == 0) {
    			while (i < size(ttuarr,1)) {
    				rwordsafter(ttuarr(i,0), pword+x) = math.max(rwordsafter(ttuarr(i,0), pword+x), 
    						fwordsb4(ttuarr(i,1), pword+x) + ttuval(i,0))
    				i += 1
    			}
    		  Mat.nflops += 3L*size(ttuarr,1)
    			i = 0
    			while (i < size(ntuarr,1)) {
    				rtreesafter(ntuarr(i,0), ptree+x) = math.max(rtreesafter(ntuarr(i,0), ptree+x), 
    						fwordsb4(ntuarr(i,1), pword+x) + ntuval(i,0))
    				i += 1
    			}
    		  Mat.nflops += 3L*size(ntuarr,1)
    		} else {
    			while (i < size(nnuarr,1)) {
    				rtreesafter(nnuarr(i,0), ptree+x) = math.max(rtreesafter(nnuarr(i,0), ptree+x), 
    						rtreesb4(nnuarr(i,1), ptree+x) + nnuval(i,0))
    				i += 1
    			}
    			Mat.nflops += 3L*size(nnuarr,1)
    		}
    		x += 1
    	}
    	itree += 1
	  }
  }
  
  // Sort rule arrays into blocks indexed by parent symbol, to speed up Viterbi search.
  def sortparent(barr:IMat, vmat:FMat, nsyms:Int):IMat = {
    val (dmy, ii) = sort2(barr(?,0))
    barr(?,?) = barr(ii,?)
    vmat(?,0) = vmat(ii,0)
    val iret = IMat(nsyms+1,1)
    var i = 1
    var j = 0
    iret(0) = 0
    while (i <= nsyms) {
      while (j < barr.nrows && barr(j,0) < i) {j += 1}
      iret(i) = j 
      i += 1
    }
    iret
  }
  
  
  def loadrules(fname:String, nnsyms0:Int, ntsyms0:Int) = {
    val nnsyms = 32*((nnsyms0+31)/32)
    val ntsyms = 32*((ntsyms0+31)/32)
    val nnrange = (0->nnsyms0)
    val ntrange = ((nnsyms0-3)->(nnsyms0+ntsyms0-3))
    
    nnbarr = load(fname+"dmodel.mat", "nnbarr")
    nnbval = load(fname+"dmodel.mat", "nnbval")
    ntbarr = load(fname+"dmodel.mat", "ntbarr")
    ntbval = load(fname+"dmodel.mat", "ntbval")
    tnbarr = load(fname+"dmodel.mat", "tnbarr")
    tnbval = load(fname+"dmodel.mat", "tnbval")
    ttbarr = load(fname+"dmodel.mat", "ttbarr")
    ttbval = load(fname+"dmodel.mat", "ttbval")
    
    nnpp = sortparent(nnbarr, nnbval, nnsyms0)
    ntpp = sortparent(ntbarr, ntbval, nnsyms0)
    tnpp = sortparent(tnbarr, tnbval, nnsyms0)
    ttpp = sortparent(ttbarr, ttbval, nnsyms0)
    
    allmap = load(fname+"dmodel.mat", "allmap")    
    nmap = allmap(0->nnsyms0,0)
    tmap = allmap((nnsyms0-3)->(nnsyms0+ntsyms0-3),0)
    
    nnuarr = load(fname+"umodel.mat", "nnuarr")
    nnuval = load(fname+"umodel.mat", "nnuval")
    ntuarr = load(fname+"umodel.mat", "ntuarr")
    ntuval = load(fname+"umodel.mat", "ntuval")
    ttuarr = load(fname+"umodel.mat", "ttuarr")
    ttuval = load(fname+"umodel.mat", "ttuval")
    
    nnupp = sortparent(nnuarr, nnuval, nnsyms0)
    ntupp = sortparent(ntuarr, ntuval, nnsyms0)
    ttupp = sortparent(ttuarr, ttuval, ntsyms0)
    
    if (doGPU) {    
    	dpp = GIMat(nnpp on (nnbarr.nrows + (ntpp on (ntbarr.nrows + (tnpp on (tnbarr.nrows + ttpp))))))
    	barr = GIMat(nnbarr on ntbarr on tnbarr on ttbarr)
    	bval = GMat(nnbval on ntbval on tnbval on ttbval)

    	upp = GIMat(nnupp on (nnuarr.nrows + (ntupp on (ntuarr.nrows + ttupp))))
    	uarr = GIMat(nnuarr on ntuarr on ttuarr)
    	uval = GMat(nnuval on ntuval on ttuval)
    }


  }
  
  // Compute inner scores. Work one level at a time, then one tree at a time. This just queues left-right symbol pair
  // jobs for the kState instances. They do the work when their buffers are full. 
  def innerscores() {
  	print("level 0")
  	var itree = 0
  	if (doGPU) {
  		wordsb4 <-- fwordsb4
  		while (itree < ntrees) {                               // Process unary rules on input words
  			var sentlen = iwordptr(itree+1)-iwordptr(itree)
  			var x = 0;
  			while (x < sentlen) {
  				untState.update(itree, x, 0, 0, sentlen)
  				uttState.update(itree, x, 0, 0, sentlen)
  				x += 1
  			}
  			itree += 1
  		}
  		if (untState.ndo > 0) {untState.doblock(0)}            // Finish pending work on input words
  		if (uttState.ndo > 0) {uttState.doblock(0)}
  	}
  	if (docheck && !crosswire) chkUnary(0)                   // Run check of unary rules on input words
  	var level = 1
  	while (level < maxheight) {                              // Process this level 
  	  if (doGPU) {
  	  	itree = 0
  	  	while (itree < ntrees) {                             // Binary rules first
  	  		var sentlen = iwordptr(itree+1)-iwordptr(itree)
  	  		var x = 0;
  	  		while (x < sentlen - level) {                      // Process each node at this level
  	  			var y = 0
  	  			while (y < level) {                              // All splits of the current span
  	  				if (level == 1) {
  	  					ttState.update(itree, x, y, level, sentlen)  // Process two terminal symbols (level 1 only)
  	  				} 
  	  				if (y == 0) {                                  // Left symbol is a terminal
  	  					tnState.update(itree, x, y, level, sentlen)
  	  				} 
  	  				if (y == level-1) {                            // Right symbol is a terminal
  	  					ntState.update(itree, x, y, level, sentlen)
  	  				} 
  	  				nnState.update(itree, x, y, level, sentlen)    // Always do two non-terminals 
  	  				y = y +1
  	  			}    	      
  	  			x = x + 1
  	  		}  	    
  	  		itree = itree+1
  	  	} 
  	  	if (ttState.ndo > 0) {ttState.doblock(level)}        // Almost done with this level, so finish any queued data
  	  	if (tnState.ndo > 0) {tnState.doblock(level)}
  	  	if (ntState.ndo > 0) {ntState.doblock(level)}
  	  	if (nnState.ndo > 0) {nnState.doblock(level)}
  	  	itree = 0
  	  	while (itree < ntrees) {                             // Now run unary rules at this level
  	  		var sentlen = iwordptr(itree+1)-iwordptr(itree)
  	  		var x = 0;
  	  		while (x < sentlen - level) {
  	  			unnState.update(itree, x, level, level, sentlen) 	      
  	  			x += 1
  	  		}
  	  		itree += 1
  	  	}
  	  	if (unnState.ndo > 0) {unnState.doblock(level)}
  	  }
  	  if (docheck && !crosswire) {
  	    chkBinary(level)
  	    chkUnary(level)
  	  }
  	  print(",%d" format level)
  	  level = level+1
  	}
  	if (doGPU) {
  	  fwordsafter <-- wordsafter
  	  ftreesb4    <-- treesb4
  	  ftreesafter <-- treesafter
  	  if (crosswire) {
  	  	rwordsafter <-- wordsafter
  	  	rtreesb4    <-- treesb4
  	  	rtreesafter <-- treesafter
  	  }
  	}
  	println("")
  }
  
  def vitunary(sym:Int, ltree:FMat, pleft:Int, upp:IMat, uarr:IMat, uvals:FMat):(Float, Int) = {
    var imax = 0
    var vmax = minval
    var i = upp(sym)
    while (i < upp(sym+1)) {
      if (ltree(uarr(i,1), pleft) + uvals(i) > vmax) {
        imax = uarr(i,1)
        vmax = ltree(uarr(i,1), pleft) + uvals(i)
      }
      i += 1
    }
    (vmax, imax)    
  }
  
  def vitbinary(sym:Int, ltree:FMat, pleft:Int, rtree:FMat, pright:Int, bpp:IMat, barr:IMat, bvals:FMat):(Float, Float, Float, Int, Int) = {
    var lmax = -1
    var rmax = -1
    var vmax = minval
    var i = bpp(sym)
    while (i < bpp(sym+1)) {
      if (ltree(barr(i,1), pleft) + rtree(barr(i,2), pright) + bvals(i) > vmax) {
        lmax = barr(i,1)
        rmax = barr(i,2)
        vmax = ltree(barr(i,1), pleft) + rtree(barr(i,2), pright) + bvals(i)
      }
      i += 1
    }
    (vmax, if (lmax >= 0) ltree(lmax, pleft) else minval, if (rmax >= 0) rtree(rmax, pright) else minval, lmax, rmax)
  }
  
  // CPU reference Viterbi implementation. 
  // The output trees are stored in a 3xn array. For a sentence of length k, the tree is a block of 3x(2k-1) values. 
  // Each column represents a node. The first row contains the heights of the nodes. 
  // Height is zero-based, so leaf nodes have height zero, and the root node for a sentence of length k has height k-1. 
  // The second row contains the index of the post-unary best symbol for that node.
  // The third row contains the index of the pre-unary best symbol for that node. 
  // The tree is saved in prefix order, and the heights uniquely specify the tree structure. 
  // e.g. for a node at a given position p, its left child is at position p+1. If the left
  // child has height h, the right child will be at position p+2*h. 
  //
  // This representation allows a non-recursive Viterbi search. Each node is processed to find its left and right 
  // children and to add their heights and post-unary scores to the tree array (positions p+1 and p+2*h). 
  // The function walks from left to right through the tree array since nodes to be processed will have already been queued. 
  // The GPU version closely mirrors this routine. 
  
  def vittree(x0:Int, t0:Int, sentlen:Int):Unit = {
  	var tpos = 2*x0                                                              // Address of the root of this tree
  	var x = 0                                                                    // Address of the current word
  	while (tpos < 2*(x0+sentlen)-1) {                                            // Walk through the tree array
  		val height = rparsetrees(0, tpos)                                          // Height of the current tree
  		val isym = rparsetrees(1, tpos)                                            // Symbol at root of current tree
  		if (height > 0) {                                                          // Non-leaf
  			val (uscore, tsym) = vitunary(isym, rtreesb4, t0+treestep(x,height,sentlen), nnupp, nnuarr, nnuval) // Push down through unary rules
  			rparsetrees(2, tpos) = tsym                                              // Save this nodes pre-unary score
  			var y = 0
  			var lmax = -1                                                            // Best left symbol
  			var rmax = -1                                                            // Best right symbol
  			var ylmax = -2                                                           // Height of best left symbol
  			var yrmax = -2                                                           // Height of best right symbol
  			var vmax = minval
  			var vlmax = minval
  			var vrmax = minval
  			while (y < height) {
  				val ts1 = t0+treestep(x, y, sentlen)
  				val ts2 = t0+treestep(x+y+1, height-y-1, sentlen)
  				val (vm0, vl0, vr0, lm0, rm0) = vitbinary(tsym, rtreesafter, ts1, rtreesafter, ts2, nnpp, nnbarr, nnbval)
  				if (vm0 > vmax) {
  					vmax = vm0
  					lmax = lm0
  					rmax = rm0
  					vlmax = vl0
  					vrmax = vr0
  					vrmax = minval
  					ylmax = y
  					yrmax = height-y-1
  				}
  				if (y == 0) {
  					val (vm0, vl0, vr0, lm0, rm0) = vitbinary(tsym, rwordsafter, x+x0, rtreesafter, ts2,  tnpp, tnbarr, tnbval)
  					if (vm0 > vmax) {
  						vmax = vm0
  						lmax = nnsyms-3+lm0
  						rmax = rm0
  						vlmax = vl0
  						vrmax = vr0
  						ylmax = 0
  						yrmax = height-y-1
  					}
  				}
  				if (y == height-1) {
  					val (vm0, vl0, vr0, lm0, rm0) = vitbinary(tsym, rtreesafter, ts1, rwordsafter, x+x0+y+1, ntpp, ntbarr, ntbval) 
  					if (vm0 > vmax) {
  						vmax = vm0
  						lmax = lm0
  						rmax = nnsyms-3+rm0
  						vlmax = vl0
  						vrmax = vr0
  						ylmax = y
  						yrmax = 0
  					}
  				}
  				if (1 == height) {
  					val (vm0, vl0, vr0, lm0, rm0) = vitbinary(tsym, rwordsafter, x+x0, rwordsafter, x+x0+y+1, ttpp, ttbarr, ttbval) 
  					if (vm0 > vmax) {
  						vmax = vm0
  						lmax = nnsyms-3+lm0
  						rmax = nnsyms-3+rm0
  						vlmax = vl0
  						vrmax = vr0
  						ylmax = 0
  						yrmax = 0
  					}
  				}        
  				y += 1
  			}
  			val ymax = math.max(0, ylmax)
  			rparsetrees(0, tpos+1) = ylmax
  			rparsetrees(1, tpos+1) = lmax
  			rparsevals(tpos+1) = vlmax
  			rparsetrees(0, tpos+2*(ymax+1)) = yrmax
  			rparsetrees(1, tpos+2*(ymax+1)) = rmax
  			rparsevals(tpos+2*(ymax+1)) = vrmax     
  		} else {                                                               // height == 0, advance the leaf pointer
  		  if (isym < nnsyms-3) {                                               // max symbol was a non-terminal
  		  	val (uscore, tsym) = vitunary(isym, fwordsb4, x+x0, ntupp, ntuarr, ntuval) // back down through unary rules
  		  	rparsetrees(2, tpos) = tsym+nnsyms-3
  		  } else {                                                             // max symbol was a terminal
  		    val (uscore, tsym) = vitunary(isym-nnsyms+3, fwordsb4, x+x0, ttupp, ttuarr, ttuval)  	
  		    rparsetrees(2, tpos) = tsym+nnsyms-3
  		  }
  		  x += 1  		  
  		}
  		tpos += 1                                                              // Advance the tree pointer
  	}
  }
  
  def viterbi(rootsym:Int) = {
    if (docheck) {
      var itree = 0
      while (itree < ntrees) {                                                // perform Viterbi search over each tree
      	val iwbase = iwordptr(itree)
      	var sentlen = iwordptr(itree+1) - iwbase                   
      	rparsetrees(1, iwbase*2) = rootsym                                    // Initialize the root symbol
      	rparsetrees(0, iwbase*2) = sentlen-1                                  // Initialize root tree height                                                       // Root tree score
      	vittree(iwbase, itreeptr(itree), sentlen)
        itree += 1
      }
    }
    if (doGPU) {
      gwordptr <-- iwordptr
      gtreeptr <-- itreeptr
      val err = viterbiGPU(nnsyms, ntsyms, nnsymsr, ntsymsr, ntrees, rootsym,
                           wordsb4.data, wordsafter.data, treesb4.data, treesafter.data, 
                           gwordptr.data, gtreeptr.data, parsetrees.data, parsevals.data, 
                           dpp.data, barr.data, bval.data, barr.nrows,
                           upp.data, uarr.data, uval.data, uarr.nrows)
      if (err != 0) throw new RuntimeException("CUDA error in viterbiGPU")
      fparsetrees <-- parsetrees
      fparsevals <-- parsevals
    }
  }
} 

import edu.berkeley.nlp.PCFGLA._

class BIDParser(maxlen: Int = 40,
                docheck: Boolean = false,
                doGPU: Boolean = true,
                crosswire: Boolean = false,
                corpusPath:String="/data/wsj/wsj/", 
                grammarPath:String="grammar/eng_sm6.gr", 
                symbolsPath:String="grammar/") {
  val cdict:CSMat = load(symbolsPath+"dmodel.mat", "allmap")
  val corpus: Corpus = new Corpus(corpusPath, TreeBankType.WSJ, 1.0, true)
  val pData = ParserData.Load(grammarPath)
  if (pData==null) {
  	System.out.println("Failed to load grammar from file"+grammarPath+".")
  	System.exit(1)
  }
  val grammar = pData.getGrammar
  grammar.splitRules()
  val lexicon = pData.getLexicon
  val parser =  new CoarseToFineMaxRuleParser(grammar, lexicon, 1.0, -1, true, false, false, false, false, false, true) {
   // so hacky
    def rootScore = iScore(0)(length)(0)(0)

    def treeAndScore(sent: IndexedSeq[String]):(Tree[String],Double) = {
      val tree = getBestParse(sent.asJava)
      tree->rootScore
    }
  }

  def berkeleyParses(trees: IndexedSeq[IndexedSeq[String]]) = {
     trees.map(parser.treeAndScore _)
  }

  private val sdict = cdict.data

  val substateIds = Array.ofDim[Array[Int]](grammar.numStates)
  val ssmap = CSMat(cdict.length,1)
  var id = 0
  for(i <- 0 until grammar.numSubStates.length) {
    substateIds(i) = new Array[Int](grammar.numSubStates(i))
    for(j <- 0 until grammar.numSubStates(i)) {
      val sym = grammar.getTagNumberer.`object`(i) + "_" + j
      val ix = sdict.indexOf(sym)
      substateIds(i)(j) = ix
      val obj = grammar.getTagNumberer.`object`(i).toString
      ssmap(ix) = obj
      id += 1
    }
  }

  private val nnsyms = sdict.indexOf("EX_1")+1
  private val rootpos = sdict.indexOf("ROOT_0")
  // Overlap by three symbols, since ROOT_0, EX_0 and EX_1 are both terminal and non-terminal
  private val ntsyms = sdict.length - nnsyms + 3
  private val (ffrac, gmem, gtmem) = GPUmem
  private val maxnodes = (32 * (gmem/12/nnsyms/32)).toInt
  private val maxwords = 32 * (maxnodes/10/32)
  private val maxtrees = maxwords/8
  private val stride = 8192
  private val nthreads = 1024

  def parse(sentences: IndexedSeq[IndexedSeq[String]]) = {
    getBatches(sentences).map(createBatch _).map(parseBatch _).foldLeft(IndexedSeq.empty[Tree[String]])(_ ++ _)
  }

  private def parseBatch(batch: Batch):IndexedSeq[Tree[String]] = {
    import batch._
    val ts = new TreeStore(nnsyms, ntsyms, maxwords, maxnodes, maxtrees, maxlen, stride, nthreads, cdict, docheck, doGPU, crosswire)
    ts.clearState
    ts.loadrules(symbolsPath, nnsyms, ntsyms)

    ts.createKstates
    ts.ssmap = ssmap
    posScores.foreach(ssc => ts.addtree(ssc))
    ts.innerscores
     ts.viterbi(rootpos)
    val trees = for(i <- 0 until numSentences) yield {
      var tree = if (doGPU) {
      	BIDParser.getTree(i, ts, tsents, true)
      }	else {
      	BIDParser.getRTree(i, ts, tsents, true)
      } 
      tree = new Trees.FunctionNodeStripper().transformTree(tree)
      tree = TreeAnnotations.unAnnotateTree(tree)
      tree
    }
    trees
  }


  case class Batch(tsents: Array[CSMat],
                   posScores: Array[FMat]) {
    def numSentences = tsents.length
  }

  private def createBatch(sentences: IndexedSeq[(IndexedSeq[String])]): Batch = {

    val tsents = new Array[CSMat](sentences.size)
    val slist = ArrayBuffer[FMat]()
    for( (s, i) <- sentences.zipWithIndex) {
      val ctmp = CSMat(s.size,1)
      tsents(i) = ctmp
      for (i <- 0 until s.size)  {ctmp(i) = s(i)}
      // pos -> coarse Sym -> refinement -> score
      val scores: mutable.Buffer[Array[Array[Double]]] = parser.getTagScores(s.asJava).asScala
      val ssc = FMat(ntsyms, scores.length)
      for ( (score,pos) <- scores.zipWithIndex) {
        for(i <- 0 until score.length if score(i) != null; j <- 0 until score(i).length) {
          ssc(substateIds(i)(j)-nnsyms+3, pos) = score(i)(j).toFloat
        }
      }
      slist += ssc

    }

    Batch(tsents, slist.toArray)
  }

  private def getBatches(sentences: IndexedSeq[IndexedSeq[String]]): IndexedSeq[IndexedSeq[(IndexedSeq[String])]] = {
    val result = ArrayBuffer[IndexedSeq[(IndexedSeq[String])]]()
    var current = ArrayBuffer[(IndexedSeq[String])]()
    var nnodes = 0
    var nwords = 0
    for( (s, i) <- sentences.zipWithIndex) {
      val len = s.length
      if (nwords + len > maxwords || nnodes + len*(len+1)/2 > maxnodes) {
        result += current
        current = ArrayBuffer()
      }

      nwords += len
      nnodes += len*(len+1)/2
      current += s
    }

    if(current.nonEmpty) result += current
    result
  }
}


object ParseText {
  import BIDParser._
  def main(args: Array[String]) {
    val parser = loadParser(args(0), args(1))

    val fin = if(args.length < 3) System.in else new FileInputStream(args(2))
    // TODO: currently slurps whole file

    val source = Source.fromInputStream(fin).mkString
    import java.text.BreakIterator
    val break = BreakIterator.getSentenceInstance()
    break.setText(source)
    var start = break.first()
    var end = break.next()
    val sentences = new ArrayBuffer[IndexedSeq[String]]()
    while(end != BreakIterator.DONE) {
      val sent = source.substring(start, end).trim.split("\n\n").map(_.replaceAll("\\s"," ").trim).filter(_.nonEmpty).toIndexedSeq
      sentences ++= sent.map(s => new PTBLineLexer().tokenizeLine(s).asScala.toIndexedSeq)
      start = end
      end = break.next
    }

    parser.parse(sentences) foreach println
  }


}


object BIDParser {
   
  def main(args:Array[String]):Unit = {
    runParser(args(0).toInt, args(1).toInt, args(2).toBoolean, args(3).toBoolean, args(4).toBoolean, args(5).toBoolean, args(6), args(7), args(8))
  }

  def runParser(maxlen:Int=30, maxsents:Int=10000, docheck:Boolean=false, doGPU:Boolean=true, crosswire:Boolean=false, parallel:Boolean=false,
      corpusPath:String="/data/wsj/wsj/", 
      grammarPath:String="grammar/eng_sm6.gr", 
      symbolsPath:String="grammar/") = {
    
    val corpus: Corpus = new Corpus(corpusPath, TreeBankType.WSJ, 1.0, true)
    // TODO: maxlen is usually computed without punctuation.
    val testTrees = corpus.getDevTestingTrees.asScala.filter(_.getTerminalYield.size <= maxlen).asJava

    val eval = new EnglishPennTreebankParseEvaluator.LabeledConstituentEval[String](new util.HashSet[String](util.Arrays.asList("ROOT","PSEUDO")),
        new util.HashSet(util.Arrays.asList("''", "``", ".", ":", ",")))
    System.out.println("The computed F1,LP,LR scores are just a rough guide. They are typically 0.1-0.2 lower than the official EVALB scores.")

    val parser = loadParser(grammarPath, symbolsPath, docheck, doGPU, crosswire)
    val toDecode = testTrees.asScala.map(_.getTerminalYield.asScala.toIndexedSeq).toIndexedSeq.take(maxsents)
    flip
    val decoded = parser.parse(toDecode)
    val ff = gflop
    val tt = ff._2
    val f1 = eval.evaluateMultiple(decoded.asJava, testTrees, new PrintWriter(System.out))
    println("time= %f secs, %f sents/sec, %f gflops".format(tt, testTrees.size / tt, ff._1) )
    println("GPU parsesr F1: " + f1)

    println("getting berkeley parser parses... this will take a while.")
    val berkeleyParses = parser.berkeleyParses(toDecode)
    val berkeleyf1 = eval.evaluateMultiple(berkeleyParses.map(_._1).asJava, testTrees, new PrintWriter(System.out))
    println("Berkeley F1: " + berkeleyf1)

    (decoded, testTrees.asScala.toIndexedSeq, berkeleyParses)
  }


  def loadParser(grammarPath: String="grammar/eng6_sm6.gr", symbolsPath: String="grammar/", docheck: Boolean = false, doGPU: Boolean = true, crosswire: Boolean = false) = {
    val pData = ParserData.Load(grammarPath)
    if (pData == null) {
      throw new RuntimeException("Failed to load grammar from file " + grammarPath + ".")
    }
    val grammar = pData.getGrammar
    grammar.splitRules()
    val lexicon = pData.getLexicon
    Numberer.setNumberers(pData.getNumbs)

    new BIDParser(docheck = docheck, doGPU = doGPU, crosswire = crosswire)
  }


  def runParser_old(maxlen:Int=30, maxsents:Int=10000, docheck:Boolean=false, doGPU:Boolean=true, crosswire:Boolean=false, parallel:Boolean=false,
      corpusPath:String="/data/wsj/wsj/", 
      grammarPath:String="grammar/eng_sm6.gr", 
      symbolsPath:String="grammar/") = {
    
    val corpus: Corpus = new Corpus(corpusPath, TreeBankType.WSJ, 1.0, true)
    val testTrees = corpus.getDevTestingTrees

    val eval = new EnglishPennTreebankParseEvaluator.LabeledConstituentEval[String](new util.HashSet[String](util.Arrays.asList("ROOT","PSEUDO")),
    		new util.HashSet(util.Arrays.asList("''", "``", ".", ":", ",")))
    System.out.println("The computed F1,LP,LR scores are just a rough guide. They are typically 0.1-0.2 lower than the official EVALB scores.")

    val pData = ParserData.Load(grammarPath)
    if (pData==null) {
    	System.out.println("Failed to load grammar from file"+grammarPath+".")
    	System.exit(1)
    }
    val grammar = pData.getGrammar
    grammar.splitRules()
    val lexicon = pData.getLexicon
    val spanPredictor = pData.getSpanPredictor
    Numberer.setNumberers(pData.getNumbs)


    val vitParse = true
    val cdict:CSMat = load(symbolsPath+"dmodel.mat", "allmap")
    val sdict = cdict.data

    val substateIds = Array.ofDim[Array[Int]](grammar.numStates)
    val ssmap = CSMat(cdict.length,1)
    var id = 0
    for(i <- 0 until grammar.numSubStates.length) {
    	substateIds(i) = new Array[Int](grammar.numSubStates(i))
    	for(j <- 0 until grammar.numSubStates(i)) {
    		val sym = grammar.getTagNumberer.`object`(i) + "_" + j
    		val ix = sdict.indexOf(sym)
    		substateIds(i)(j) = ix
    		ssmap(ix) = grammar.getTagNumberer.`object`(i).toString
    		id += 1
    	}
    }
    
    val nnsyms = sdict.indexOf("EX_1")+1
    val rootpos = sdict.indexOf("ROOT_0")
    // Overlap by three symbols, since ROOT_0, EX_0 and EX_1 are both terminal and non-terminal 
    val ntsyms = sdict.length - nnsyms + 3
    val (ffrac, gmem, gtmem) = GPUmem
    val maxnodes = (32 * (gmem/12/nnsyms/32)).toInt
    val maxwords = 32 * (maxnodes/10/32)
    val maxtrees = maxwords/8
    val stride = 8192
    val nthreads = 1024
    val parser =  new CoarseToFineMaxRuleParser(grammar, lexicon, 1.0, -1, true, false, false, false, false, false, false);

    val tsents = new Array[CSMat](testTrees.size)
    val slist = ListBuffer[FMat]()
    //parser.binarization = pData.getBinarization	  
    val testTreesIterator = testTrees.iterator()
    var nwords = 0
    var nsentences = 0
    var nnodes = 0
    var done = false
    while(testTreesIterator.hasNext && !done) {
    	val testTree = testTreesIterator.next()
    	val testSentence = testTree.getYield
    	val len = testSentence.size
    	if (len <= maxlen) {
    		if (nwords + len > maxwords || nnodes + len*(len+1)/2 > maxnodes || nsentences >= maxsents) {
    			done = true
    		} else {
    			val ctmp = CSMat(testSentence.size,1)
    			tsents(nsentences) = ctmp
    			var tsiter = testSentence.iterator
    			for (i <- 0 until testSentence.size)  {ctmp(i) = tsiter.next}
    			// pos -> coarse Sym -> refinement -> score
    			val scores: mutable.Buffer[Array[Array[Double]]] = parser.getTagScores(testSentence).asScala
    			val ssc = FMat(ntsyms, scores.length)
    			for ( (score,pos) <- scores.zipWithIndex) {
    				for(i <- 0 until score.length if score(i) != null; j <- 0 until score(i).length) {
    					ssc(substateIds(i)(j)-nnsyms+3, pos) = score(i)(j).toFloat
    				}
    			} 
    			slist += ssc
    			nsentences += 1
    			nwords += len
    			nnodes += len*(len+1)/2
    		}
    	}
    }
    val nGPUthreads = if (parallel) Mat.hasCUDA else 1
    var tss = new Array[TreeStore](nGPUthreads)
    for (ithread <- 0 until nGPUthreads) { 
      setGPU(ithread)
      val ts = new TreeStore(nnsyms, ntsyms, maxwords, maxnodes, maxtrees, maxlen, stride, nthreads, cdict, docheck, doGPU, crosswire)
      tss(ithread) = ts
      ts.clearState
      ts.loadrules(symbolsPath, nnsyms, ntsyms)
      ts.createKstates
      ts.grammar = grammar
      ts.parser = parser
      ts.ssmap = ssmap
      slist.foreach(ssc => ts.addtree(ssc))
    }
    flip
    var tdone = izeros(nGPUthreads,1)
    var vittime = zeros(nGPUthreads,1)
    for (ithread <- 0 until nGPUthreads) { 
      Actor.actor { 
        setGPU(ithread)
        val ts = tss(ithread)
        ts.innerscores
        val (gf, tt) = gflop
        ts.viterbi(rootpos)
        val (gf2, tt2) = gflop
        vittime(ithread) = tt2-tt
        tdone(ithread) = 1
      }
    }
    while (mini(tdone).v == 0) {Thread.`yield`}
    var allrules = 0L
    for (ithread <- 0 until nGPUthreads) {allrules += tss(ithread).totrules}
    val ff = gflop
    val tt = ff._2
    nsentences *= nGPUthreads
    nwords *= nGPUthreads
    nnodes *= nGPUthreads
    println("maxlen=%d, nsentences=%d, nwords=%d, nnodes=%d, maxsents=%d, maxwords=%d, maxnodes=%d\ntime= %f secs, %f sents/sec, %f Brules/sec, %f gflops," +
        " vittime= %f" format 
	    (maxlen, nsentences, nwords, nnodes, maxtrees, maxwords, maxnodes, tt, nsentences/tt, allrules/tt/1e9, ff._1, mean(vittime).dv))
    (tss, testTrees, tsents)
  }
  
  def printTree(itree:Int, ts:TreeStore, tsents:Array[CSMat]) = {
    val iword = ts.iwordptr(itree)
    printSubTree(itree, 2*iword, iword, ts.fparsetrees, ts.fparsevals, ts, tsents)    
  }
  
  def printRTree(itree:Int, ts:TreeStore, tsents:Array[CSMat]) = {
    val iword = ts.iwordptr(itree)
    printSubTree(itree, 2*iword, iword, ts.rparsetrees, ts.rparsevals, ts, tsents)    
  }
  
  def printSubTree(itree:Int, inode:Int, iword:Int, trees:IMat, tvals:FMat, ts:TreeStore, tsents:Array[CSMat]):Unit = {
    val height = trees(0, inode)
    val isym = trees(1, inode)
    print("(")
    print(ts.ssmap(isym)+" ")
    if (height > 0) {
      printSubTree(itree, inode+1, iword, trees, tvals, ts, tsents)
      print(" ")
      val lheight = trees(0, inode+1)
      printSubTree(itree, inode+2*lheight+2, iword+lheight+1, trees, tvals, ts, tsents)
    } else {
      print(tsents(itree)(iword-ts.iwordptr(itree)))
    }
    print(")")
  }
  
   def printTree(pw:PrintWriter, itree:Int, ts:TreeStore, tsents:Array[CSMat], binary:Boolean) = {
    val iword = ts.iwordptr(itree)
    printSubTree(pw, itree, 2*iword, iword, ts.fparsetrees, ts.fparsevals, ts, tsents, binary)    
  }
  
  def printRTree(pw: PrintWriter, itree:Int, ts:TreeStore, tsents:Array[CSMat], binary:Boolean) = {
    val iword = ts.iwordptr(itree)
    printSubTree(pw, itree, 2*iword, iword, ts.rparsetrees, ts.rparsevals, ts, tsents, binary)    
  }
  
  def printSubTree(pw: PrintWriter, itree:Int, inode:Int, iword:Int, trees:IMat, tvals:FMat, ts:TreeStore, tsents:Array[CSMat], binary:Boolean):Unit = {
    import pw._
    val height = trees(0, inode)
    val isym = trees(1, inode)
    val jsym = trees(2, inode)
    val dounary = 
      if (isym != jsym) { 
        print("("+ts.ssmap(isym)+" ")
        true
      } else false
    val jstr = ts.ssmap(jsym)
    val atsym = jstr.startsWith("@",0)
    if (binary || !atsym) print("("+jstr+" ")
    if (height > 0) {
      printSubTree(pw, itree, inode+1, iword, trees, tvals, ts, tsents, binary)
      print(" ")
      val lheight = trees(0, inode+1)
      printSubTree(pw, itree, inode+2*lheight+2, iword+lheight+1, trees, tvals, ts, tsents, binary)
    } else {
      print(tsents(itree)(iword-ts.iwordptr(itree)))
    }
    if (binary || !atsym) print(")")
    if (dounary) print(")")
  }

  
  def getRTree(i: Int, ts: TreeStore, tsents: Array[CSMat], binary: Boolean = true):Tree[String] = {
    val sw = new StringWriter()
    val pw = new PrintWriter(sw)
    printRTree(pw, i, ts, tsents, binary)
    pw.close()
    val t = Trees.PennTreeReader.parseEasy(sw.toString)
    assert(t ne null, sw.toString)
    t
  }
  
  def getTree(i: Int, ts: TreeStore, tsents: Array[CSMat], binary: Boolean = true):Tree[String] = {
    val sw = new StringWriter()
    val pw = new PrintWriter(sw)
    printTree(pw, i, ts, tsents, binary)
    pw.close()
    val t = Trees.PennTreeReader.parseEasy(sw.toString)
    assert(t ne null, sw.toString)
    t
  }

} 







