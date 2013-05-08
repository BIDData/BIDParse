package BIDParse
import BIDMat.{Mat, DMat, FMat, IMat, CMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat}
import BIDMat.SciFunctions._
import BIDMat.MatFunctions._


import jcuda._
import jcuda.runtime.JCuda
import jcuda.runtime.cudaMemcpyKind._
import jcuda.jcublas.JCublas;
import edu.berkeley.bid.CUMAT
import edu.berkeley.bid.CBLAS
import edu.berkeley.bid.PARSE._
import edu.berkeley.nlp.syntax.Tree
import edu.berkeley.nlp.PCFGLA.Corpus.TreeBankType
import scala.collection.JavaConverters._
import edu.berkeley.nlp.parser.EnglishPennTreebankParseEvaluator
import java.util
import edu.berkeley.nlp.util.Numberer
import collection.mutable
import collection.mutable.ArrayBuffer
import io.Source


class TreeStore(val nnsyms0:Int, val ntsyms0:Int, val maxwords:Int, val maxnodes:Int, val maxtrees:Int, 
		            val maxlen:Int, val stride:Int, val nthreads:Int, val cdict:CSMat) {

  val docheck = true
  val doGPU = true
  val usemaxsum = true
  val nnsyms = 32 * ((nnsyms0+31) / 32)
  val ntsyms = 32 * ((ntsyms0+31) / 32)

  val minval = if (usemaxsum) -1e8f else 0f
  val floatmin = 1e-40f
  // Storage for input words and score trees
  val fwordsb4     = FMat(ntsyms, maxwords)            // Accumulate terminal scores here
  val wordsb4      = GMat(ntsyms, maxwords)            // Buffer in GPU memory for terminal scores
  val wordsafter   = GMat(ntsyms, maxwords)            // Terminal scores after unary rule application
  val treesb4      = GMat(nnsyms, maxnodes)            // Score tree storage before unary rules
  val treesafter   = GMat(nnsyms, maxnodes)            // Score trees after unary rule application
  
  val fwordsafter  = if (docheck) FMat(ntsyms, maxwords) else null // Copies of GPU matrices
  val ftreesb4     = if (docheck) FMat(nnsyms, maxnodes) else null
  val ftreesafter  = if (docheck) FMat(nnsyms, maxnodes) else null
  val rwordsafter  = if (docheck) FMat(ntsyms, maxwords) else null // For reference calculations
  val rtreesb4     = if (docheck) FMat(nnsyms, maxnodes) else null
  val rtreesafter  = if (docheck) FMat(nnsyms, maxnodes) else null
    
  var nndarr:IMat = null
  var nnval:FMat = null
  var ntdarr:IMat = null
  var ntval:FMat = null
  var tndarr:IMat = null
  var tnval:FMat = null
  var ttdarr:IMat = null
  var ttval:FMat = null
  
  var nnuarr:IMat = null
  var nnuval:FMat = null
  var ntuarr:IMat = null
  var ntuval:FMat = null
  var ttuarr:IMat = null
  var ttuval:FMat = null
  
  var allmap:CSMat = null
  var nmap:CSMat = null
  var tmap:CSMat = null
  
  // Buffers (transposed relative to tree store) for binary rule application
  val leftbuff     = GMat(stride, ntsyms)              // Left score buffer       
  val rightbuff    = GMat(stride, ntsyms)              // Right score buffer 
  val parbuff      = GMat(stride, ntsyms)              // Parent score buffer 
 
  // Arrays of indices to and from word and tree storage for copies into the above buffers 
  val itreeptr     = IMat(maxtrees,1)
  val iwordptr     = IMat(maxwords,1)
 	val leftptrs     = GIMat(stride,1) 
 	val rightptrs    = GIMat(stride,1)
  val parptrs      = GIMat(stride,1)
 
  // State variables to track stored number of trees, words, nodes etc
  var ntrees = 0
  var nnodes = 0
  var nwords = 0
  var maxheight = 0
  itreeptr(0) = 0  
	iwordptr(0) = 0

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

  case class OutData(wordsb4:FMat, wordsafter:FMat, treesb4:FMat, treesafter:FMat)

  def copyOut: OutData = {
    OutData(fwordsb4, FMat(wordsafter), FMat(treesb4), FMat(treesafter))
  }
  
  // Create all the kernel states. Binary rules first
  val nnState = new kState(treesb4, treesafter, treesafter, itreeptr, itreeptr, itreeptr, 373375, nnrules _)
  val ntState = new kState(treesb4, treesafter, wordsafter, itreeptr, itreeptr, iwordptr, 334338, ntrules _)
  val tnState = new kState(treesb4, wordsafter, treesafter, itreeptr, iwordptr, itreeptr, 417022, tnrules _)
  val ttState = new kState(treesb4, wordsafter, wordsafter, itreeptr, iwordptr, iwordptr, 600835, ttrules _)
  // Unary rule kernels
  val unnState = new kState(treesafter, treesb4, null, itreeptr, itreeptr, null, 35343, nnrulesu _)
  val untState = new kState(treesafter, wordsb4, null, itreeptr, iwordptr, null, 95253, ntrulesu _)
  val uttState = new kState(wordsafter, wordsb4, null, iwordptr, iwordptr, null, 684, ttrulesu _)
  
  // Clear the state for another parsing run
  def clearState = {
	  itreeptr(0) = 0  
	  iwordptr(0) = 0
	  ntrees = 0
	  nnodes = 0
	  nwords = 0
	  maxheight = 0
	  nnState.ndo = 0
	  ntState.ndo = 0
	  tnState.ndo = 0
	  ttState.ndo = 0
	  unnState.ndo = 0
	  untState.ndo = 0
	  uttState.ndo = 0
	  fwordsb4.set(minval)
	  wordsb4.set(minval)
	  wordsafter.set(minval)
	  treesb4.set(minval)
	  treesafter.set(minval)
	  if (docheck) {
	    rwordsafter.set(minval)
	    rtreesb4.set(minval)
	    rtreesafter.set(minval)
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
	    	  while (i < size(nndarr,1)) {    // Always do NN rules
	    	    rtreesb4(nndarr(i,0), ptree) = math.max(rtreesb4(nndarr(i,0), ptree),
	    	        rtreesafter(nndarr(i,1), ltree) + rtreesafter(nndarr(i,2), rtree) + nnval(i,0))   	    
	    	    i += 1
	    	  }
	    	  Mat.nflops += 3L*size(nndarr,1)
	    	  if (y == level-1) {             // Do NT rules if y at max level
	    	  	i = 0
	    	  	while (i < size(ntdarr,1)) {
	    	  		rtreesb4(ntdarr(i,0), ptree) = math.max(rtreesb4(ntdarr(i,0), ptree),
	    	  		    rtreesafter(ntdarr(i,1), ltree) + rwordsafter(ntdarr(i,2), rword) + ntval(i,0))  	    
	    	  		i += 1
	    	  	}
	    	  	Mat.nflops += 3L*size(ntdarr,1)
	    	  }
	    	  if (y == 0) {                   // Do TN rules if y == 0
	    	  	i = 0
	    	  	while (i < size(tndarr,1)) {
	    	  		rtreesb4(tndarr(i,0), ptree) = math.max(rtreesb4(tndarr(i,0), ptree),
	    	  		    rwordsafter(tndarr(i,1), lword) + rtreesafter(tndarr(i,2), rtree) + tnval(i,0))  	    
	    	  		i += 1
	    	  	}
	    	  	Mat.nflops += 3L*size(tndarr,1)
	    	  }
	    	  if (level == 1) {               // Do TT rules if at level 1
	    	  	i = 0
	    	  	while (i < size(ttdarr,1)) {
	    	  		rtreesb4(ttdarr(i,0), ptree) = math.max(rtreesb4(ttdarr(i,0), ptree),
	    	  		    rwordsafter(ttdarr(i,1), lword) + rwordsafter(ttdarr(i,2), rword) + ttval(i,0))  	    
	    	  		i += 1
	    	  	}
	    	  	Mat.nflops += 3L*size(ttdarr,1)
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
    	val ptree = treepos(itree, level, level, sentlen)
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
  
  
  
  def loadrules(fname:String, nnsyms0:Int, ntsyms0:Int) = {
    val nnsyms = 32*((nnsyms0+31)/32)
    val ntsyms = 32*((ntsyms0+31)/32)
    val nnrange = (0->nnsyms0)
    val ntrange = ((nnsyms0-3)->(nnsyms0+ntsyms0-3))
    
    nndarr = load(fname+"dmodel.mat", "nndarr")
    nnval = load(fname+"dmodel.mat", "nnval")
    ntdarr = load(fname+"dmodel.mat", "ntdarr")
    ntval = load(fname+"dmodel.mat", "ntval")
    tndarr = load(fname+"dmodel.mat", "tndarr")
    tnval = load(fname+"dmodel.mat", "tnval")
    ttdarr = load(fname+"dmodel.mat", "ttdarr")
    ttval = load(fname+"dmodel.mat", "ttval")
    
    allmap = load(fname+"dmodel.mat", "allmap")
    nmap = allmap(0->nnsyms0,0)
    tmap = allmap((nnsyms0-3)->(nnsyms0+ntsyms0-3),0)
    
    nnuarr = load(fname+"umodel.mat", "nnuarr")
    nnuval = load(fname+"umodel.mat", "nnuval")
    ntuarr = load(fname+"umodel.mat", "ntuarr")
    ntuval = load(fname+"umodel.mat", "ntuval")
    ttuarr = load(fname+"umodel.mat", "ttuarr")
    ttuval = load(fname+"umodel.mat", "ttuval")

  }
  
  // Compute inner scores. Work one level at a time, then one tree at a time. This just queues left-right symbol pair
  // jobs for the kState instances. They do the work when their buffers are full. 
  def innerscores() {
  	print("level 0")
  	var itree = 0
  	if (doGPU) {
  		while (itree < ntrees) {                               // Process unary rules on input words
  			var sentlen = iwordptr(itree+1)-iwordptr(itree)
  			var x = 0;
  			while (x < sentlen) {
  				uttState.update(itree, x, 0, 0, sentlen)
  				untState.update(itree, x, 0, 0, sentlen)
  				x += 1
  			}
  			itree += 1
  		}
  		wordsb4 <-- fwordsb4
  		if (untState.ndo > 0) {untState.doblock(0)}            // Finish pending work on input words
  		if (uttState.ndo > 0) {uttState.doblock(0)}
  	}
  	if (docheck) chkUnary(0)                                 // Run check of unary rules on input words
  	var level = 1
//  	maxheight = 1
  	while (level <= maxheight) {                             // Process this level 
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
  	  if (docheck) {
  	    chkBinary(level)
  	    chkUnary(level)
  	  }
  	  print(", %d" format level)
  	  level = level+1
  	}
  	if (docheck && doGPU) {
  	  fwordsafter <-- wordsafter
  	  ftreesb4    <-- treesb4
  	  ftreesafter <-- treesafter
  	}
  	println("")
  }

} 

import edu.berkeley.nlp.PCFGLA._

object BIDParser {
  
  def defaults = {
    runParser("c:\\data\\Grammar\\eng_sm6.gr",
        "c:\\data\\wsj\\wsj\\",
        "c:\\data\\Grammar\\",
        20,
        4)
  }
  
	def main(args:Array[String]):Unit = {
		val berkeleyGrammar = args(0)
		val cpath = args(1)
		val sympath = args(2)
		var maxlen = 36
		var maxsents = 100000
		if (args.size > 3) maxlen = args(3).toInt
		if (args.size > 4) maxsents = args(4).toInt
		runParser(args(0), args(1), args(2), maxlen, maxsents)
	}

  def runParser(grammarPath:String, corpusPath:String, symbolsPath:String, maxlen:Int, maxsents:Int) = {
  	val corpus: Corpus = new Corpus(corpusPath, TreeBankType.WSJ, 1.0, true)
    val testTrees = corpus.getDevTestingTrees
  // testTrees = corpus.getTestTrees

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
	  val parser = new CoarseToFineMaxRuleParser(grammar, lexicon, 1.0, -1, true, false, false, false, false, false, false);
  //parser.binarization = pData.getBinarization

	  val sdict:IndexedSeq[String] =(
	  		Source.fromFile(symbolsPath+"fulldict.txt")
	  		.getLines()
	  		.map(_.trim)
	  		.toIndexedSeq
	  )
	  val cdict = CSMat(sdict.length,1)
	  for (i <- 0 until sdict.length) {cdict(i,0) = sdict(i)}

	  val substateIds = Array.ofDim[Array[Int]](grammar.numStates)
	  var id = 0
	  for(i <- 0 until grammar.numSubStates.length) {
	  	substateIds(i) = new Array[Int](grammar.numSubStates(i))
	  	for(j <- 0 until grammar.numSubStates(i)) {
	  		val sym = grammar.getTagNumberer.`object`(i) + "_" + j
	  		substateIds(i)(j) = sdict.indexOf(sym)
	  		id += 1
	  	}
	  }
	  
	  val nnsyms = sdict.indexOf("EX_1")+1
	  // Overlap by three symbols, since ROOT_0, EX_0 and EX_1 are both terminal and non-terminal 
	  val ntsyms = sdict.length - nnsyms + 3
	  val (ffrac, gmem, gtmem) = GPUmem
	  val maxnodes = (32 * (gmem/12/nnsyms/32)).toInt
	  val maxwords = 32 * (maxnodes/10/32)
	  val maxtrees = maxwords/8
	  val stride = 8192
	  val nthreads = 1024

	  var ts:TreeStore = new TreeStore(nnsyms, ntsyms, maxwords, maxnodes, maxtrees, maxlen, stride, nthreads, cdict)
	  ts.clearState
	  ts.loadrules(symbolsPath, nnsyms, ntsyms)
	  val (ffrac2, gmem2, gtmem2) = GPUmem
	  println("frac=%g, abs=%d" format (ffrac2, gmem2))

	  val testTreesIterator = testTrees.iterator()
	  val outTrees = new ArrayBuffer[Tree[String]]()
	  val currentTrees = new ArrayBuffer[Tree[String]]()
	  var nwords = 0
	  var nsentences = 0
	  var nnodes = 0
	  var done = false
	  
	  while(testTreesIterator.hasNext && nsentences < maxsents) {
	  	val testTree = testTreesIterator.next()
	  	val testSentence = testTree.getYield
	  	val len = testSentence.size
	  	if (len <= maxlen) {
	  		if (nwords + len > maxwords || nnodes + len*(len+1)/2 > maxnodes) {
	  		  done = true
	  		} else {

	  		// pos -> coarse Sym -> refinement -> score
	  			val scores: mutable.Buffer[Array[Array[Double]]] = parser.getTagScores(testSentence).asScala
	  			val ssc = FMat(ntsyms, scores.length)
	  			for ( (score,pos) <- scores.zipWithIndex) {
	  				for(i <- 0 until score.length if score(i) != null; j <- 0 until score(i).length) {
	  					ssc(substateIds(i)(j)-nnsyms+3, pos) = score(i)(j).toFloat
	  				}
	  			} 
	  		  ts.addtree(ssc)
	  		  currentTrees += testTree
	  		  nsentences += 1
	  		  nwords += len
	  		  nnodes += len*(len+1)/2
	  		}
	  	}
	  }
	  flip
	  try {
	  	ts.innerscores
	  } catch {
	    case e:Exception => println("\nException thrown, exiting "+e.toString)
	  }
	  val ff = gflop
	  val tt = ff._2
	  println("maxlen=%d, nsentences=%d, nwords=%d, nnodes=%d, maxsents=%d, maxwords=%d, maxnodes=%d\ntime= %f secs, %f sents/sec, %f gflops" format 
	  		(maxlen, nsentences, nwords, nnodes, maxtrees, maxwords, maxnodes, tt, nsentences/tt, ff._1))
 	  (ts, testTrees)
  }
  
  private def doParse(ts: TreeStore, goldTrees: IndexedSeq[Tree[String]]) = {
    val out = new ArrayBuffer[Tree[String]]
    val nsyms = ts.nnsyms
    ts.innerscores()
    // copy in the matrices
    val ts.OutData(wordsb4, wordsAfter, treesb4, treesAfter) = ts.copyOut
    for(s <- 0 until ts.ntrees) {
      val wordoffset = ts.iwordptr(s, 0)
      val nodeoffset = ts.itreeptr(s, 0)
      val len = goldTrees(s).size()
      def getL(beg: Int, end: Int, sym: Int) = {
        if(beg == end + 1)
          wordsb4(sym, wordoffset + beg)
        else
          treesb4(sym, ts.treepos(nodeoffset, beg, end, len))
      }

      def getU(beg: Int, end: Int, sym: Int) = {
         if(beg == end + 1)
          wordsAfter(sym, wordoffset + beg)
        else
          treesAfter(sym, ts.treepos(nodeoffset, beg, end, len))
      }
    }
  }
} 

/*

object TestParseKernel {
  
  def test1(args:Array[String]) = {
    val ddarr:DMat = load("\\code\\Parser\\grammar\\binrules.mat","nndarr")
    val diassign:DMat = load("\\code\\Parser\\grammar\\binrules.mat","fassign")
    val darr = IMat(ddarr)
    val iassign = IMat(diassign)
    val lens = icol(7,2,2)
    val nt = maxi(darr(?,0)).v
    val nblks = (lens(0)*lens(1)*lens(2)).v
    val indx1 = iassign(?,2)-1 + lens(2)*(iassign(?,1)-1 + lens(1)*(iassign(?,0)-1))

    val bcounts = accum(indx1,1,nblks)
    val prcounts = accum(((darr(?,0)-1) \ indx1), iones(size(darr,1),1), nt, nblks)
    val lcounts = accum(((darr(?,1)-1) \ indx1), iones(size(darr,1),1), nt, nblks)
    val rcounts = accum(((darr(?,2)-1) \ indx1), iones(size(darr,1),1), nt, nblks)
    val prpos = prcounts > 0
    val lpos = lcounts > 0
    val rpos = rcounts > 0
    
    iassign
  }
  
  
  def test2(args:Array[String]) = {
    val nsyms = 512
    val stride = 8192
    val nrules = 373375
    val pars = grand(nsyms, stride)
    val left = grand(nsyms, stride)
    val right = grand(nsyms, stride)
    val scales = grand(stride, 1)
    val nreps = args(0).toInt
    tic
    var i = 0
    while (i < nreps) {
      nnrules(pars.data, left.data, right.data, scales.data, stride, 1024)
      i += 1
    }
    val tt = toc
    println("kernel time=%f, gflops=%f" format (tt, 3L*stride*nrules*nreps/tt/1e9))
  }
  
  def main(args:Array[String]) {
    Mat.checkCUDA
    test2(args)
  }
}

*/






