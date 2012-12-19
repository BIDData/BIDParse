package BIDParse
import BIDMat.{Mat, DMat, FMat, IMat, CMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat}
import BIDMat.SciFunctions._
import BIDMat.MatFunctions._

import jcuda._
import jcuda.runtime.JCuda
import jcuda.runtime.cudaMemcpyKind._
import jcuda.jcublas.JCublas;
import edu.berkeley.bid.CUMAT
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
		            val maxlen:Int, val stride:Int, val nthreads:Int, leveldiff:Float, val cdict:CSMat) {
  val nnsyms = 32 * ((nnsyms0+31) / 32)
  val ntsyms = 32 * ((ntsyms0+31) / 32)
  // Storage for input words and score trees
  val fwordsb4     = FMat(ntsyms, maxwords)            // Accumulate terminal scores here
  val wordsb4      = GMat(ntsyms, maxwords)            // Buffer in GPU memory for terminal scores
  val wordsafter   = GMat(ntsyms, maxwords)            // Terminal scores after unary rule application
  val treesb4      = GMat(nnsyms, maxnodes)            // Score tree storage before unary rules
  val treesafter   = GMat(nnsyms, maxnodes)            // Score trees after unary rule application
  val fscales      = FMat(maxlen, 1)                   // Log scale factor by level
  for (i <- 0 until maxlen) {fscales(i,0) = i*leveldiff}
  val scales       = GMat(fscales)
  val flscale      = ones(stride, 1) * 3000.0f
  val lscale       = GMat(flscale)
  
  var nnunary:GMat = null                              // Unary rule matrix (dense), non-terminal to non-terminal
  var nnunaryx:GMat = null                             // Unary rule matrix (dense), non-terminal to non-terminal
  var ntunary:GMat = null                              // Unary rule matrix (dense), non-terminal to terminal
  var ttunary:GMat = null                              // Unary rule matrix (dense), terminal to terminal
  
  // Buffers (transposed relative to tree store) for binary rule application
  val leftbuff     = GMat(stride, ntsyms)              // Left score buffer       
  val rightbuff    = GMat(stride, ntsyms)              // Right score buffer 
  val parb4buff    = GMat(stride, nnsyms)              // Parent score buffer (before unary rules)
  val parafterbuff = GMat(stride, nnsyms)              // Parent score buffer (after unary rules)
  
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
  class kState(val parrb4:GMat, val parrafter:GMat, val leftarr:GMat, val rightarr:GMat, 
      val lpos:IMat, val rpos:IMat, val nrules:Int, fctn:(Pointer, Pointer, Pointer, Pointer, Int, Int) => Int) {
	  val ileftptrs       = IMat(stride,1)                             // IMat indices into tree storage, updated on host
	  val irightptrs      = IMat(stride,1)
	  val iparptrs        = IMat(stride,1)
	  val ileftpt         = Pointer.to(ileftptrs.data)                 // JCuda Pointers to the above
	  val irightpt        = Pointer.to(irightptrs.data)	
	  val iparpt          = Pointer.to(iparptrs.data)
	  
	  // JCuda pointers to the scale factor column (the last column) within the main buffers
	  val leftscalept     = leftbuff.data.withByteOffset(stride*(leftbuff.ncols-1)*Sizeof.FLOAT)
	  val rightscalept    = rightbuff.data.withByteOffset(stride*(rightbuff.ncols-1)*Sizeof.FLOAT)
	  val parb4scalept    = parb4buff.data.withByteOffset(stride*(parb4buff.ncols-1)*Sizeof.FLOAT)
	  val parafterscalept = parafterbuff.data.withByteOffset(stride*(parafterbuff.ncols-1)*Sizeof.FLOAT)
	  val levelscalept    = scales.data
	  var ndo = 0

	  // Process a block of data. Copy and transpose from tree storage, run kernels, copy back
	  def doblock(level:Int) = {
		  JCuda.cudaMemcpy(parptrs.data, iparpt, ndo*Sizeof.INT, cudaMemcpyHostToDevice)                     // Copy IMat data to GIMats
	    JCuda.cudaMemcpy(leftptrs.data, ileftpt, ndo*Sizeof.INT, cudaMemcpyHostToDevice)
	    JCuda.cudaMemcpy(rightptrs.data, irightpt, ndo*Sizeof.INT, cudaMemcpyHostToDevice)	  
//	    leftbuff.clear
//	    rightbuff.clear
//	    parb4buff.clear
//	    parafterbuff.clear
//	    println("ndo=%d" format ndo)
//		  println("lnnz=%d, rnnz=%d, pb4=%d, paft=%d" format (FMat(leftbuff).nnz, FMat(rightbuff).nnz, FMat(parb4buff).nnz, FMat(parafterbuff).nnz))

	    pcopyTXin(leftptrs.data, leftarr.data, leftbuff.data, stride, leftarr.nrows, ndo)                  // Do the data copy/transpose
	    pcopyTXin(rightptrs.data, rightarr.data, rightbuff.data, stride, rightarr.nrows, ndo) 
	    pcopyTXin(parptrs.data, parrb4.data, parb4buff.data, stride, parrb4.nrows, ndo)
	    
//	    CUMAT.applyop(leftscalept, stride, 1, rightscalept, stride, 1, leftscalept, GMat.BinOp.op_add)     // Compute prescale factors
//	    CUMAT.applyop(levelscalept.withByteOffset(level*Sizeof.FLOAT), 1, 1, leftscalept, stride, 1, rightscalept, GMat.BinOp.op_sub)
//	    CUMAT.applygfun(rightscalept, parb4scalept, stride, GMat.TransF.exp)
	    
	    fctn(parb4buff.data, leftbuff.data, rightbuff.data, lscale.data, ndo, nthreads)                   // Apply binary rules
	    Mat.nflops += 3L * ndo * nrules
	    parafterbuff ~ parb4buff * nnunary                                                                 // Apply unary rules
//	    println("lnnz=%d, rnnz=%d, pb4=%d, paft=%d" format (FMat(leftbuff).nnz, FMat(rightbuff).nnz, FMat(parb4buff).nnz, FMat(parafterbuff).nnz))
//	    println("parb4nnz=%d, parannz=%d" format (FMat(parrb4).nnz, FMat(parrafter).nnz))
	    
//	    println("par   %d %d %d %d %d %d %d %d %d %d" format (iparptrs(0), iparptrs(1), iparptrs(2), iparptrs(3), iparptrs(4), iparptrs(5), iparptrs(6), iparptrs(7), iparptrs(8), iparptrs(9)))
//		  println("left  %d %d %d %d %d %d %d %d %d %d" format (ileftptrs(0), ileftptrs(1), ileftptrs(2), ileftptrs(3), ileftptrs(4), ileftptrs(5), ileftptrs(6), ileftptrs(7), ileftptrs(8), ileftptrs(9)))
//		  println("right %d %d %d %d %d %d %d %d %d %d" format (irightptrs(0), irightptrs(1), irightptrs(2), irightptrs(3), irightptrs(4), irightptrs(5), irightptrs(6), irightptrs(7), irightptrs(8), irightptrs(9)))
	    pcopyTXout(parptrs.data, parb4buff.data, parrb4.data, stride, parrb4.nrows, ndo)                   // Copy/transpose data out
	    pcopyTXout(parptrs.data, parafterbuff.data, parrafter.data, stride, parrafter.nrows, ndo)
//	    println("parb4nnz=%d, parannz=%d" format (FMat(parrb4).nnz, FMat(parrafter).nnz))
	    ndo = 0
	  }
	  
	  def update(itree:Int, x:Int, y:Int, level:Int, treelen:Int) = {                                     // Add the indices for one binary rule application
	  	iparptrs(ndo) = itreeptr(itree) + treepos(x, level, treelen)
  	  ileftptrs(ndo) = lpos(itree) + treepos(x, y, treelen);
  	  irightptrs(ndo) = rpos(itree) + treepos(x+y+1, level-y-1, treelen)
  	  ndo = ndo + 1
  	  if (ndo >= stride) doblock(level);
	  }
  }

  case class OutData(wordsb4:FMat, wordsafter:FMat, treesb4:FMat, treesafter:FMat)

  def copyOut: OutData = {
    OutData(fwordsb4, wordsafter.toFMat, treesb4.toFMat, treesafter.toFMat)
  }
  
  // Create all the kernel states. 
  val nnState = new kState(treesb4, treesafter, treesafter, treesafter, itreeptr, itreeptr, 373375, nnrules _)
  val ntState = new kState(treesb4, treesafter, treesafter, wordsafter, itreeptr, iwordptr, 334338, ntrules _)
  val tnState = new kState(treesb4, treesafter, wordsafter, treesafter, iwordptr, itreeptr, 417022, tnrules _)
  val ttState = new kState(treesb4, treesafter, wordsafter, wordsafter, iwordptr, iwordptr, 600835, ttrules _)
  
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
	  wordsb4.clear
	  wordsafter.clear
	  treesb4.clear
	  treesafter.clear
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
    fwordsb4(0->size(syms,1), (nwords-len)->nwords) = syms
  }
  
  // Map the x,y coordinates of a tree node to a linear index for that tree
  def treepos(x:Int, y:Int, len:Int) = {
    x + y*len - y*(y-1)/2
  }
  
  // Apply unary rules just to the terminal nodes (non-terminals handled in doblock)
  def unaryrules {
	  JCuda.cudaMemcpy(wordsb4.data, Pointer.to(fwordsb4.data), nwords*ntsyms*Sizeof.FLOAT, cudaMemcpyHostToDevice)
	  var itree = 0
	  while (itree < ntrees) {
	    val sentsize = iwordptr(itree+1)-iwordptr(itree)
	  	val inptr = wordsb4.data.withByteOffset(iwordptr(itree)*Sizeof.FLOAT)
	    JCublas.cublasSgemm('n', 'n', ntsyms, sentsize, ntsyms, 1.0f, ttunary.data, ntsyms, 
	          	inptr, ntsyms, 0f, wordsafter.data.withByteOffset(iwordptr(itree)*Sizeof.FLOAT), ntsyms)
	    JCublas.cublasSgemm('n', 'n', nnsyms, sentsize, ntsyms, 1.0f, ntunary.data, ntsyms, 
	            inptr, ntsyms, 0f, treesafter.data.withByteOffset(itreeptr(itree)*Sizeof.FLOAT), nnsyms)
	    itree = itree + 1
	  }
  }
  
  def applyscales(level:Int) = {
    
  }
  
  def loadrules(fname:String, nnsyms0:Int, ntsyms0:Int) = {
    val nnsyms = 32*((nnsyms0+31)/32)
    val ntsyms = 32*((ntsyms0+31)/32)
    val nnrange = (0->nnsyms0)
    val ntrange = ((nnsyms0-3)->(nnsyms0+ntsyms0-3))
    val fmodel = FMat(load(fname, "umodel").asInstanceOf[DMat])    
    val fnn = zeros(nnsyms, nnsyms)
    fnn(nnrange, nnrange) = fmodel(nnrange, nnrange)
    nnunary = GMat(fnn)
    val fnnx = zeros(ntsyms, ntsyms)
    fnnx(nnrange, nnrange) = fmodel(nnrange, nnrange) 
    nnunaryx = GMat(fnnx)
    val fnt = zeros(nnsyms, ntsyms)
    fnt(nnrange, 0->ntsyms0) = fmodel(nnrange, ntrange)
    ntunary = GMat(fnt)
    val ftt = zeros(ntsyms, ntsyms)
    ftt(0->ntsyms0, 0->ntsyms0) = fmodel(ntrange, ntrange)
    ttunary = GMat(ftt)
  }
  
  // Compute inner scores. Work one level at a time, then one tree at a time. This just queues left-right symbol pair
  // jobs for the four kState instances. They do the work when their buffers are full. 
  def innerscores() {
  	unaryrules
  	var level = 1
  	print("level 1")
  	while (level < maxheight) {
  	  var itree = 0
  	  while (itree < ntrees) {
  	    var sentlen = iwordptr(itree+1)-iwordptr(itree)
  	    var x = 0;
  	    while (x < sentlen - level) {
  	      var y = 0
  	      while (y < level) {
  	      	if (y == 0 && level == 1) {
  	      	  ttState.update(itree, x, y, level, sentlen)
  	      	} 
  	      	if (y == 0) {
  	      	  tnState.update(itree, x, y, level, sentlen)
  	      	} 
  	      	if (y == level-1) {
   	      	  ntState.update(itree, x, y, level, sentlen)
  	      	} 
  	      	nnState.update(itree, x, y, level, sentlen)  // Always do this since terminals will have generated NTs through unary rules
  	        y = y +1
  	      }  	      
  	      x = x + 1
  	    }  	    
  	    itree = itree+1
  	  } 
  	  if (ttState.ndo > 0) {ttState.doblock(level)}       // Almost done with this level, so flush (run rules on) any queued data
  	  if (tnState.ndo > 0) {tnState.doblock(level)}
  	  if (ntState.ndo > 0) {ntState.doblock(level)}
  	  if (nnState.ndo > 0) {nnState.doblock(level)}
  	  applyscales(level)
  	  level = level+1
  	  print(", %d" format level)
  	}
  	println("")
  }

} 

import edu.berkeley.nlp.PCFGLA._

object BIDParser {
  
	def main(args:Array[String]):Unit = {
		val berkeleyGrammar = args(0)
		val cpath = args(1)
		val sympath = args(2)
		var maxlen:Int = 36
		if (args.size > 3) maxlen = args(3).toInt
		runParser(args(0), args(1), args(2), maxlen)
	}

  def runParser(grammarPath:String, corpusPath:String, symbolsPath:String, maxlen:Int) = {
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
	  val GPUmem = 1024L*1024*3000
	  val maxnodes = (32 * (GPUmem/8/nnsyms/32)).toInt
	  val maxwords = 32 * (maxnodes/10/32)
	  val maxtrees = maxwords/8
	  val stride = 8192
	  val nthreads = 1024
	  val leveldiff = 10f

	  var ts:TreeStore = new TreeStore(nnsyms, ntsyms, maxwords, maxnodes, maxtrees, maxlen, stride, nthreads, leveldiff, cdict)
	  ts.clearState
	  ts.loadrules(symbolsPath+"uni_closure.mat", nnsyms, ntsyms)

	  val testTreesIterator = testTrees.iterator()
	  val outTrees = new ArrayBuffer[Tree[String]]()
	  val currentTrees = new ArrayBuffer[Tree[String]]()
	  var nwords = 0
	  var nsentences = 0
	  var nnodes = 0
	  var done = false
	  
	  while(testTreesIterator.hasNext) {//} && nsentences < 1) {
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
	  ts.innerscores
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
          treesb4(sym, nodeoffset + ts.treepos(beg, end, len))
      }

      def getU(beg: Int, end: Int, sym: Int) = {
         if(beg == end + 1)
          wordsAfter(sym, wordoffset + beg)
        else
          treesAfter(sym, nodeoffset + ts.treepos(beg, end, len))
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






