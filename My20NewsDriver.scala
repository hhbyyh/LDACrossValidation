/**
 * Created by yuhao on 4/14/15.
 */
package My20NewsDriver

/**
 * Created by yuhao on 3/3/15.
 */


import java.io.{File, PrintWriter}
import java.text.BreakIterator
import org.apache.spark.mllib.clustering.{OnlineLDAOptimizer, OnlineLDA, LDA, MyLDA}
import scala.collection.mutable
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, kron, sum}

import scala.io.Source


object My20NewsDriver extends Serializable{

  private case class Params(
                             var input: Seq[String] = Seq.empty,
                             var k: Int = 10,
                             maxIterations: Int = 100,
                             docConcentration: Double = -1,
                             topicConcentration: Double = -1,
                             vocabSize: Int = 100000,
                             var stopwordFile: String = "",
                             checkpointDir: Option[String] = None,
                             checkpointInterval: Int = 10)

  def main(args: Array[String]) {
    val defaultParams = Params()
    //    defaultParams.input = defaultParams.input.:+ (inputDir + "wiki/small/")
    defaultParams.input = defaultParams.input.:+ (args(0) + "20news-bydate-train-stanford-classifier.txt")
    //    defaultParams.input = defaultParams.input.:+ ("/home/yuhao/workspace/DocSet/apple/texts" )
    defaultParams.stopwordFile = args(0) + "stop.txt"
    defaultParams.k = args(1).toInt
    //    defaultParams.input = defaultParams.input.:+ ("/home/yuhao/workspace/DocSet/temp.txt")
    run(defaultParams)
  }

  private def run(params: Params) {
    val conf = new SparkConf().setAppName(s"LDAExample with $params")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    // Load documents, and prepare them for LDA.
    val preprocessStart = System.nanoTime()
    val (corpus, vocabArray, actualNumTokens) =
      preprocess(sc, params.input, params.vocabSize, params.stopwordFile)
    corpus.cache()
    val actualCorpusSize = corpus.count()
    val actualVocabSize = vocabArray.size
    val preprocessElapsed = (System.nanoTime() - preprocessStart) / 1e9

    println(s"Corpus summary:")
    println(s"\t Training set size: $actualCorpusSize documents")
    println(s"\t Vocabulary size: $actualVocabSize terms")
    println(s"\t Training set size: $actualNumTokens tokens")
    println(s"\t Preprocessing time: $preprocessElapsed sec")
    println()

    val vocabSize = corpus.first._2.size
    val D = corpus.count().toInt // total documents count
    val onlineLDA = new OnlineLDAOptimizer(20, 11314, vocabSize, 0.1, 0.01, 65, 0.75)
    val docArr = corpus.collect()
    val S = 10

    val matrixLine = Source.fromFile("/home/yuhao/workspace/DocSet/20_newsgroups/matrix.data")
      .getLines()
    val doubles = matrixLine.flatMap(line => line.split(",")).map(_.toDouble).toArray

    for(i <- 0 until 20){
      onlineLDA.lambda(i, ::) := new BDV[Double](doubles.slice(i * 75720, (i + 1) * 75720)).t
    }

    onlineLDA.update()

    val startTime = System.nanoTime()
    for(i <- 0 until 10){
      val batch = docArr.slice(S * i, S*(i + 1))
      onlineLDA.submitMiniBatch(corpus.context.parallelize(batch, 1))
      println(i)
    }

    val elapsed = (System.nanoTime() - startTime) / 1e9
    println(s"Finished training LDA model.  Summary:")
    println(s"\t Training time: $elapsed sec")


    for(i<- 0 until 20){
      val row = onlineLDA.lambda(i, ::).t.toArray
      val top = row.zipWithIndex.sortBy(-_._1).take(10)
      println()
      println("topic " + i)
      top.foreach(pair =>{
        val term = vocabArray(pair._2)
        println(term + ": " + pair._1 )
      })
    }

  }

  /**
   * Load documents, tokenize them, create vocabulary, and prepare documents as term count vectors.
   * @return (corpus, vocabulary as array, total token count in corpus)
   */
  private def preprocess(
                          sc: SparkContext,
                          paths: Seq[String],
                          vocabSize: Int,
                          stopwordFile: String): (RDD[(Long, Vector)], Array[String], Long) = {

    // Get dataset of document texts
    // One document per line in each text file.
    val textRDD: RDD[String] = sc.textFile(paths.mkString(",")).coalesce(4)

    // Split text into words
    val tokenizer = new SimpleTokenizer(sc, stopwordFile)
    val tokenized: RDD[(Long, IndexedSeq[String])] = textRDD.zipWithIndex().map { case (text, id) =>
      id -> tokenizer.getWords(text)
    }
    tokenized.cache()

    // Counts words: RDD[(word, wordCount)]
    val wordCounts: RDD[(String, Long)] = tokenized
      .flatMap { case (_, tokens) => tokens.map(_ -> 1L) }
      .reduceByKey(_ + _)
    wordCounts.cache()
    val fullVocabSize = wordCounts.count()
    // Select vocab
    //  (vocab: Map[word -> id], total tokens after selecting vocab)
    var (vocab: Map[String, Int], selectedTokenCount: Long) = {
      val tmpSortedWC: Array[(String, Long)] = if (vocabSize == -1 || fullVocabSize <= vocabSize) {
        // Use all terms
        wordCounts.collect().sortBy(-_._2)
      } else {
        // Sort terms to select vocab
        wordCounts.sortBy(_._2, ascending = false).take(vocabSize)
      }
      (tmpSortedWC.map(_._1).zipWithIndex.toMap, tmpSortedWC.map(_._2).sum)
    }

    vocab = sc.textFile("/home/yuhao/workspace/DocSet/20_newsgroups/dict.data")
                      .flatMap(line => line.split("\\s+")).collect().zipWithIndex.toMap

    val documents = tokenized.map { case (id, tokens) =>
      // Filter tokens by vocabulary, and create word count vector representation of document.
      val wc = new mutable.HashMap[Int, Int]()
      tokens.foreach { term =>
        if (vocab.contains(term)) {
          val termIndex = vocab(term)
          wc(termIndex) = wc.getOrElse(termIndex, 0) + 1
        }
      }
      val indices = wc.keys.toArray.sorted
      val values = indices.map(i => wc(i).toDouble)

      val sb = Vectors.sparse(vocab.size, indices, values)
      (id, sb)
    }

    val vocabArray = new Array[String](vocab.size)
    vocab.foreach { case (term, i) => vocabArray(i) = term }

    (documents, vocabArray, selectedTokenCount)
  }
}

/**
 * Simple Tokenizer.
 *
 * TODO: Formalize the interface, and make this a public class in mllib.feature
 */
private class SimpleTokenizer(sc: SparkContext, stopwordFile: String) extends Serializable {

  private val stopwords: Set[String] = if (stopwordFile.isEmpty) {
    Set.empty[String]
  } else {
    val stopwordText = sc.textFile(stopwordFile).collect()
    stopwordText.flatMap(_.stripMargin.split("\\s+")).toSet
  }

  // Matches sequences of Unicode letters
  private val allWordRegex = "^(\\p{L}*)$".r

  // Ignore words shorter than this length.
  private val minWordLength = 3

  def getWords(text: String): IndexedSeq[String] = {

    val words = new mutable.ArrayBuffer[String]()

    // Use Java BreakIterator to tokenize text into words.
    val wb = BreakIterator.getWordInstance
    wb.setText(text)

    // current,end index start,end of each word
    var current = wb.first()
    var end = wb.next()
    while (end != BreakIterator.DONE) {
      // Convert to lowercase
      val word: String = text.substring(current, end).toLowerCase
      // Remove short words and strings that aren't only letters
      word match {
        case allWordRegex(w) if w.length >= minWordLength && !stopwords.contains(w) =>
          words += w
        case _ =>
      }

      current = end
      try {
        end = wb.next()
      } catch {
        case e: Exception =>
          // Ignore remaining text in line.
          // This is a known bug in BreakIterator (for some Java versions),
          // which fails when it sees certain characters.
          end = BreakIterator.DONE
      }
    }
    words
  }

}


