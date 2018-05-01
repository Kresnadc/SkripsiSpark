package Controller

import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.{HashingTF, IDF, IDFModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD


object TFIDFController {
  var outputText: String = ""

  def runTFIDF(sc: SparkContext,
               inPath: String,
               outPath: String,
               numOfFeature: Int,
               minDocFreq: Int): Unit = {
    // Load documents (one per line).
    val documents: RDD[Seq[String]] = sc.textFile(inPath)
      .map(_.split(" ").toSeq)

    var hashingTF: HashingTF = null
    if(numOfFeature == 0){
      hashingTF = new HashingTF()
    }else{
      hashingTF = new HashingTF(numOfFeature)
    }

    val initStartTFTime = System.nanoTime()
    val tf: RDD[Vector] = hashingTF.transform(documents)
    val tfTimeInSecond = (System.nanoTime() - initStartTFTime)

    tf.cache()

    var idf : IDFModel = null;
    if(minDocFreq != 0){
       new IDF(minDocFreq = 2).fit(tf)
    }else{
      idf = new IDF().fit(tf)
    }

    val initStartTFIDFTime = System.nanoTime()
    val tfidf: RDD[Vector] = idf.transform(tf)
    val tfidfTimeInSecond = (System.nanoTime() - initStartTFIDFTime)

    tfidf.saveAsTextFile(outPath)

    outputText = s"TF-IDF result save in '$outPath' \n"
    outputText += "Sample TF-IDF (top 100): \n"
    tfidf.take(100).foreach(x => outputText += x + "\n")
    outputText += s"TF Execution Time : ${tfTimeInSecond / 1000000000.0} seconds. \n"
    outputText += s"TF-IDF Execution Time : ${tfidfTimeInSecond / 1000000000.0} seconds. \n"
    TextToFile.saveResultToTextFile("TFIDFResult.txt", outputText)
  }
}
