import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.IDFModel
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD


object TFIDF {
  var outputText: String = ""

  def runTFIDF(sc: SparkContext, inPath: String, outPath: String, minDocFreq: Int): Unit = {
    // Load documents (one per line).
    var temp = "E:/InputTest/sample_mllib_kmeans_data_small.txt"
    val documents: RDD[Seq[String]] = sc.textFile(inPath)
      .map(_.split(" ").toSeq)

    val hashingTF = new HashingTF()

    val initStartTFTime = System.nanoTime()
    val tf: RDD[Vector] = hashingTF.transform(documents)
    val tfTimeInSecond = (System.nanoTime() - initStartTFTime)


    // While applying HashingTF only needs a single pass to the data, applying IDF needs two passes:
    // First to compute the IDF vector and second to scale the term frequencies by IDF.
    // Note : spark.mllib IDF implementation provides an option for ignoring terms which occur in less than
    // a minimum number of documents. In such cases, the IDF for these terms is set to 0.
    // This feature can be used by passing the minDocFreq value to the IDF constructor.
    tf.cache()
    var idf : IDFModel = null;
    val initStartIDFTime = System.nanoTime()
    if(minDocFreq != 0){
      idf = new IDF(minDocFreq = 2).fit(tf)
    }else{
      idf = new IDF().fit(tf)
    }
    val idfTimeInSecond = (System.nanoTime() - initStartIDFTime)

    val initStartTFIDFTime = System.nanoTime()
    val tfidf: RDD[Vector] = idf.transform(tf)
    val tfidfTimeInSecond = (System.nanoTime() - initStartTFIDFTime)

    tfidf.saveAsTextFile(outPath + "TFIDFResult/")
    outputText += s"TF-IDF result save in '$outPath' \n"
    outputText += "Sample TF-IDF (top 100): \n"
    tfidf.take(100).foreach(x => outputText += x + "\n")
    outputText += s"TF Execution Time : ${idfTimeInSecond / 1000000000.0} seconds. \n"
    outputText += s"IDF Execution Time : ${tfTimeInSecond / 1000000000.0} seconds. \n"
    outputText += s"TF-IDF Execution Time : ${tfidfTimeInSecond / 1000000000.0} seconds. \n"
  }
}
