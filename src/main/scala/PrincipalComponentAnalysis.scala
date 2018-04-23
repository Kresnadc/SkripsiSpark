import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
//PCA
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import java.io.{File, PrintWriter}

object PrincipalComponentAnalysis {
  var outputText: String = ""

//  def main(args: Array[String]): Unit = {
//    val data = Array(
//      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
//      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
//      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0),
//      Vectors.dense(5.0, 2.0, 0.0, 5.0, 8.0),
//      Vectors.dense(5.0, 0.0, 1.0, 7.0, 9.0),
//      Vectors.dense(3.0, 1.0, 0.0, 5.0, 7.0))
//    val rows2 =  Seq(
//      new LabeledPoint(0, Vectors.dense(1, 0, 0, 0, 1)),
//      new LabeledPoint(1, Vectors.dense(1, 1, 0, 1, 0)),
//      new LabeledPoint(1, Vectors.dense(1, 1, 0, 0, 0)),
//      new LabeledPoint(0, Vectors.dense(1, 0, 0, 0, 0)),
//      new LabeledPoint(1, Vectors.dense(1, 1, 0, 0, 0)))
//  }

  def runPCA(sc: SparkContext,
             inPath: String,
             saveResultPath: String,
             topPC: Int): Unit = {
    // Load data file
    val rows = sc.textFile(inPath)
    val dataRDD = rows.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    // Convert to RowMatrix
    val mat: RowMatrix = new RowMatrix(dataRDD)

    // Comput top n principal components(PC).
    val initStartTrainingTime = System.nanoTime()
    val pc: Matrix = mat.computePrincipalComponents(topPC)
    val trainingTimeInSecond = (System.nanoTime() - initStartTrainingTime)

    // Project the rows to the linear space spanned by the top 4 principal components.
    val projected: RowMatrix = mat.multiply(pc)

    // Save result
    projected.rows.saveAsTextFile(saveResultPath)

    // Show Sample
    val collect = projected.rows.take(100)
    var resultSample = ""
    collect.foreach {vector => resultSample += vector + "\n"}

    this.outputText += "Sample Projected Row Matrix principal component: \n"
    this.outputText += resultSample
    this.outputText += s"\n Execution Time : ${trainingTimeInSecond / 1000000000.0} seconds"
    TextToFile.saveResultToTextFile("PCAResult.txt", outputText)
    // Compute the top 5 principal components.
//    val data2: RDD[LabeledPoint] = sc.parallelize(rows2)
//    val pca = new PCA(5).fit(data2.map(_.features))
//
//    val projected2 = data2.map(p => p.copy(features = pca.transform(p.features)))
//
//    val collect2 = projected2.collect()
//    println("Projected vector of principal component:")
//    collect2.foreach { vector => println(vector) }
  }
}