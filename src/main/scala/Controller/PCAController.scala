package Controller

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
//PCA
import org.apache.spark.mllib.linalg.Vectors

object PCAController {
  var outputText: String = ""

  def runPCA(sc: SparkContext,
             inPath: String,
             saveResultPath: String,
             topPC: Int): Unit = {
    // Load data file
    val rows = sc.textFile(inPath)
    val dataRDD = rows.map(s => Vectors.dense(s.split(' ')
      .map(_.toDouble)))
      .cache()

    // Convert to RowMatrix
    val mat: RowMatrix = new RowMatrix(dataRDD)

    // Comput top n principal components(PC).
    val initStartTrainingTime = System.nanoTime()
    val pc: Matrix = mat.computePrincipalComponents(topPC)
    val trainingTimeInSecond = (System.nanoTime() - initStartTrainingTime)

    // Project data
    val initProjectTime = System.nanoTime()
    val projected: RowMatrix = mat.multiply(pc)
    val finishProjectInSecond = (System.nanoTime() - initProjectTime)

    // Save result
    projected.rows.saveAsTextFile(saveResultPath)

    // Show Sample
    val collect = projected.rows.take(100)
    var resultSample = ""
    collect.foreach {vector => resultSample += vector + "\n"}

    this.outputText = "Sample Projected Row Matrix principal component: \n"
    this.outputText += resultSample
    this.outputText += s"\n Compute PC Time : ${trainingTimeInSecond / 1000000000.0} seconds"
    this.outputText += s"\n Projecting Data Time : ${finishProjectInSecond / 1000000000.0} seconds"
    TextToFile.saveResultToTextFile("PCAResult.txt", outputText)
  }
}