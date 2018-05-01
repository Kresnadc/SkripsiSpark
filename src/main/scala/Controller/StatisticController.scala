package Controller

import java.io._

import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics};

object StatisticController {
  var outputText: String = ""

  def runStatistic(sc: SparkContext,
          inPath: String,
          outPath: String,
          conf: Array[Boolean]): Unit = {
    //Start Training
    println("Start Training")

    // Load data file.
    val data = sc.textFile(inPath)
    val dataRDD = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    // Train Data
    val initStartTrainingTime = System.nanoTime()
    val summary: MultivariateStatisticalSummary = Statistics.colStats(dataRDD)
    val trainingTimeInSecond = (System.nanoTime() - initStartTrainingTime)

    // Result
    var result = "";
    if(conf(0)){
      result = "Count : " + summary.count + "\n"
    }
    if(conf(1)){
      result += "Mean : " + summary.mean + "\n"
    }
    if(conf(2)){
      result += "Min : " + summary.min + "\n"
    }
    if(conf(3)){
      result += "Max : " + summary.max + "\n"
    }
    if(conf(4)){
      result += "Variance : " + summary.variance + "\n"
    }
    result += s"\n Execution Time : ${trainingTimeInSecond / 1000000000.0} seconds"
    outputText = result
    TextToFile.saveResultToTextFile("SummaryStatisticResult.txt", outputText)
  }
}