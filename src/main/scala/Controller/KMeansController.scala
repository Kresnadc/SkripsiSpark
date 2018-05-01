package Controller

import ModifiedKMeans.CustomKMeans
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.io.Source

/**
  *
  */
object KMeansController {
  var outputText: String = ""

  def runKMeans(sc: SparkContext,
          inPath: String,
          saveModelPath: String,
          numClusters: Int,
          maxIterations: Int,
          resultConf : Array[Boolean]): Unit ={

    // Load and parse the data
    val parsedData = preprocessingDataPhoneCSV(sc, inPath)

    // Train data using KMeans
    val clusters = CustomKMeans.train(parsedData, numClusters, maxIterations, resultConf)

    // Save and load model
    clusters.save(sc, saveModelPath)
    val modelKMeans = KMeansModel.load(sc, saveModelPath)

    // Load result
    outputText = Source.fromFile("OutputTextKMEANS.txt").mkString

    // New Cluster Center
    outputText += "Cluster Centers: \n"
    modelKMeans.clusterCenters.foreach(outputText += _ + "\n")
  }

  def preprocessingDataPhoneCSV(sc: SparkContext, path: String): RDD[Vector] ={
    //Read file
    //datetime, CellID, countrycode, smsin, smsout, callin, callout, internet
    val csv = sc.textFile(path)  // original file

    //find the headers
    val header = csv.first;

    //remove the header
    val data = csv.filter(_(0) != header(0));

    //To create a RDD of (Vector) pairs
    data.map { line =>
      val parts = line.split(",", -1)
      Vectors.dense(
        parseDouble(parts(3)),
        parseDouble(parts(4)),
        parseDouble(parts(5)),
        parseDouble(parts(6))
      )
    }.cache()
  }

  def parseDouble(str: String): Double ={
    if(str != ""){
      str.toDouble
    }else{
      0.0
    }
  }
}
