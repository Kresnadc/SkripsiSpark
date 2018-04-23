import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.io.{BufferedSource, Source}

object KMeansTester {
  var outputText: String = ""

//  def main(args: Array[String]): Unit = {
//    var inPath = "E:/InputTest/sms-call-internet-mi-2013-11-07.csv"
//    var saveModelPath = "E:/Output/ModelKmeans"
//    var numCluster = 4
//    var maxIteration = 3
//    var resultConf = Array(true,false,true,false)
//    val conf = new SparkConf().setAppName("KMeansExample").setMaster("local")
//    val sc = new SparkContext(conf)
//    runKMeans(sc, inPath, saveModelPath, numCluster, maxIteration, resultConf)
//  }

  def runKMeans(sc: SparkContext,
          inPath: String,
          saveModelPath: String,
          numClusters: Int,
          maxIterations: Int,
          resultConf : Array[Boolean]): Unit ={

    // Load and parse the data
    val parsedData = preprocessingDataPhoneCSV(sc, inPath)

    // Cluster the data into two classes using KMeans
    val clusters = CustomKMeans.train(parsedData, numClusters, maxIterations, resultConf)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors (COST) = " + WSSSE)

    // Save and load model
    clusters.save(sc, saveModelPath)
    val modelKMeans = KMeansModel.load(sc, saveModelPath)

    // Load result
    outputText += Source.fromFile("OutputTextKMEANS.txt").mkString

    // New Cluster Center
    outputText += "Cluster Centers: \n"
    modelKMeans.clusterCenters.foreach(outputText += _ + "\n")
  }

  def preprocessingDataPhoneCSV(sc: SparkContext, path: String): RDD[Vector] ={
    //Read the file
    //datetime, CellID, countrycode, smsin, smsout, callin, callout, internet
    val csv = sc.textFile(path)  // original file

    //To find the headers
    val header = csv.first;

    //To remove the header
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
