import org.apache.spark.{SparkConf, SparkContext}
//import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}

object TestKMeans {

  def main(args: Array[String]) {
    //Create SparkContext
    val conf = new SparkConf().setAppName("KMeansExample").setMaster("local")
    val sc = new SparkContext(conf)

    // Load and parse the data
    // val data = sc.textFile("hdfs://localhost:9001/user/hadoop/iris-dataset")
    val parsedData = preprocessingDataPhoneCSV(sc, "E:/InputTest/sms-call-internet-mi-2013-11-07.csv")

    //println("Parsed Data :")
    //parsedData.collect().foreach(println)

    // Cluster the data into two classes using KMeans
    val numClusters = 5
    val numIterations = 20
    val clusters = CustomKMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors (COST) = " + WSSSE)

    // Save and load model
    clusters.save(sc, "E:/Output/ModelKmeans")
    val sameModel = KMeansModel.load(sc, "E:/Output/ModelKmeans")

    // New Cluster Center
    println("Cluster Centers: ")
    sameModel.clusterCenters.foreach(println)

    //Stop
    println("Type 's' to stop process:")
    var scanner = new java.util.Scanner(System.in)

    while( scanner.next() != "s"){
      println("Type 's' to stop process:")
      scanner = new java.util.Scanner(System.in)
    }
    sc.stop()
  }

  def preprocessingDataPhoneCSV(sc: SparkContext, path: String): RDD[Vector] ={
    //Read the file
    //datetime,CellID,countrycode,smsin,smsout,callin,callout,internet
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