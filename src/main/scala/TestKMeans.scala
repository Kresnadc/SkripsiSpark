import org.apache.spark.{SparkConf, SparkContext}
//import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.Vectors

object TestKMeans {

  def main(args: Array[String]) {
    //Create SparkContext
    val conf = new SparkConf().setAppName("KMeansExample").setMaster("local")
    val sc = new SparkContext(conf)

    // Load and parse the data
    // val data = sc.textFile("hdfs://localhost:9001/user/hadoop/iris-dataset")
    val data = sc.textFile("E:/InputTest/sample_mllib_kmeans_data2.txt")
    //println("Element of RDD: "+ data.count())
    val a = data.map(s => s.split(';'))
    val b = a.map(s => Array(s(2), s(3), s(4), s(5), s(6), s(7), s(8)))
    val parsedData = b.map(s => Vectors.dense(s.map(_.toDouble))).cache()

    println("Parsed Data :")
    parsedData.collect().foreach(println)

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
}