package org.apache.spark.examples.mllib

import org.apache.spark.{SparkConf, SparkContext}
// Naive Bayes
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.MLUtils
// Naive Bayes

object NaiveBayesTester {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("NaiveBayesExample").setMaster("local")
    val sc = new SparkContext(conf)
    // Contoh :

    // Load and parse the data file.
    println("Load and parse the data file.")
    // Sparse data format LibSVM
    val data = MLUtils.loadLibSVMFile(sc, "E:/InputTest/sample_mllib_naive.txt")
    data.collect().foreach(println)
    val abc = data.collect()(1)
    // Split data into training (60%) and test (40%).
    val Array(training, test) = data.randomSplit(Array(0.6, 0.4))

    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    // Save and load model
    println("Save and load model")
    model.save(sc, "E:/Output/ModelNaive/")
    val naiveModel = NaiveBayesModel.load(sc, "E:/Output/ModelNaive/")
    // End Contoh
    println("model type : "+ naiveModel.modelType)
    val res = naiveModel.predict(abc.features)
    println("result " + res)

    //    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    //    val newDataDF = sqlContext.read.parquet("E:/Output/ModelNaive/data/*.parquet")
    //    val haha = newDataDF.collect()
    //    val keNol = haha(1)
    //    println(newDataDF)

    sc.stop()
  }
}
