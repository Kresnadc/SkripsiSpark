package org.apache.spark.examples.mllib

import org.apache.spark.{SparkConf, SparkContext}
// Naive Bayes
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils
// Naive Bayes

object NaiveBayesTester {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("NaiveBayesExample")
    val sc = new SparkContext(conf)
    // Contoh :

    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(sc, "E:/InputTest/sample_mllib_naive.txt")

    // Split data into training (60%) and test (40%).
    val Array(training, test) = data.randomSplit(Array(0.6, 0.4))

    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    // Save and load model
    model.save(sc, "E:/TestKmeans")
    val sameModel = NaiveBayesModel.load(sc, "E:/TestKmeans")
    // End Contoh

    sc.stop()
  }
}
