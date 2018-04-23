import java.io.{File, PrintWriter}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

object AlternatingLeastSquares {
  var outputText: String = ""
  var model: MatrixFactorizationModel = null;

//  def main(args: Array[String]): Unit = {
//    val inPath = "E:/InputTest/ALS_data.txt"
//    val rank = 10
//    val numIterations = 10
//  }

  def startTraining(sc: SparkContext,
             inPath: String,
             saveModelPath: String,
             rank: Int,
             numIteration: Int): Unit = {
    // Load and parse the data
    val data = sc.textFile(inPath)
    val ratings = data.map(_.split(',') match { case Array(user, item, rate) =>
      Rating(user.toInt, item.toInt, rate.toDouble)
    })

    // Build model ALS
    val initStartTrainingTime = System.nanoTime()
    val model = ALS.train(ratings, rank, numIteration, 0.01)
    val trainingTimeInSecond = (System.nanoTime() - initStartTrainingTime)

    // Save and load model
    model.save(sc, saveModelPath)
    this.outputText += "Model generated at '" + saveModelPath + "'.\n"
    this.outputText += s"\n Execution Time : ${trainingTimeInSecond / 1000000000.0} seconds"
  }

  def predictByModel(sc: SparkContext, inputDataPath: String, savedModelPath: String, predictResultPath: String): String ={
    if(this.model == null){
      this.model = MatrixFactorizationModel.load(sc, savedModelPath)
    }

    // Load and parse the data
    val data = sc.textFile(inputDataPath)

    // Evaluate the model on rating data
    val ratings = data.map(_.split(',') match { case Array(user, item, rate) =>
      Rating(user.toInt, item.toInt, rate.toDouble)
    })
    val usersProducts = ratings.map { case Rating(user, product, rate) =>
      (user, product)
    }

    // Predict
    println("predictions : ")
    val initStartPredictTime = System.nanoTime()
    val predictions = model.predict(usersProducts).map { case Rating(user, product, rate) =>
      (user, product, rate)
    }
    val predictTimeInSecond = (System.nanoTime() - initStartPredictTime)

    // Save result
    predictions.saveAsTextFile((predictResultPath + "ALSResult/"))

    // Show sample result
    var predictionResult: String = "Result saved at "+ predictResultPath +
      "ALSResult/" + "\nSample hasil prediksi (top 100):\n"
    predictions.takeOrdered(100).foreach { case (user, product, rate) =>
      predictionResult+= user + "\n"
    }
    predictionResult += s"\n Execution Time : ${predictTimeInSecond / 1000000000.0} seconds"
    predictionResult
  }
}
