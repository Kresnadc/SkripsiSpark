package Controller

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}

object ALSController {
  var outputText: String = ""
  var model: MatrixFactorizationModel = null;

  def startTraining(sc: SparkContext,
             inPath: String,
             saveModelPath: String,
             rank: Int,
             numIteration: Int): Unit = {
    // Load and parse the data
    val data = sc.textFile(inPath)
    val ratings = data.map(_.split(',') match { case Array(user, product, rate) =>
      Rating(user.toInt, product.toInt, rate.toDouble)
    })

    ratings.persist()

    // Build model ALS
    val initStartTrainingTime = System.nanoTime()
    val model = ALS.train(ratings, rank, numIteration, 0.01)
    val trainingTimeInSecond = (System.nanoTime() - initStartTrainingTime)

    // Save and load model
    model.save(sc, saveModelPath)
    this.model = model
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
    val ratings = data.map(_.split(',') match { case Array(user, product, rate) =>
      Rating(user.toInt, product.toInt, rate.toDouble)
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
    predictions.saveAsTextFile(predictResultPath)

    // Show sample result
    var predictionResult: String = "Result saved at "+ predictResultPath +
      "ALSResult/" + "\nSample hasil prediksi (top 100):\n"
    predictionResult += "User, Product, Rating\n"
    predictions.takeOrdered(100).foreach { case (user, product, rate) =>
      predictionResult+= user + ", "+ product +", "+ rate + "\n"
    }
    predictionResult += s"\n Prediction Time : ${predictTimeInSecond / 1000000000.0} seconds"
    TextToFile.saveResultToTextFile("ALSResult.txt", predictionResult)
    predictionResult
  }
}
