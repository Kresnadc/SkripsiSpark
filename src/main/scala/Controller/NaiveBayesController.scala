package Controller

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  *
  */
object NaiveBayesController {
  var outputText: String = ""
  var model: NaiveBayesModel = null;

  def startTraining(sc: SparkContext,
                    inPath: String,
                    saveModelPath: String,
                    trainingPercent: Double,
                    testPercent: Double): Unit = {
    // Load data file.
    val data = preprocessingDataIrisCSV(sc, inPath)

    // Split data training
    val Array(training, test) = data.randomSplit(Array(trainingPercent, testPercent))

    // Train model
    val initStartTrainingTime = System.nanoTime()
    val modelNaive = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")
    val trainingTimeInSecond = (System.nanoTime() - initStartTrainingTime)

    // Test model with testPercent
    val initStartTestTime = System.nanoTime()
    val predictionAndLabel = test.map(p =>
      (modelNaive.predict(p.features), p.label)
    )
    val testTimeInSecond = (System.nanoTime() - initStartTestTime)
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    outputText += "Test Accuracy : " + accuracy +"\n"

    // Save and load model
    modelNaive.save(sc, saveModelPath)
    this.model = modelNaive
    outputText = "Trained Model Saved! Location at '"+ saveModelPath +"'\n"
    outputText += "Data Labels : \n"
    this.model.labels.foreach(outputText += _ +"\n")

    outputText += "\nSplit data into training ("+ (trainingPercent * 100).toInt +
      "%) and test ("+ (testPercent * 100).toInt +"%)\n"
    outputText += "Training Time : " + (trainingTimeInSecond / 1000000000.0) +"(Second)\n"
    outputText += "Test Time : " + (testTimeInSecond / 1000000000.0) +"(Second) \n"
  }

  def predictByModel(sc: SparkContext,
                     inputDataPath: String,
                     savedModelPath: String,
                     predictResultPath: String): String ={
    val data = preprocessingPredictDataIrisCSV(sc, inputDataPath)
    //Check model loaded
    if(this.model == null){
      this.model = NaiveBayesModel.load(sc, savedModelPath)
    }

    //Start predict time
    val initStartPredictTime = System.nanoTime()
    var result = this.model.predict(data)
    val predictTimeInSecond = (System.nanoTime() - initStartPredictTime)

    result.saveAsTextFile(predictResultPath)

    var predictionResult: String = "Result saved at "+ predictResultPath +
      "\nPrediction result:\n"
    result.takeOrdered(100).foreach(kelas => predictionResult+= kelas + "\n")
    predictionResult += "Prediction Time : " + (predictTimeInSecond / 1000000000.0) +"(Second)\n"
    predictionResult
  }

  def preprocessingDataIrisCSV(sc: SparkContext, path: String) : RDD[LabeledPoint]= {
    //Read the file
    //Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
    val csv = sc.textFile(path)  // original file

    //To find the headers
    val header = csv.first;

    //To remove the header
    val data = csv.filter(_(0) != header(0));

    //To create a RDD of (label, features) pairs
    data.map { line =>
      val parts = line.split(',')
      LabeledPoint(defineClassIris(parts(5)), Vectors.dense(
        parts(1).toDouble,
        parts(2).toDouble,
        parts(3).toDouble,
        parts(4).toDouble))
    }.cache()
  }

  def preprocessingPredictDataIrisCSV(sc: SparkContext, path: String) : RDD[Vector]= {
    //Read the file
    //Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm,Species
    val csv = sc.textFile(path)  // original file

    //Find the headers
    val header = csv.first;

    //Remove the header
    val data = csv.filter(_(0) != header(0));

    //To create a RDD of (label, features) pairs
    data.map { line =>
      val parts = line.split(',')
      Vectors.dense(
        parts(1).toDouble,
        parts(2).toDouble,
        parts(3).toDouble,
        parts(4).toDouble)
    }.cache()
  }

  def defineClassIris(label : String): Double = {
    // Iris-setosa, Iris-versicolor, Iris-virginica
    if(label == "Iris-setosa"){
      1.0
    }else if(label == "Iris-versicolor"){
      2.0
    }else {
      3.0
    }
  }
}

