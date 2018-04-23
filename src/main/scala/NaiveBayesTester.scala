import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD


object NaiveBayesMain {
  var outputText: String = ""
  var model: NaiveBayesModel = null;

//  def main(args: Array[String]): Unit = {
//  inPath = "E:/InputTest/Iris.csv"
//  saveModelPath = "E:/Output/ModelNaive/"
//  trainingPercent = 0.6
//  testPercent = 0.4
//    runNaiveBayes("E:/InputTest/Iris.csv", "E:/Output/ModelNaive/", 0.6, 0.4)
//  }

  def startTraining(sc: SparkContext,
                    inPath: String,
                    saveModelPath: String,
                    trainingPercent: Double,
                    testPercent: Double): Unit = {
    //Start Training
    println("Start Training")

    // Load data file.
    println("Load and parse the data file.")
    val data = preprocessingDataIrisCSV(sc, inPath)

    //Split data training (Double cth: 40% => 0.4)
    val Array(training, test) = data.randomSplit(Array(trainingPercent, testPercent))

    //Train model
    val initStartTrainingTime = System.nanoTime()
    val modelNaive = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")
    val trainingTimeInSecond = (System.nanoTime() - initStartTrainingTime)

    //Test model with testPercent
    val initStartTestTime = System.nanoTime()
    if(modelNaive == null){
      println("model null")
    }

    val predictionAndLabel = test.map(p =>
      (modelNaive.predict(p.features), p.label)
    )
    val testTimeInSecond = (System.nanoTime() - initStartTestTime)
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    outputText += "Test Accuracy : " + accuracy +"\n"

    // Save and load model
    modelNaive.save(sc, saveModelPath)
    this.model = modelNaive
    outputText += "Trained Model Saved! Location at '"+ saveModelPath +"'\n"
    outputText += "Data Labels : \n"
    this.model.labels.foreach(outputText += _ +"\n")

    outputText += "\nSplit data into training ("+ (trainingPercent * 100).toInt +
      "%) and test ("+ (testPercent * 100).toInt +"%)\n"
    outputText += "Test Time : " + (trainingTimeInSecond / 1000000000.0) +"(Second) \n"
    outputText += "Training Time : " + (trainingTimeInSecond / 1000000000.0) +"(Second)\n"
  }

  def predictByModel(sc: SparkContext, inputDataPath: String, savedModelPath: String, predictResultPath: String): String ={
    val data = preprocessingPredictDataIrisCSV(sc, inputDataPath)

    if(this.model == null){
      val model = NaiveBayesModel.load(sc, savedModelPath)
    }

    var result = this.model.predict(data)
    result.saveAsTextFile(predictResultPath)

    var predictionResult: String = "Result saved at "+ predictResultPath +
      "\nSample preditction result(top 100):\n"
    result.takeOrdered(100).foreach(kelas => predictionResult+= kelas + "\n")
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

