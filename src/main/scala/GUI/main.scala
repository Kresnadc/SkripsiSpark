package GUI

import Controller._
import org.apache.spark.{SparkConf, SparkContext}
import scala.swing.BorderPanel.Position._
import scala.swing.Swing._
import scala.swing.TabbedPane._
import scala.swing._

object MainGUI {
  var sc: SparkContext = null
  var isSparkContextUsed = false

  val statsLabel = new Label("SparkContext Status : -")

  val status_starting = "SparkContext Status : Starting..."
  val status_stopped = "SparkContext Status : Stopped"
  val status_error = "SparkContext Status : Error! SparkContext Stopped"
  val status_running_idle = "SparkContext Status : -"
  val status_running_processing = "SparkContext Status : Processing..."
  val status_running_predicting = "SparkContext Status : Predicting..."

  def main(args: Array[String]): Unit = {
    val mainFrame = new MainFrame {
      title = "Perangkat Lunak Demo Spark MLlib"
      preferredSize = new Dimension(1280, 900)
      visible = true

      /*
       * Create a menu bar
       * set the result as this frame's menu bar.
       */
      menuBar = new MenuBar {
        contents += new Menu("File") {
          contents += new MenuItem(Action("About"){
            Dialog.showMessage(null,"Apache Spark 2.2.0 \n Scala 2.11.8 \n Scala Swing 2.11.0-M7 " +
              "\n \n Author : \n Kresna Dwi Cahyo \n 2014730048 \n Informatics " +
              "\n Parahyangan Catholic University", "Software About", Dialog.Message.Plain, null)
          })
          contents += new Separator
          contents += new MenuItem(Action("Exit") {
            System.exit(0)
          })
        }
        contents += new Menu("SparkContext"){
          contents += new MenuItem(Action("Reset SparkContext") {
            if (!sc.isStopped) {
              sc.stop()
              statsLabel.text = status_stopped
            }
            createSparkContext()
            statsLabel.text = status_starting
            isSparkContextUsed = false
          })

          contents += new MenuItem(Action("Stop SparkContext") {
            sc.stop()
            statsLabel.text = status_stopped
            isSparkContextUsed = false
          })

          contents += new MenuItem(Action("Start SparkContext") {
            if (sc.isStopped) {
              createSparkContext()
              statsLabel.text = status_starting
            }else{
              Dialog.showMessage(null,"SparkContext still running", "SparkContext still running", Dialog.Message.Warning, null)
            }
            isSparkContextUsed = false
          })
        }
      }
      contents = new BorderPanel {
        val tabs = new TabbedPane {
          // Naive Bayes
          val naiveBayes = new BorderPanel(){
            // Genereate Model View variable
            val inputDataPath = new TextField("", 25)
            val outputModelPath = new TextField("", 25)
            val trainingPercent = new ComboBox(Array(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
            val outputTextTA = new TextArea(){ text = ""; lineWrap = true}
            val scrollOutputTextTA = new ScrollPane(outputTextTA) {
              border = CompoundBorder(TitledBorder(EtchedBorder, "Naive Bayes Model Result"), EmptyBorder(5, 5, 5, 10))
            }
            val executeBtn = new Button(Action("Generate Model"){
              if (!isSparkContextUsed) {
                isSparkContextUsed = true
                val thread1 = new Thread {
                  override def run {
                    var inPath = inputDataPath.text.trim
                    var saveModelPath = outputModelPath.text.trim
                    var train: Double = trainingPercent.selection.item
                    var test: Double = 1.0 - train

                    // Compute Model Naive Bayes
                    NaiveBayesController.startTraining( sc, inPath , saveModelPath , train, test)
                    outputTextTA.text = NaiveBayesController.outputText

                    // Update Status
                    statsLabel.text = status_running_idle
                    if(sc.isStopped){
                      statsLabel.text = status_stopped
                    }
                    isSparkContextUsed = false
                  }
                }
                statsLabel.text = status_running_processing
                thread1.start()
              } else{
                Dialog.showMessage(null,"SparkContext still running!", "SparkContext is being used", Dialog.Message.Warning, null)
              }
            })

            // Predict View variable
            val inputPredictPath = new TextField("", 25)
            val outputPredictPath = new TextField("", 25)
            val outputPredictTextTA = new TextArea(){ text = ""; lineWrap = true}
            val scrollOutputPredictTextTA = new ScrollPane(outputPredictTextTA){
              preferredSize = new Dimension(700, 280)
              maximumSize = new Dimension(700, 280)
              border = CompoundBorder(TitledBorder(EtchedBorder, "Predict Result "), EmptyBorder(5, 5, 5, 10))
            }

            val predictBtn = new Button(Action("Predict"){
              if (!isSparkContextUsed) {
                isSparkContextUsed = true
                val thread1 = new Thread {
                  override def run {
                    var saveModelPath = outputModelPath.text.trim
                    var inPath = inputPredictPath.text.trim
                    var outPath = outputPredictPath.text.trim

                    //Compute Prediction Naive Bayes
                    outputPredictTextTA.text = NaiveBayesController.predictByModel(sc, inPath, saveModelPath, outPath)

                    //Update status
                    statsLabel.text = "SparkContext Status : -"
                    if(sc.isStopped){
                      statsLabel.text = "SparkContext Status : Error! SparkContext Stopped"
                    }
                    isSparkContextUsed = false
                  }
                }
                statsLabel.text = "SparkContext Status : Predicting..."
                thread1.start()
              } else{
                Dialog.showMessage(null,"SparkContext is being used!", "SparkContext is being used", Dialog.Message.Warning, null)
              }
            })

            // Display Layout
            layout(
              new Label("Classification - Naive Bay\u00E9s"){
                preferredSize = new Dimension(300, 70)
                horizontalAlignment = Alignment.Center
              }
            ) = North

            layout(
              new SplitPane(Orientation.Horizontal){
                leftComponent = new BoxPanel(Orientation.Vertical){
                  contents += new FlowPanel() {
                    contents += new Label("Model Generator") {
                      preferredSize = new Dimension(300, 70)
                      horizontalAlignment = Alignment.Center
                    }
                  }
                  contents += new FlowPanel() {
                    contents += new GridPanel(1, 3) {
                      contents += new Label("Input Path : "){horizontalAlignment = Alignment.Left}
                      contents += inputDataPath
                    }
                  }
                  contents += new FlowPanel(){
                    contents += new GridPanel(1,3){
                      contents += new Label("Output Path : "){horizontalAlignment = Alignment.Left}
                      contents += outputModelPath
                    }
                  }
                  contents += new FlowPanel(){
                    contents += new GridPanel(1,3){
                      contents += new Label("Training Percentage : "){horizontalAlignment = Alignment.Left}
                      contents += trainingPercent
                    }
                  }
                  contents += new FlowPanel() {
                    contents += executeBtn
                  }
                  contents += new FlowPanel(){
                    contents += new BoxPanel(Orientation.Vertical){
                      contents += scrollOutputTextTA
                      preferredSize = new Dimension(700, 300)
                    }
                  }
                }
                rightComponent = new BoxPanel(Orientation.Vertical){
                  contents += new FlowPanel() {
                    contents += new Label("Predict by Model Generated") {
                      preferredSize = new Dimension(300, 70)
                      horizontalAlignment = Alignment.Center
                    }
                  }
                  contents += new FlowPanel() {
                    contents += new GridPanel(1, 3) {
                      contents += new Label("Input Data Path : "){horizontalAlignment = Alignment.Left}
                      contents += inputPredictPath
                    }
                  }
                  contents += new FlowPanel() {
                    contents += new GridPanel(1, 3) {
                      contents += new Label("Output Data Path : "){horizontalAlignment = Alignment.Left}
                      contents += outputPredictPath
                    }
                  }
                  contents += new FlowPanel() {
                    contents += predictBtn
                  }
                  contents += new FlowPanel(){
                    contents += new BoxPanel(Orientation.Vertical){
                      contents += scrollOutputPredictTextTA
                      preferredSize = new Dimension(700, 300)
                    }
                  }
                }
              }
            ) = Center
          }

          // K-Means
          val kmeans = new FlowPanel(){
            contents += new BorderPanel(){
              val inputPath = new TextField("", 25){ maximumSize = new Dimension( 700, 25)}
              val outputPath = new TextField("", 25){ maximumSize = new Dimension( 500, 25)}
              val clusterCB = new ComboBox(List.range(1, 20))
              val iterationCB = new ComboBox(List.range(1, 20))
              val avgConfCheckBox = new CheckBox("Average")
              val minConfCheckBox = new CheckBox("Minimum")
              val maxConfCheckBox = new CheckBox("Maximum")
              val devConfCheckBox = new CheckBox("Standar Deviation")
              val outputTextTA = new TextArea(){ text = ""; lineWrap = true}
              val scrollOutputTextTA = new ScrollPane(outputTextTA) {
                border = CompoundBorder(TitledBorder(EtchedBorder, "Result"), EmptyBorder(5, 5, 5, 10))
              }
              statsLabel.text = "SparkContext Status : -"
              val executeBtn = new Button(Action("Generate Model"){
                if (!isSparkContextUsed) {
                  isSparkContextUsed = true
                  val thread1 = new Thread {
                    override def run {
                      var inPath = inputPath.text.trim
                      var saveModelPath = outputPath.text.trim
                      var numCluster = clusterCB.selection.item
                      var maxIteration = iterationCB.selection.item
                      var resultConf = Array(avgConfCheckBox.selected,
                          minConfCheckBox.selected,
                          maxConfCheckBox.selected,
                          devConfCheckBox.selected)

                      //Compute K-Means
                      KMeansController.runKMeans(sc, inPath, saveModelPath, numCluster, maxIteration, resultConf)
                      outputTextTA.text = KMeansController.outputText

                      // Status update
                      statsLabel.text = "SparkContext Status : -"
                      if(sc.isStopped){
                        statsLabel.text = "SparkContext Status : Stopped"
                      }
                      isSparkContextUsed = false
                    }
                  }
                  statsLabel.text = "SparkContext Status : Processing..."
                  thread1.start()
                } else{
                  Dialog.showMessage(null,"SparkContext still running!", "SparkContext is being used", Dialog.Message.Warning, null)
                }

              })

              layout(
                new Label("Clustering - K-Means (Modified)"){
                  preferredSize = new Dimension(500, 70)
                }
              ) = North

              layout(
                new BoxPanel(Orientation.Vertical){
                  contents += new FlowPanel() {
                    contents += new GridPanel(1, 3) {
                      contents += new Label("Input Path : ")
                      contents += inputPath
                    }
                  }
                  contents += new FlowPanel(){
                    contents += new GridPanel(1,3){
                      contents += new Label("Output Model Path : ")
                      contents += outputPath
                    }
                  }
                  contents += new FlowPanel(){
                    contents += new GridPanel(1,3){
                      contents += new Label("Num of Cluster : ")
                      contents += clusterCB
                    }
                  }
                  contents += new FlowPanel(){
                    contents += new GridPanel(1,3){
                      contents += new Label("Num of Iteration : ")
                      contents += iterationCB
                    }
                  }
                  contents += new FlowPanel(){
                    contents += new GridPanel(1,3){
                      contents += new Label("Pattern options : "){horizontalAlignment = Alignment.Right}
                      contents += new BoxPanel(Orientation.Vertical) {
                        border = CompoundBorder(TitledBorder(EtchedBorder, ""), EmptyBorder(5, 5, 5, 10))
                        contents.append(avgConfCheckBox, minConfCheckBox, maxConfCheckBox, devConfCheckBox)
                      }
                    }
                  }
                  contents += new FlowPanel() {
                    contents += executeBtn
                  }
                  contents += new FlowPanel(){
                    contents += new BoxPanel(Orientation.Vertical){
                      contents += scrollOutputTextTA
                      preferredSize = new Dimension(700, 500)
                    }
                  }
                }
              ) = Center
            }
          }

          // Statistic
          val statistic = new FlowPanel(){
            contents += new BorderPanel(){
              // Generate Summary Stats variable
              val inputDataPath = new TextField("", 25)
              val outputResultPath = new TextField("", 25)
              val ctnConfCheckBox = new CheckBox("Count")
              val avgConfCheckBox = new CheckBox("Average")
              val minConfCheckBox = new CheckBox("Minimum")
              val maxConfCheckBox = new CheckBox("Maximum")
              val devConfCheckBox = new CheckBox("Variance")
              val outputTextTA = new TextArea(){ text = ""; lineWrap = true}
              val scrollOutputTextTA = new ScrollPane(outputTextTA) {
                border = CompoundBorder(TitledBorder(EtchedBorder, "Summary Statistic Result"), EmptyBorder(5, 5, 5, 10))
              }

              val executeBtn = new Button(Action("Compute Statistic"){
                if (!isSparkContextUsed) {
                  isSparkContextUsed = true
                  val thread1 = new Thread {
                    override def run {
                      var inPath = inputDataPath.text.trim
                      var saveResultPath = outputResultPath.text.trim
                      var resultConf = Array(ctnConfCheckBox.selected,
                        avgConfCheckBox.selected,
                        minConfCheckBox.selected,
                        maxConfCheckBox.selected,
                        devConfCheckBox.selected)

                      // Compute Statistic
                      StatisticController.runStatistic(sc, inPath , saveResultPath, resultConf)
                      outputTextTA.text = StatisticController.outputText

                      // Update Status
                      statsLabel.text = "SparkContext Status : -"
                      if(sc.isStopped){
                        statsLabel.text = "SparkContext Status : Stopped"
                      }
                      isSparkContextUsed = false
                    }
                  }
                  statsLabel.text = "SparkContext Status : Processing..."
                  thread1.start()
                } else{
                  Dialog.showMessage(null,"SparkContext still running!", "SparkContext is being used", Dialog.Message.Warning, null)
                }
              })

              // Display Layout
              layout(
                new Label("Statistics - Summary Statistic"){
                  preferredSize = new Dimension(300, 70)
                  horizontalAlignment = Alignment.Center
                }
              ) = North

              layout(
                new BoxPanel(Orientation.Vertical){
                  contents += new FlowPanel() {
                    contents += new Label("Statistic Generator") {
                      preferredSize = new Dimension(300, 70)
                      horizontalAlignment = Alignment.Center
                    }
                  }
                  contents += new FlowPanel() {
                    contents += new GridPanel(1, 3) {
                      contents += new Label("Input Data Path : "){horizontalAlignment = Alignment.Right}
                      contents += inputDataPath
                    }
                  }
                  contents += new FlowPanel(){
                    contents += new GridPanel(1,3){
                      contents += new Label("Result Save Path : "){horizontalAlignment = Alignment.Right}
                      contents += outputResultPath
                    }
                  }
                  contents += new FlowPanel(){
                    contents += new GridPanel(1,3){
                      contents += new Label("Option : "){horizontalAlignment = Alignment.Right}
                      contents += new BoxPanel(Orientation.Vertical) {
                        border = CompoundBorder(TitledBorder(EtchedBorder, ""), EmptyBorder(5, 5, 5, 10))
                        contents.append(ctnConfCheckBox, avgConfCheckBox, minConfCheckBox, maxConfCheckBox, devConfCheckBox)
                      }
                    }
                  }
                  contents += new FlowPanel() {
                    contents += executeBtn
                  }
                  contents += new FlowPanel(){
                    contents += new BoxPanel(Orientation.Vertical){
                      contents += scrollOutputTextTA
                      preferredSize = new Dimension(700, 300)
                    }
                  }
                }
              ) = Center

              layout(
                new FlowPanel(){
                  contents += statsLabel
                }
              ) = South
            }}

          // PCA - Dimention Reduction
          val dimReduct = new FlowPanel() {
            contents += new BorderPanel() {
              // Genereate variable
              val inputDataPath = new TextField("", 25)
              val outputResultPath = new TextField("", 25)
              val topPCOption = new ComboBox(List.range(1, 20))
              val outputTextTA = new TextArea() {
                text = ""; lineWrap = true
              }
              val scrollOutputTextTA = new ScrollPane(outputTextTA) {
                border = CompoundBorder(TitledBorder(EtchedBorder, "PCA Result"), EmptyBorder(5, 5, 5, 10))
              }
              val executeBtn = new Button(Action("Compute PCA") {
                if (!isSparkContextUsed) {
                  isSparkContextUsed = true
                  val thread1 = new Thread {
                    override def run {
                      var inPath = inputDataPath.text.trim
                      var saveResultPath = outputResultPath.text.trim
                      var topPC: Int = topPCOption.selection.item

                      // Execute PCA
                      PCAController.runPCA(sc, inPath, saveResultPath, topPC)
                      outputTextTA.text = PCAController.outputText

                      // UpdateStatus
                      statsLabel.text = "SparkContext Status : -"
                      if (sc.isStopped) {
                        statsLabel.text = "SparkContext Status : Stopped"
                      }
                      isSparkContextUsed = false
                    }
                  }
                  statsLabel.text = "SparkContext Status : Processing..."
                  thread1.start()
                } else {
                  Dialog.showMessage(null, "SparkContext still running!", "SparkContext is being used", Dialog.Message.Warning, null)
                }
              })

              // Display Layout
              layout(
                new Label("Dimensionality Reduction - Principal Component Analysis (PCA)") {
                  preferredSize = new Dimension(300, 70)
                  horizontalAlignment = Alignment.Center
                }
              ) = North

              layout(
                new BoxPanel(Orientation.Vertical) {
                  contents += new FlowPanel() {
                    contents += new GridPanel(1, 3) {
                      contents += new Label("Input Path : ") {
                        horizontalAlignment = Alignment.Right
                      }
                      contents += inputDataPath
                    }
                  }
                  contents += new FlowPanel() {
                    contents += new GridPanel(1, 3) {
                      contents += new Label("Output Path : ") {
                        horizontalAlignment = Alignment.Right
                      }
                      contents += outputResultPath
                    }
                  }
                  contents += new FlowPanel() {
                    contents += new GridPanel(1, 3) {
                      contents += new Label("Num of Top PC : ")
                      contents += topPCOption
                    }
                  }
                  contents += new FlowPanel() {
                    contents += executeBtn
                  }
                  contents += new FlowPanel() {
                    contents += new BoxPanel(Orientation.Vertical) {
                      contents += scrollOutputTextTA
                      preferredSize = new Dimension(700, 300)
                    }
                  }
                }
              ) = Center

              layout(
                new FlowPanel() {
                  contents += statsLabel
                }
              ) = South
            }
          }

          // ALS - Collaborative filtering
          val collabFil = new BorderPanel() {
              // Genereate Model View variable
              val inputDataPath = new TextField("", 25)
              val outputModelPath = new TextField("", 25)
              val rankNum = new ComboBox(List.range(1, 10))
              val iterationNum = new ComboBox(List.range(1, 20))
              val outputTextTA = new TextArea() {
                text = ""; lineWrap = true
              }
              val scrollOutputTextTA = new ScrollPane(outputTextTA) {
                border = CompoundBorder(TitledBorder(EtchedBorder, "ALS Model Result"), EmptyBorder(5, 5, 5, 10))
              }
              val executeBtn = new Button(Action("Generate Model") {
                if (!isSparkContextUsed) {
                  isSparkContextUsed = true
                  val thread1 = new Thread {
                    override def run {
                      var inPath = inputDataPath.text.trim
                      var saveModelPath = outputModelPath.text.trim
                      var rank: Int = rankNum.selection.item
                      var numIteration: Int = iterationNum.selection.item

                      // Execute ALS
                      ALSController.startTraining(sc, inPath, saveModelPath, rank, numIteration)
                      outputTextTA.text = ALSController.outputText

                      // Update Status
                      statsLabel.text = "SparkContext Status : -"
                      if (sc.isStopped) {
                        statsLabel.text = "SparkContext Status : Stopped"
                      }
                      isSparkContextUsed = false
                    }
                  }
                  statsLabel.text = "SparkContext Status : Processing..."
                  thread1.start()
                } else {
                  Dialog.showMessage(null, "SparkContext still running!", "SparkContext is being used", Dialog.Message.Warning, null)
                }
              })

              // Predict View variable
              val inputPredictPath = new TextField("", 25)
              val outputPredictPath = new TextField("", 25)
              val outputPredictTextTA = new TextArea(){ text = ""; lineWrap = true}
              val scrollOutputPredictTextTA = new ScrollPane(outputPredictTextTA){
                preferredSize = new Dimension(700, 280)
                maximumSize = new Dimension(700, 280)
                border = CompoundBorder(TitledBorder(EtchedBorder, "Predict Result "), EmptyBorder(5, 5, 5, 10))
              }

              val predictBtn = new Button(Action("Predict"){
                if (!isSparkContextUsed) {
                  isSparkContextUsed = true
                  val thread1 = new Thread {
                    override def run {
                      var saveModelPath = outputModelPath.text.trim
                      var inPath = inputPredictPath.text.trim
                      var outPath = outputPredictPath.text.trim

                      //Compute Prediction ALS
                      outputPredictTextTA.text = ALSController.predictByModel(sc, inPath, saveModelPath, outPath)

                      //Update status
                      statsLabel.text = "SparkContext Status : -"
                      if(sc.isStopped){
                        statsLabel.text = "SparkContext Status : Stopped"
                      }
                      isSparkContextUsed = false
                    }
                  }
                  statsLabel.text = "SparkContext Status : Predicting..."
                  thread1.start()
                } else{
                  Dialog.showMessage(null,"SparkContext is being used!", "SparkContext is being used", Dialog.Message.Warning, null)
                }
              })

              // Display Layout
              layout(
                new Label("Collaborative filtering - Alternating Least Squares (ALS)") {
                  preferredSize = new Dimension(300, 70)
                  horizontalAlignment = Alignment.Center
                }
              ) = North

              layout(
                new SplitPane(Orientation.Horizontal){
                  leftComponent = new BoxPanel(Orientation.Vertical){
                    contents += new FlowPanel() {
                      contents += new Label("Model Generator") {
                        preferredSize = new Dimension(300, 70)
                        horizontalAlignment = Alignment.Center
                      }
                    }
                    contents += new FlowPanel() {
                      contents += new GridPanel(1, 3) {
                        contents += new Label("Input Path : "){horizontalAlignment = Alignment.Right}
                        contents += inputDataPath
                      }
                    }
                    contents += new FlowPanel(){
                      contents += new GridPanel(1,3){
                        contents += new Label("Output Model Path : "){horizontalAlignment = Alignment.Right}
                        contents += outputModelPath
                      }
                    }
                    contents += new FlowPanel(){
                      contents += new GridPanel(1,3){
                        contents += new Label("Num of Rank : ")
                        contents += rankNum
                      }
                    }
                    contents += new FlowPanel(){
                      contents += new GridPanel(1,3){
                        contents += new Label("Num of Iteration : ")
                        contents += iterationNum
                      }
                    }
                    contents += new FlowPanel() {
                      contents += executeBtn
                    }
                    contents += new FlowPanel(){
                      contents += new BoxPanel(Orientation.Vertical){
                        contents += scrollOutputTextTA
                        preferredSize = new Dimension(700, 300)
                      }
                    }
                  }
                  rightComponent = new BoxPanel(Orientation.Vertical){
                    contents += new FlowPanel() {
                      contents += new Label("Predict by Model Generated") {
                        preferredSize = new Dimension(300, 70)
                        horizontalAlignment = Alignment.Center
                      }
                    }
                    contents += new FlowPanel() {
                      contents += new GridPanel(1, 3) {
                        contents += new Label("Input Data Path : "){horizontalAlignment = Alignment.Left}
                        contents += inputPredictPath
                      }
                    }
                    contents += new FlowPanel() {
                      contents += new GridPanel(1, 3) {
                        contents += new Label("Result Data Path : "){horizontalAlignment = Alignment.Left}
                        contents += outputPredictPath
                      }
                    }
                    contents += new FlowPanel() {
                      contents += predictBtn
                    }
                    contents += new FlowPanel(){
                      contents += new BoxPanel(Orientation.Vertical){
                        contents += scrollOutputPredictTextTA
                        preferredSize = new Dimension(700, 300)
                      }
                    }
                  }
                }
              ) = Center
            }

          // TF-IDF - Feature Extraction
          val featureExt = new FlowPanel() {
            contents += new BorderPanel() {
              // Genereate Model View variable
              val inputDataPath = new TextField("", 25)
              val outputModelPath = new TextField("", 25)
              val numFeatOption = new ComboBox(List.range(0,30))
              val minDocOption = new ComboBox(List.range(0,10))
              val outputTextTA = new TextArea() {
                text = ""; lineWrap = true
              }
              val scrollOutputTextTA = new ScrollPane(outputTextTA) {
                border = CompoundBorder(TitledBorder(EtchedBorder, "TFIDF Result"), EmptyBorder(5, 5, 5, 10))
              }
              val executeBtn = new Button(Action("Compute TFIDF") {
                if (!isSparkContextUsed) {
                  isSparkContextUsed = true
                  val thread1 = new Thread {
                    override def run {
                      var inPath = inputDataPath.text.trim
                      var saveModelPath = outputModelPath.text.trim
                      var minDocFreq: Int = minDocOption.selection.item
                      var numOfFeature: Int = numFeatOption.selection.item

                      // Compute TF-IDF
                      TFIDFController.runTFIDF(sc, inPath, saveModelPath, numOfFeature, minDocFreq)
                      outputTextTA.text = TFIDFController.outputText

                      // Update Status
                      statsLabel.text = "SparkContext Status : -"
                      if (sc.isStopped) {
                        statsLabel.text = "SparkContext Status : Stopped"
                      }
                      isSparkContextUsed = false
                    }
                  }
                  statsLabel.text = "SparkContext Status : Processing..."
                  thread1.start()
                } else {
                  Dialog.showMessage(null, "SparkContext still running!", "SparkContext is being used", Dialog.Message.Warning, null)
                }
              })

              // Display Layout
              layout(
                new Label("Feature Extraction - TF-IDF") {
                  preferredSize = new Dimension(300, 70)
                  horizontalAlignment = Alignment.Center
                }
              ) = North

              layout(
                new BoxPanel(Orientation.Vertical) {
                  contents += new FlowPanel() {
                    contents += new GridPanel(1, 3) {
                      contents += new Label("Input Path : ") {
                        horizontalAlignment = Alignment.Right
                      }
                      contents += inputDataPath
                    }
                  }
                  contents += new FlowPanel() {
                    contents += new GridPanel(1, 3) {
                      contents += new Label("Output Path : ") {
                        horizontalAlignment = Alignment.Right
                      }
                      contents += outputModelPath
                    }
                  }
                  contents += new FlowPanel() {
                    contents += new GridPanel(1, 3) {
                      contents += new Label("Num Of Feature: ")
                      contents += numFeatOption
                    }
                  }
                  contents += new FlowPanel() {
                    contents += new GridPanel(1, 3) {
                      contents += new Label("IDF Minimum Document frequency: ")
                      contents += minDocOption
                    }
                  }
                  contents += new FlowPanel() {
                    contents += executeBtn
                  }
                  contents += new FlowPanel() {
                    contents += new BoxPanel(Orientation.Vertical) {
                      contents += scrollOutputTextTA
                      preferredSize = new Dimension(700, 300)
                    }
                  }
                }
              ) = Center

              layout(
                new FlowPanel() {
                  contents += statsLabel
                }
              ) = South
            }
          }

          // Tab view Pages
          pages += new Page("Naive Bayes", naiveBayes)
          pages += new Page("K-Means", kmeans)
          pages += new Page("Statistic", statistic)
          pages += new Page("PCA", dimReduct)
          pages += new Page("ALS", collabFil)
          pages += new Page("TF-IDF", featureExt)

          createSparkContext()
        }
        val center = new ScrollPane(tabs)
        layout(center) = Center
        layout(statsLabel) = South
      }
    }
    mainFrame.centerOnScreen()
  }

  def showWarning(): Unit ={
    Dialog.showMessage(null,"SparkContext is still in use!", "SparkContext is being used", Dialog.Message.Warning, null)
  }

  def createSparkContext(): Unit ={
    new Thread {
      override def run {
        statsLabel.text = status_starting
        val conf = new SparkConf().setAppName("MLlib Spark 2.2.0")
          .setMaster("yarn")
          .set("spark.submit.deployMode", "client")
          .set("spark.executor.memory", "5g")
		  .set("spark.executor.memoryOverhead", "600")
		  .set("spark.executor.instances", "2")
        sc = new SparkContext(conf)
        statsLabel.text = status_running_idle
        if(sc.isStopped){
          statsLabel.text = status_stopped
        }
      }
    }.start()
  }
}

