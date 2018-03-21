import javax.swing.filechooser.FileNameExtensionFilter
import javax.swing.{BorderFactory, JFileChooser, SwingWorker}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}


import scala.swing._
import scala.swing.event._
import scala.swing.BorderPanel.Position._
import scala.io.{BufferedSource, Source}


object myGUI extends SimpleSwingApplication {
  val inputFileTF = new TextField("E:\\", 25){ preferredSize = new Dimension( 300, 25)}
  val outputFileTF = new TextField("E:\\", 25){ preferredSize = new Dimension( 300, 25)}
  val outputTextTA = new TextArea(){ text = ""; lineWrap = true; rows = 10; columns = 30 }
  val scrollOutputTextTA = new ScrollPane(outputTextTA)
  val loadingIcon = new Label("Loading Icon"){visible = false}

  var outputTextFile : BufferedSource = null

  override def top: Frame = new MainFrame{
    preferredSize = new Dimension(1000,750)
    title = "2014730048"
    contents = new BorderPanel(){
      layout(
          new Label("Apache Spark v2.2.0"){
            preferredSize = new Dimension(500, 70)
          }
      ) = North

      layout(
        new BoxPanel(Orientation.Vertical) {
          contents += new FlowPanel(){
            contents += new Label("Choose :")
            contents += new ComboBox(Seq("KMeans","Naive Bayes"))
          }
        }
      ) = West

      layout(
        new BoxPanel(Orientation.Vertical){
          contents += new FlowPanel() {
            contents += new GridPanel(1, 3) {
              contents += new Label("Input Path : ")
              contents += inputFileTF
              contents += new Button() {
                action = new Action("Browse") {
                  override def apply(): Unit = {
                    //java.awt.Desktop.getDesktop().browse(java.net.URI.create("http://google.com"))
                    inputFileTF.text = runFileChooser(true)
                  }
                }
              }
            }
          }
          contents += new FlowPanel(){
            contents += new GridPanel(1,3){
              contents += new Label("Output Model Path : ")
              contents += outputFileTF
              contents += new Button(){
                action = new Action("Browse") {
                  override def apply(): Unit = {
                    outputFileTF.text = runFileChooser(false)
                  }
                }
              }
            }
          }
          contents += new FlowPanel() {
            contents += new Button(){
              action = new Action("Execute") {
                override def apply(): Unit = {
                  loadingIcon.text = "Executing..."
                  loadingIcon.visible = true
                  new SwingWorker[Unit, String]() {
                    override def doInBackground(): Unit =
                      runKMeans(inputFileTF.text, outputFileTF.text, 3, 3)
                    override def done(): Unit =
                      loadingIcon.text = "Completed"
                  }.execute()
                }
              }
            }
          }
          contents += new FlowPanel() {
            contents += loadingIcon
          }
          contents += new FlowPanel(){
            contents += new BoxPanel(Orientation.Vertical){
              contents += scrollOutputTextTA
              border = BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("Output Text"),
                BorderFactory.createEmptyBorder(5,5,5,5)
              )
              preferredSize = new Dimension(550, 350)
            }
          }
        }
      ) = Center

      layout(
        new FlowPanel(){
          contents += new Label("============== &&& ==============")
        }
      ) = South
    }
  }

  /**
    *
    * @param inputPath
    * @param outputPath
    * @param numClusters
    * @param numIterations
    */
  def runKMeans(inputPath: String, outputPath: String, numClusters: Int, numIterations: Int) ={
    //Create SparkContext
    val conf = new SparkConf().setAppName("KMeansExample").setMaster("local")
    val sc = new SparkContext(conf)

    // Load and parse the data
    // val data = sc.textFile("hdfs://localhost:9001/user/hadoop/iris-dataset")
    val input = "E:/InputTest/sample_mllib_kmeans_data.txt"
    val data = sc.textFile(input)
    //println("Element of RDD: "+ data.count())
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()
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
    outputTextFile = Source.fromFile("OutputTextKMEANS.txt")
    outputTextTA.text = outputTextFile.mkString

    // New Cluster Center
    println("Cluster Centers: ")
    sameModel.clusterCenters.foreach(println)
    var centroid = ""
    for( i <- 1 to sameModel.clusterCenters.size ){
      centroid += "\nCentroid Kluster "+ i +" : "+ sameModel.clusterCenters(i-1)
    }
    outputTextTA.text += "\nCentroid :"
    outputTextTA.text += centroid
    sc.stop()
  }

  /**
    *
    */
  def runNaiveBayes: Unit ={
    val conf = new SparkConf().setAppName("NaiveBayesExample").setMaster("local")
    val sc = new SparkContext(conf)
    // Contoh :

    // Load and parse the data file.
    println("Load and parse the data file.")
    // Sparse data format LibSVM
    val data = org.apache.spark.mllib.util.MLUtils.loadLibSVMFile(sc, "E:/InputTest/sample_mllib_naive.txt")
    //data.collect().foreach(println)
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

  /**
    *
    * @param boolean
    * @return
    */
  def runFileChooser(boolean: Boolean) : String = {
      var chooser = new JFileChooser();
      chooser.setCurrentDirectory(new java.io.File("."));
      chooser.setDialogTitle("Open File");
      if (boolean) {
        chooser.setFileFilter(new FileNameExtensionFilter("Text File(.txt)","txt"))
        chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
      }else{
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
      }
      chooser.setAcceptAllFileFilterUsed(true);
      if (chooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
        System.out.println("getCurrentDirectory(): " + chooser.getCurrentDirectory());
        System.out.println("getSelectedFile() : " + chooser.getSelectedFile());
        (chooser.getSelectedFile() + "")
      } else {
        System.out.println("No Selection ");
        ("")
      }
  }

  //  def initGUI() {
  //    val frame  = new Frame  { title = "Spark 2.2.0 Application GUI"    }
  //    val button = new Button { text  = "test button" }
  //    val uslPanel = new BoxPanel(Orientation.Vertical) {
  //      contents += button
  //    }
  //    val r = new Reactor {}
  //    r.listenTo(button)
  //    r.reactions += {
  //      case event.ButtonClicked(_) => println("Clicked")
  //    }
  //    frame.contents = uslPanel
  //    frame.visible  = true  // or use `frame.open()`
  //  }
}
