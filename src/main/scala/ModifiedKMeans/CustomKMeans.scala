package ModifiedKMeans

import scala.collection.mutable.ArrayBuffer

//import org.apache.spark.annotation.Since
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.clustering.{KMeans => NewKMeans}
//import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.mllib.linalg.{Vector, Vectors}
//import org.apache.spark.mllib.linalg.BLAS.{axpy, scal}
//import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

//import org.apache.spark.util.Utils
//import org.apache.spark.util.random.XORShiftRandom
//import org.apache.spark.mllib.clustering.LocalKMeans
import java.io._
import java.util.Random

import org.apache.spark.mllib.clustering.KMeansModel

class CustomKMeans (
                     private var k: Int,
                     private var maxIterations: Int,
                     private var initializationMode: String,
                     private var initializationSteps: Int,
                     private var epsilon: Double,
                     private var seed: Long,
                     private var outputText: String,
                     private var arrResult: Array[Array[Array[Double]]],
                     private var resultConf: Array[Boolean]) extends Serializable with Logging {

  /**
    * Constructs a KMeans instance with default parameters: {k: 2, maxIterations: 20,
    * initializationMode: "k-means||", initializationSteps: 2, epsilon: 1e-4, seed: random}.
    */
  def this() = this(2, 20, CustomKMeans.K_MEANS_PARALLEL, 2, 1e-4, new Random().nextLong(), "", null, null)

  /**
    * Method untuk print out ke File
    * @param inputText text that will printed out to file
    * @return param
    */
  def printToFileTxt(inputText: String)={
    // Menggunakan Java io
    val pw = new PrintWriter(new File("OutputTextKMEANS.txt" ))
    pw.write(inputText)
    pw.close
    (inputText)
  }

  /**
    * Result of last iteration (arrResult).
    *
    */
  def getResult: Array[Array[Array[Double]]] = arrResult

  def getResultConf: Array[Boolean] = resultConf

  def setResultConf(conf: Array[Boolean]): this.type ={
    this.resultConf = conf
    this
  }

  /**
    * Number of clusters to create (k).
    *
    * @note It is possible for fewer than k clusters to
    * be returned, for example, if there are fewer than k distinct points to cluster.
    */
  def getK: Int = k

  /**
    * Set the number of clusters to create (k).
    *
    * @note It is possible for fewer than k clusters to
    * be returned, for example, if there are fewer than k distinct points to cluster. Default: 2.
    */
  def setK(k: Int): this.type = {
    require(k > 0,
      s"Number of clusters must be positive but got ${k}")
    this.k = k
    this
  }

  /**
    * Maximum number of iterations allowed.
    */
  def getMaxIterations: Int = maxIterations

  /**
    * Set maximum number of iterations allowed. Default: 20.
    */
  def setMaxIterations(maxIterations: Int): this.type = {
    require(maxIterations >= 0,
      s"Maximum of iterations must be nonnegative but got ${maxIterations}")
    this.maxIterations = maxIterations
    this
  }

  /**
    * The initialization algorithm. This can be either "random" or "k-means||".
    */
  def getInitializationMode: String = initializationMode

  /**
    * Set the initialization algorithm. This can be either "random" to choose random points as
    * initial cluster centers, or "k-means||" to use a parallel variant of k-means++
    * (Bahmani et al., Scalable K-Means++, VLDB 2012). Default: k-means||.
    */
  def setInitializationMode(initializationMode: String): this.type = {
    CustomKMeans.validateInitMode(initializationMode)
    this.initializationMode = initializationMode
    this
  }


  /**
    * Number of steps for the k-means|| initialization mode
    */

  def getInitializationSteps: Int = initializationSteps

  /**
    * Set the number of steps for the k-means|| initialization mode. This is an advanced
    * setting -- the default of 2 is almost always enough. Default: 2.
    */

  def setInitializationSteps(initializationSteps: Int): this.type = {
    require(initializationSteps > 0,
      s"Number of initialization steps must be positive but got ${initializationSteps}")
    this.initializationSteps = initializationSteps
    this
  }

  /**
    * The distance threshold within which we've consider centers to have converged.
    */
  def getEpsilon: Double = epsilon

  /**
    * Set the distance threshold within which we've consider centers to have converged.
    * If all centers move less than this Euclidean distance, we stop iterating one run.
    */
  def setEpsilon(epsilon: Double): this.type = {
    require(epsilon >= 0,
      s"Distance threshold must be nonnegative but got ${epsilon}")
    this.epsilon = epsilon
    this
  }

  /**
    * The random seed for cluster initialization.
    */
  def getSeed: Long = seed

  /**
    * Set the random seed for cluster initialization.
    */
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  // Initial cluster centers can be provided as a KMeansModel object rather than using the
  // random or k-means|| initializationMode
  private var initialModel: Option[KMeansModel] = None

  /**
    * Set the initial starting point, bypassing the random initialization or k-means||
    * The condition model.k == this.k must be met, failure results
    * in an IllegalArgumentException.
    */
  def setInitialModel(model: KMeansModel): this.type = {
    require(model.k == k, "mismatched cluster count")
    initialModel = Some(model)
    this
  }

  /**
    * Train a K-means model on the given set of points; `data` should be cached for high
    * performance, because this is an iterative algorithm.
    */
  def run(data: RDD[Vector]): KMeansModel = {
    run(data, None)
  }

  private def run(
                   data: RDD[Vector],
                   instr: Option[Instrumentation[NewKMeans]]): KMeansModel = {

    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    // Compute squared norms and cache them.
    val norms = data.map(Vectors.norm(_, 2.0))
    norms.persist()
    val zippedData = data.zip(norms).map { case (v, norm) =>
      new VectorWithNorm(v, norm)
    }
    val model = runAlgorithm(zippedData, instr)
    norms.unpersist()
    println("KMeans Model Generated")
    // Warn at the end of the run as well, for increased visibility.
    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data was not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }
    model
  }

  /**
    * Implementation of K-Means algorithm.
    */
  private def runAlgorithm(
                            data: RDD[VectorWithNorm],
                            instr: Option[Instrumentation[NewKMeans]]): KMeansModel = {

    val sc = data.sparkContext

    val initStartTime = System.nanoTime()

    val centers = initialModel match {
      case Some(kMeansCenters) =>
        kMeansCenters.clusterCenters.map(new VectorWithNorm(_))
      case None =>
        if (initializationMode == CustomKMeans.RANDOM) {
          initRandom(data)
        } else {
          initKMeansParallel(data)
        }
    }
    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
    logInfo(f"Initialization with $initializationMode took $initTimeInSeconds%.3f seconds.")

    var converged = false
    var cost = 0.0
    var iteration = 0

    val iterationStartTime = System.nanoTime()

    arrResult = new Array[Array[Array[Double]]](this.k)

    instr.foreach(_.logNumFeatures(centers.head.vector.size))

    // Execute iterations of Lloyd's algorithm until converged
    while (iteration < maxIterations && !converged) {
      val costAccum = sc.doubleAccumulator
      val bcCenters = sc.broadcast(centers) //broadcast agar bisa dibaca secara distributed

      // Find the new centers
      val totalContribs = data.mapPartitions { points =>
        val thisCenters = bcCenters.value
        val dims = thisCenters.head.vector.size

        val sums = Array.fill(thisCenters.length)(Vectors.zeros(dims))
        val counts = Array.fill(thisCenters.length)(0L)
        val mins = Array.fill(thisCenters.length)(Vectors.dense(Array.fill(dims)(Double.PositiveInfinity)))
        val maxs = Array.fill(thisCenters.length)(Vectors.zeros(dims))
        val devs = Array.fill(thisCenters.length)(Vectors.zeros(dims)) // Jumlah dari sum kuadrat

        points.foreach { point =>
          val (bestCenter, cost) = CustomKMeans.findClosest(thisCenters, point)
          costAccum.add(cost)
          val sum = sums(bestCenter)
          BLAS.axpy(1.0, point.vector, sum)
          counts(bestCenter) += 1

          // Minimum
          var arrMin = new Array[Double](dims)
          for( i <- 0 to dims-1){
            arrMin(i) = math.min(point.vector.apply(i), mins(bestCenter).apply(i))
          }
          mins(bestCenter) = Vectors.dense(arrMin)

          // Maximum
          var arrMax = new Array[Double](dims)
          for( i <- 0 to dims-1){
            arrMax(i) = math.max(point.vector.apply(i), maxs(bestCenter).apply(i))
          }
          maxs(bestCenter) = Vectors.dense(arrMax)

          // Deviasi
          val dev = devs(bestCenter)
          var arrDev = new Array[Double](dims)
          for( i <- 0 to dims-1){
            arrDev(i) = point.vector.apply(i) * point.vector.apply(i)
          }
          BLAS.axpy(1.0, Vectors.dense(arrDev), dev)
        }
        counts.indices.filter(counts(_) > 0).map(j => (j, (sums(j), counts(j), mins(j), maxs(j), devs(j)))).iterator
      }.reduceByKey { case ((sum1, count1, min1, max1, dev1),
      (sum2, count2, min2, max2, dev2)) =>
        BLAS.axpy(1.0, sum2, sum1)
        // Minimum dan Maximum atribut
        var arrMin = new Array[Double](sum1.size)
        var arrMax = new Array[Double](sum1.size)
        for( i <- 0 to sum1.size-1){
          arrMin(i) = math.min(min1.apply(i), min2.apply(i))
          arrMax(i) = math.max(max1.apply(i), max2.apply(i))
        }
        var minRes: Vector = Vectors.dense(arrMin)
        var maxRes: Vector = Vectors.dense(arrMax)
        // Deviasi atribut
        BLAS.axpy(1.0, dev2, dev1)

        (sum1, count1 + count2, minRes, maxRes, dev1)
      }.collectAsMap()

      //bcCenters.destroy(blocking = false)
      bcCenters.destroy()

      //Update the cluster centers and costs
      converged = true
      totalContribs.foreach { case (j, (sum, count, min, max, dev)) =>
        var sumCopy = sum.toArray
        BLAS.scal(1.0 / count, sum)
        val newCenter = new VectorWithNorm(sum)
        if (converged && CustomKMeans.fastSquaredDistance(newCenter, centers(j)) > epsilon * epsilon) {
          converged = false
        }
        centers(j) = newCenter
        println(s"Center $j")
        newCenter.vector.toArray.foreach(println)

        if(iteration == maxIterations || !converged){
          // Print Cluster dan Jumlah Object
          //println("\nKlaster Ke-: "+ j)
          //println("Jumlah Objek Klaster ke-" + j + " : " + count)

          var dims = sum.size
          var mins : Array[Double] = null
          var maxs : Array[Double] = null
          var mean : Array[Double] = null
          var deviasi : Array[Double] = null

          //If conf not set, Set all true
          if(resultConf == null){
            resultConf = Array.fill[Boolean](4)(true)
          }
          if(resultConf(0)){
            mean = Array.fill[Double](dims)(0)
            println(s"Mean $j:")
            for (i <- 0 until dims) {
              mean(i) = (sum.apply(i))
              println(mean(i))
              //mean(i) =
              // Print Average
              println("Rata-rata atribut " + (i + 1) + " : " + mean(i))
              //this.outputText += "Rata-rata atribut " + (i + 1) + " : " + mean(i) +"\n"
            }
          }
          if(resultConf(1)){
            mins = min.toArray
            for (i <- 0 to dims - 1) {
              // Print Mininimum
              println("Minimum atribut " + (i + 1) + " : " + min.apply(i))
              //this.outputText += "Minimum atribut " + (i + 1) + " : " + min(i) +"\n"
            }
          }
          if(resultConf(2)){
            maxs = max.toArray
            for (i <- 0 to dims - 1) {
              // Print Max
              println("Maximum atribut " + (i + 1) + " : " + max.apply(i))
              //this.outputText += "Maximum atribut " + (i + 1) + " : " + max(i) +"\n"
            }
          }
          if(resultConf(3)){
            deviasi = Array.fill[Double](dims)(0)
            for (i <- 0 to dims - 1) {
              deviasi(i) = scala.math.sqrt(((count * dev(i)) - (sumCopy(i) * sumCopy(i))) / (count * (count - 1)))
              println("Deviasi atribut "+ (i+1) + " : " + deviasi(i))
              //this.outputText += "Deviasi atribut "+ (i+1) + " : " + deviasi(i)+"\n"
            }
          }
          arrResult(j) = Array(Array(count), mean, mins, maxs, deviasi)
        }
      }
      cost = costAccum.value
      iteration += 1
    }

    val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
    logInfo(f"Iterations took $iterationTimeInSeconds%.3f seconds.")
    println(f"Iterations took $iterationTimeInSeconds%.3f seconds.")

    if (iteration == maxIterations) {
      logInfo(s"KMeans reached the max number of iterations: $maxIterations.")
      println(s"KMeans reached the max number of iterations: $maxIterations.")
    } else {
      logInfo(s"KMeans converged in $iteration iterations.")
      println(s"KMeans converged in $iteration iterations.")
    }

    logInfo(s"The cost is $cost.")
    println(s"The cost is $cost.")

    var strResult = ""
    for (i <- 0 until arrResult.length) {
      strResult += s"\nPattern ${i+1} :("
      for (j <- 0 until arrResult(i).length) {
        if (arrResult(i)(j) != null) {
          strResult += "["
          for (k <- 0 until arrResult(i)(j).length) {
            strResult += arrResult(i)(j)(k)
            if (k < arrResult(i)(j).length - 1) {
              strResult += ", "
            }
          }
          if (j < arrResult(i).length - 1) {
            strResult += "], "
          } else {
            strResult += "]"
          }
        }
      }
      strResult += ")\n"
    }


    this.outputText += "Pattern Result : \n" + strResult + "\n"
    this.outputText += "KMeans converged in " + iteration + " iterations.\n"
    this.outputText += f"K-Means iteration time : $iterationTimeInSeconds%.3f seconds.\n"
    this.outputText += "Cost : " + f"$cost%.3f" + "\n"
    printToFileTxt(outputText)

    new KMeansModel(centers.map(_.vector))
  }

  /**
    * Initialize a set of cluster centers at random.
    */
  private def initRandom(data: RDD[VectorWithNorm]): Array[VectorWithNorm] = {
    // Select without replacement; may still produce duplicates if the data has < k distinct
    // points, so deduplicate the centroids to match the behavior of k-means|| in the same situation
    data.takeSample(false, k, new XORShiftRandom(this.seed).nextInt())
      .map(_.vector).distinct.map(new VectorWithNorm(_))
  }

  /**
    * Initialize a set of cluster centers using the k-means|| algorithm by Bahmani et al.
    * (Bahmani et al., Scalable K-Means++, VLDB 2012). This is a variant of k-means++ that tries
    * to find dissimilar cluster centers by starting with a random center and then doing
    * passes where more centers are chosen with probability proportional to their squared distance
    * to the current cluster set. It results in a provable approximation to an optimal clustering.
    *
    * The original paper can be found at http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf.
    */
  private def initKMeansParallel(data: RDD[VectorWithNorm]): Array[VectorWithNorm] = {
    // Initialize empty centers and point costs.
    var costs = data.map(_ => Double.PositiveInfinity)

    // Initialize the first center to a random point.
    val seed = new XORShiftRandom(this.seed).nextInt()
    val sample = data.takeSample(false, 1, seed)
    // Could be empty if data is empty; fail with a better message early:
    require(sample.nonEmpty, s"No samples available from $data")

    val centers = ArrayBuffer[VectorWithNorm]()
    var newCenters = Seq(sample.head.toDense)
    centers ++= newCenters

    // On each step, sample 2 * k points on average with probability proportional
    // to their squared distance from the centers. Note that only distances between points
    // and new centers are computed in each iteration.
    var step = 0
    val bcNewCentersList = ArrayBuffer[Broadcast[_]]()
    while (step < initializationSteps) {
      val bcNewCenters = data.context.broadcast(newCenters)
      bcNewCentersList += bcNewCenters
      val preCosts = costs
      costs = data.zip(preCosts).map { case (point, cost) =>
        math.min(CustomKMeans.pointCost(bcNewCenters.value, point), cost)
      }.persist(StorageLevel.MEMORY_AND_DISK)
      val sumCosts = costs.sum()

      bcNewCenters.unpersist(blocking = false)
      preCosts.unpersist(blocking = false)

      val chosen = data.zip(costs).mapPartitionsWithIndex { (index, pointCosts) =>
        val rand = new XORShiftRandom(seed ^ (step << 16) ^ index)
        pointCosts.filter { case (_, c) => rand.nextDouble() < 2.0 * c * k / sumCosts }.map(_._1)
      }.collect()
      newCenters = chosen.map(_.toDense)
      centers ++= newCenters
      step += 1
    }

    costs.unpersist(blocking = false)
    //bcNewCentersList.foreach(_.destroy(false))

    val distinctCenters = centers.map(_.vector).distinct.map(new VectorWithNorm(_))

    if (distinctCenters.size <= k) {
      distinctCenters.toArray
    } else {
      // Finally, we might have a set of more than k distinct candidate centers; weight each
      // candidate by the number of points in the dataset mapping to it and run a local k-means++
      // on the weighted centers to pick k of them
      val bcCenters = data.context.broadcast(distinctCenters)
      val countMap = data.map(CustomKMeans.findClosest(bcCenters.value, _)._1).countByValue()

      //bcCenters.destroy(blocking = false)

      val myWeights = distinctCenters.indices.map(countMap.getOrElse(_, 0L).toDouble).toArray
      LocalKMeans.kMeansPlusPlus(0, distinctCenters.toArray, myWeights, k, 30)
    }
  }
}

object CustomKMeans {
  val RANDOM = "random"
  val K_MEANS_PARALLEL = "k-means||"

  /**
    * Trains a k-means model using the given set of parameters.
    *
    * @param data Training points as an `RDD` of `Vector` types.
    * @param k Number of clusters to create.
    * @param maxIterations Maximum number of iterations allowed.
    * @param initializationMode The initialization algorithm. This can either be "random" or
    *                           "k-means||". (default: "k-means||")
    * @param seed Random seed for cluster initialization. Default is to generate seed based
    *             on system time.
    */
  def train(
             data: RDD[Vector],
             k: Int,
             maxIterations: Int,
             initializationMode: String,
             seed: Long): KMeansModel = {
    new CustomKMeans().setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .setSeed(seed)
      .run(data)
  }

  /**
    * Trains a k-means model using the given set of parameters.
    *
    * @param data Training points as an `RDD` of `Vector` types.
    * @param k Number of clusters to create.
    * @param maxIterations Maximum number of iterations allowed.
    * @param initializationMode The initialization algorithm. This can either be "random" or
    *                           "k-means||". (default: "k-means||")
    */
  def train(
             data: RDD[Vector],
             k: Int,
             maxIterations: Int,
             initializationMode: String): KMeansModel = {
    new CustomKMeans().setK(k)
      .setMaxIterations(maxIterations)
      .setInitializationMode(initializationMode)
      .run(data)
  }

  /**
    * Trains a k-means model using specified parameters and the default values for unspecified.
    */
  def train(
             data: RDD[Vector],
             k: Int,
             maxIterations: Int): KMeansModel = {
    new CustomKMeans().setK(k)
      .setMaxIterations(maxIterations)
      .run(data)
  }

  /**
    * Trains a k-means model using specified parameters and the default values for unspecified.
    */
  def train(
             data: RDD[Vector],
             k: Int,
             maxIterations: Int,
             resultConf: Array[Boolean]): KMeansModel = {
    new CustomKMeans().setK(k)
      .setMaxIterations(maxIterations)
      .setResultConf(resultConf)
      .run(data)
  }


  /**
    * Returns the index of the closest center to the given point, as well as the squared distance.
    */
  def findClosest(
                   centers: TraversableOnce[VectorWithNorm],
                   point: VectorWithNorm): (Int, Double) = {
    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0
    var indexForeach = 0
    centers.foreach { center =>
      // Since `\|a - b\| \geq |\|a\| - \|b\||`, we can use this lower bound to avoid unnecessary
      // distance computation.
      indexForeach += 1
      var lowerBoundOfSqDist = center.norm - point.norm
      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist
      if (lowerBoundOfSqDist < bestDistance) {
        val distance: Double = fastSquaredDistance(center, point)
        //println("distance ="+distance)
        if (distance < bestDistance) {
          bestDistance = distance
          bestIndex = i
        }
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }

  /**
    * Returns the K-means cost of a given point against the given cluster centers.
    */
  def pointCost(
                 centers: TraversableOnce[VectorWithNorm],
                 point: VectorWithNorm): Double =
    findClosest(centers, point)._2

  /**
    * Returns the squared Euclidean distance between two vectors computed by
    * [[org.apache.spark.mllib.util.MLUtils#fastSquaredDistance]].
    */
  def fastSquaredDistance(
                           v1: VectorWithNorm,
                           v2: VectorWithNorm): Double = {
    MLUtils.fastSquaredDistance(v1.vector, v1.norm, v2.vector, v2.norm)
  }

  private def validateInitMode(initMode: String): Boolean = {
    initMode match {
      case CustomKMeans.RANDOM => true
      case CustomKMeans.K_MEANS_PARALLEL => true
      case _ => false
    }
  }
}

class VectorWithNorm(val vector: Vector, val norm: Double) extends Serializable {

  def this(vector: Vector) = this(vector, Vectors.norm(vector, 2.0))

  def this(array: Array[Double]) = this(Vectors.dense(array))

  /** Converts the vector to a dense vector. */
  def toDense: VectorWithNorm = new VectorWithNorm(Vectors.dense(vector.toArray), norm)
}
