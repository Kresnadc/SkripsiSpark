package ModifiedKMeans

import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.Vectors

import scala.util.Random

/**
  * An utility object to run K-means locally. This is private to the ML package because it's used
  * in the initialization of KMeans but not meant to be publicly exposed.
  */
object LocalKMeans extends Logging {

  /**
    * Run K-means++ on the weighted point set `points`. This first does the K-means++
    * initialization procedure and then rounds of Lloyd's algorithm.
    */
  def kMeansPlusPlus(
                      seed: Int,
                      points: Array[VectorWithNorm],
                      weights: Array[Double],
                      k: Int,
                      maxIterations: Int
                    ): Array[VectorWithNorm] = {
    val rand = new Random(seed)
    val dimensions = points(0).vector.size
    val centers = new Array[VectorWithNorm](k)

    // Initialize centers by sampling using the k-means++ procedure.
    centers(0) = pickWeighted(rand, points, weights).toDense
    val costArray = points.map(CustomKMeans.fastSquaredDistance(_, centers(0)))

    for (i <- 1 until k) {
      val sum = costArray.zip(weights).map(p => p._1 * p._2).sum
      val r = rand.nextDouble() * sum
      var cumulativeScore = 0.0
      var j = 0
      while (j < points.length && cumulativeScore < r) {
        cumulativeScore += weights(j) * costArray(j)
        j += 1
      }
      if (j == 0) {
        logWarning("kMeansPlusPlus initialization ran out of distinct points for centers." +
          s" Using duplicate point for center k = $i.")
        centers(i) = points(0).toDense
      } else {
        centers(i) = points(j - 1).toDense
      }

      // update costArray
      for (p <- points.indices) {
        costArray(p) = math.min(CustomKMeans.fastSquaredDistance(points(p), centers(i)), costArray(p))
      }

    }

    // Run up to maxIterations iterations of Lloyd's algorithm
    val oldClosest = Array.fill(points.length)(-1)
    var iteration = 0
    var moved = true
    while (moved && iteration < maxIterations) {
      moved = false
      val counts = Array.fill(k)(0.0)
      val sums = Array.fill(k)(Vectors.zeros(dimensions))
      var i = 0
      while (i < points.length) {
        val p = points(i)
        val index = CustomKMeans.findClosest(centers, p)._1
        BLAS.axpy(weights(i), p.vector, sums(index))
        counts(index) += weights(i)
        if (index != oldClosest(i)) {
          moved = true
          oldClosest(i) = index
        }
        i += 1
      }
      // Update centers
      var j = 0
      while (j < k) {
        if (counts(j) == 0.0) {
          // Assign center to a random point
          centers(j) = points(rand.nextInt(points.length)).toDense
        } else {
          BLAS.scal(1.0 / counts(j), sums(j))
          centers(j) = new VectorWithNorm(sums(j))
        }
        j += 1
      }
      iteration += 1
    }

    if (iteration == maxIterations) {
      logInfo(s"Local KMeans++ reached the max number of iterations: $maxIterations.")
    } else {
      logInfo(s"Local KMeans++ converged in $iteration iterations.")
    }

    centers
  }

  def pickWeighted[T](rand: Random, data: Array[T], weights: Array[Double]): T = {
    val r = rand.nextDouble() * weights.sum
    var i = 0
    var curWeight = 0.0
    while (i < data.length && curWeight < r) {
      curWeight += weights(i)
      i += 1
    }
    data(i - 1)
  }
}