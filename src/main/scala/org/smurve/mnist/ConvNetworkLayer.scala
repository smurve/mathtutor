package org.smurve.mnist

import breeze.linalg.DenseVector
import org.smurve.deeplearning._
import org.smurve.deeplearning.layers.LocalReceptiveFieldSpec


/**
  */
class ConvNetworkLayer(frame: LocalReceptiveFieldSpec,
                       num_features: Int = 1,
                       next: Option[MNISTLayer] = None,
                       costDerivative: Option[(DV, DV) => DV] = None,
                       activation: Activation = a_identity
                      ) extends MNISTLayer(1) {

  def featureMaps: List[String] = List("Finished")


  private var w = Array.fill(num_features)(DenseVector.rand[Double](frame.size))


  private var sum_nabla_w: Array[DV] = zeroes
  private var n = 0

  /**
    * update all weights and biases with the average of the most recently finished sample batch
    *
    * @param eta the learning factor
    */
  override def update(eta: Double): Unit = {
    w = w.zip(sum_nabla_w).map(p => p._1 + p._2 * (eta / n))
    sum_nabla_w = zeroes
    n = 0
    next.foreach(_.update(eta))

  }

  def setFeatures(weights: Array[DV]): Unit = {
    assert(weights.length == num_features)
    assert(weights(0).length == frame.size)
    w = weights
  }

  private def zeroes: Array[DV] =
    Array.fill(num_features)(DenseVector.zeros[Double](frame.size))


  /**
    * calculate the output, and feed it into the next layer, if there is one.
    * return this layer's output, or the next layer's output, if there is one.
    *
    * @param x the input, possibly from the previous layer
    * @return the output of the last layer, which may be this
    */
  override def feedForward(x: DV): DV = {

    next.get.feedForward(output(x))
  }


  /**
    * This is where the actual convolution happens
    *
    * @param f     the index of the feature to be convoluted
    * @param input the layer's input vector
    * @return the resulting feature map
    */
  def convolute(f: Int, input: DV): Array[Double] = {
    (0 until frame.size_featureMap).map(k => {
      (0 until frame.size).map(j => {
        w(f)(j) * input(frame.tau(k, j))
      }).sum
    }).toArray
  }

  /**
    * Calculates the output as a single Vector combining all feature maps
    *
    * @param x the input vector
    * @return
    */
  def output(x: DV): DenseVector[Double] =
    DenseVector((0 until num_features).flatMap(n => convolute(n, x)).toArray[Double])

  /**
    * Feed forward and store the results for back propagation
    *
    * @param x the input vector
    * @param y the desired output                        T
    * @return the weighted delta for back propagation: W  *  d
    */
  override def feedForwardAndPropBack(x: DV, y: DV): DV = {
    n += 1

    val z = output(x)
    val a = activation.fn(z)

    val delta_l = if (next.isEmpty)
      costDerivative.get.apply(a, y)
    else {
      val wd = next.get.feedForwardAndPropBack(a, y)
      wd :* activation.deriv(z)
    }

    val dC_dw = (0 until num_features).map(
      n => { // for each feature map
        val delta: Array[Double] = (0 until frame.size).map(
          j => // fix the index of the weight
            (0 until frame.size_featureMap).map(
              k => { // sum up all components of x that contribute to w_j
                val kn = k + n * frame.size_featureMap
                delta_l(kn) * x(frame.tau(k, j))
              }).sum).toArray

        DenseVector(delta)
      })

    // sum up the contribution of this input vector to nabla w
    sum_nabla_w = sum_nabla_w.zip(dC_dw).map(p => p._1 + p._2)


    null // up to now, we can't support back prop from here
  }

}
