package org.smurve.deeplearning.layers

import breeze.linalg.DenseVector
import org.smurve.deeplearning.DV

/**
  * A model for the local receptive field for convolutional layers given a certain input matrix
  *
  * @param input_cols number of columns of the input matrix
  * @param input_rows number of rows of the input matrix
  * @param lrf_cols   number of columns of the local receptive field
  * @param lrf_rows   number of rows of the local receptive field
  * @param weights optional weights, will be random if not provided
  * @param bias optional bias, will be random if not provided
  * @param learn indicate weather the weights shall be fixed or participate in learning.
  */
case class LRFSpec(input_cols: Int, input_rows: Int, lrf_cols: Int, lrf_rows: Int,
                   weights: Option[DV] = None, bias: Option[Double] = None, learn: Boolean = true ) {

  val input_size: Int = input_rows * input_cols
  val fmap_cols: Int = input_cols - lrf_cols + 1
  val fmap_rows: Int = input_rows - lrf_rows + 1
  val fmap_size: Int = fmap_cols * fmap_rows
  val lrf_size: Int = lrf_rows * lrf_cols
  var w: DV = weights.getOrElse(rndDV(lrf_size))
  var b: Double = bias.getOrElse(math.random-0.5)

  private def rndDV ( size: Int ) = DenseVector.rand[Double](size) - DenseVector.fill( size ) {.5}

  /**
    * Array of target neuron indices mapped from all lrf anchors that contain d.
    * "Anchor" denotes the upper left corner of an lrf
    * Used for back-propagation
    */
  @inline
  def lrfTargets(d: Int ): Array[Int] = {
    val xr = ( 0 until lrf_cols).map(c => d % input_cols - c).filter (x => x >= 0 && x < fmap_cols)
    val yr = ( 0 until lrf_rows).map(r => d / input_cols - r).filter (y => y >= 0 && y < fmap_rows)
    val seq = for {
      y <- yr
      x <- xr
    } yield x + y * fmap_cols
    seq.toArray.sorted
  }

  /**
    * feature map coordinates from input coordinates
    * @param d the index of the incoming neuron
    * @return the index of the feature matrix
    */
  @inline
  def tD(d: Int): Int = (d / input_cols) * fmap_cols + d % input_cols

  /**
    * dTF: d from t and f: Domain index from target and field indices
    * calculate the input's index given feature map vector index = t and lrf index f
    * and the local receptive field's index j
    * @param t the index within the resulting feature map
    * @param f the index within the local receptive field (or the weight, if you wish)
    * @return the index within the input vector that is associated with the above
    *
    * @see http://localhost:4000/neural/networks/2017/04/11/yet-another-introduction-to-nn.html#some-formula
    */
  @inline
  def dTF(t: Int, f: Int): Int = dF(f) + dT(t)

  /**
    * @param f the index of the feature
    * @return the feature-index-dependent part of the domain index function
    */
  @inline
  private def dF(f: Int): Int = f % lrf_cols + (f / lrf_cols) * input_cols

  /**
    * @param t the index of the target
    * @return the target-index-dependent part of the domain index function
    */
  @inline
  private def dT(t: Int): Int = (t / fmap_cols) * input_cols + t % fmap_cols

  /**
    * @param input the input vector
    * @param t the index of the neuron on the feature map
    * @return the output value of the t'th neuron
    */
  def calcSingle(input: DV, t: Int): Double = {
    (0 until lrf_size).map(f => {
      val d = dTF(t, f)
      w(f) * input(d)
    }).sum + b
  }


}
