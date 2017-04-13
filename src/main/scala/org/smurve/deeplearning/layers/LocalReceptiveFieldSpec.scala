package org.smurve.deeplearning.layers

import breeze.linalg.DenseVector
import org.smurve.deeplearning.DV

import scala.collection.immutable

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
case class LocalReceptiveFieldSpec(input_cols: Int, input_rows: Int, lrf_cols: Int, lrf_rows: Int,
                                   weights: Option[DV] = None, bias: Option[DV] = None, learn: Boolean = true ) {

  val input_size: Int = input_rows * input_cols
  val fmap_cols: Int = input_cols - lrf_cols + 1
  val fmap_rows: Int = input_rows - lrf_rows + 1
  val fmap_size: Int = fmap_cols * fmap_rows
  val lrf_size: Int = lrf_rows * lrf_cols
  val w: DV = weights.getOrElse(rndDV(lrf_size))
  val b: DV = bias.getOrElse(rndDV(fmap_size))

  private def rndDV ( size: Int ) = DenseVector.rand[Double](size) - DenseVector.fill( size ) {.5}

  /**
    * Array of target neuron indices mapped from all lrf anchors that contain d.
    * "Anchor" denotes the upper left corner of an lrf
    * Used for back-propagation
    */
  @inline
  def lrfTargets(d: Int ): Array[Int] = {
    val xr = ( 0 until lrf_cols).map(c => d % input_cols - c).filter (_ >= 0 )
    val yr = ( 0 until lrf_rows).map(r => (d / input_cols - r) * fmap_cols).filter (_ >= 0 )
    val seq = for {
      y <- yr
      x <- xr
    } yield x + y
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
    * the feature-index-dependent part of the domain index function
    * @param f the index of the feature
    * @return
    */
  @inline
  private def dF(f: Int): Int = f % lrf_cols + (f / lrf_cols) * input_cols

  /**
    * the target-index-dependent part of the domain index function
    * @param t the index of the target
    * @return
    */
  @inline
  private def dT(t: Int): Int = (t / fmap_cols) * input_cols + t % fmap_cols
}
