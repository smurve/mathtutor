package org.smurve.mnist

import breeze.linalg.{*, DenseMatrix}

import scala.collection.immutable.Seq

/**
  */
class ConvNetworkLayer(input_cols: Int, input_rows: Int,
                       window_cols: Int, window_rows: Int,
                       num_features: Int,
                       next: Option[ConvNetworkLayer] = None,
                       initWith: InitWith = INIT_WITH_RANDOM,
                       costDerivative: (DV, DV)=>DV,
                       activation: Activation
             ) extends Layer {


  private val window_size = window_rows * window_cols
  private var w: DM = DenseMatrix.rand(num_features, window_size)

  private val feature_cols = input_cols - window_cols + 1
  private val feature_size = feature_cols * ( input_rows - window_rows + 1)
  private var n: Int = 0

  /**
    * update all weights and biases with the average of the most recently finished sample batch
    * @param eta the learning factor
    */
  override def update ( eta: Double ): Unit = {
  }


  /**
    * calculate the output, and feed it into the next layer, if there is one.
    * return this layer's output, or the next layer's output, if there is one.
    * @param x the input, possibly from the previous layer
    * @return the output of the last layer, which may be this
    */
  override def feedForward ( x: DV ) : DV = {

    val a: DV = null

    val fmaps: Array[Array[Double]] = ( 0 until num_features).map(n=>calcFMap(n, x)).toArray


    if (next.isEmpty) {
      a
    } else {
      next.get.feedForward(a)
    }
  }

  def calcFMap ( n: Int, input: DV ): Array[Double] = {
    ( 0 until feature_size ).map ( k => {
      (0 until window_size).map(j => {
        w(n, j) * input(tau(k, j))
      }).sum
    }).toArray

  }

  /**
    * Feed forward and store the results for back propagation
    * @param x the input vector
    * @param y the desired output                        T
    * @return the weighted delta for back propagation: W  *  d
    */
  override def feedForwardAndPropBack(x: DV, y: DV): DV = {
    null
  }



  @inline
  def tau ( k: Int, j: Int ) : Int = phi ( j ) + xi ( k )

  @inline
  def phi ( j: Int ) : Int = j % window_cols + j / window_cols * window_cols

  @inline
  def xi ( k: Int ) : Int = k / feature_cols * input_cols + k % feature_cols

}
