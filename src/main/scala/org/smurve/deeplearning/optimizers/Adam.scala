package org.smurve.deeplearning.optimizers

import breeze.linalg.DenseMatrix
import breeze.numerics.sqrt
import org.smurve.deeplearning.stats.NNStats
import org.smurve.deeplearning.{DM, DV}

/**
  * Adaptive Moment optimizer: Kingma/Lei Ba 2015
  * @param alpha hyper param
  * @param beta1 hyper param
  * @param beta2 hyper param
  * @param epsilon hyper param
  */
class Adam ( alpha: Double = 1E-3, beta1: Double =0.9, beta2: Double = 0.999, epsilon: Double = 1e-8
           ) extends Optimizer {

  private var m: DM = _
  private var v: DM = _
  private var width: Int = _

  private var beta1_t = 1.0
  private var beta2_t = 1.0


  private def initialize ( firstGradient: DM ) = {
    width = firstGradient.cols
    val height = firstGradient.rows
    m = DenseMatrix.fill( height, width) {0.0}
    v = DenseMatrix.fill( height, width) {0.0}
  }

  def nextStep ( gt: DM ) : DM = {

    if ( m == null ) {
      initialize(gt)
    }

    m = m * beta1 + gt * ( 1- beta1)
    v = beta2 * v + ( 1 - beta2 ) * (gt :* gt)
    beta1_t *= beta1
    beta2_t *= beta2
    val mth: DM = m / (1-beta1_t)
    val vth: DM = v / (1-beta2_t)
    val res = - mth * alpha :/ ( sqrt(vth) :+ epsilon )
    res
  }

}

