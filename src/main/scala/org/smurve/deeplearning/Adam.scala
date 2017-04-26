package org.smurve.deeplearning

import breeze.linalg.DenseMatrix

/**
  * Created by wgi on 25.04.2017.
  * defaults as suggested in the original paper.
  */
class Adam ( height: Int = 1, width: Int,
             alpha: Double = 1E-3, beta1: Double =0.9, beta2: Double = 0.999, epsilon: Double = 1e-8 ) {

  private var m = DenseMatrix.fill[Double](height, width){0.0}
  private var v = DenseMatrix.fill[Double](height, width){0.0}
  private var t = 0
  private var beta1_t = 1.0
  private var beta2_t = 1.0

  def nextStep ( gt: DM ) : DM = {
    m = m * beta1 + gt * ( 1- beta1)
    v = beta2 * v + ( 1 - beta2 ) * (gt :* gt)
    beta1_t *= beta1
    beta2_t *= beta2
    val mth: DM = m / (1-beta1_t)
    val vth: DM = v / (1-beta2_t)
    val res = - mth * alpha :/ ( breeze.numerics.sqrt(vth) :+ epsilon )
    res
  }
}
