package org.smurve.deeplearning.optimizers

import breeze.linalg.{DenseMatrix, min}
import org.smurve.deeplearning.{DM, DV}

/**
  * Adaptive Moment optimizer self-made
  * @param acc acceleration factor
  * @param dec deceleration factor
  * @param maxMomentum max momentum
  */
class SignumBasedMomentum(acc: Double = 1.2, dec: Double =0.3, maxMomentum: Double = 100.0
           ) extends Optimizer {

  private var one: DM = _
  private var m: DM = _
  private var width: Int = _

  private var ps: DM = _


  private def initialize ( firstGradient: DM ) = {
    width = firstGradient.cols
    val height = firstGradient.rows
    ps = breeze.numerics.signum(firstGradient)
    one = DenseMatrix.fill( height, width) {1.0}
    m = one
  }

  def nextStep ( gt: DM ) : DM = {

    if ( m == null ) {
      initialize(gt)
    }

    val cs: DM = breeze.numerics.signum(gt)
    val pscs: DM = cs :* ps
    val fwd: DM = ( pscs :+ one ) * 0.5
    val bwd: DM = one - fwd
    ps = cs
    m = m :* ((fwd * acc) + (bwd * dec))
    m = min ( m, one * maxMomentum )
    val res =  - cs :* m
    res
  }

  /**
    * Convenience wrapper for DenseVectors
    * @param gt: The gradient
    * @return the next delta to update the given weights
    */
  def nextStep ( gt: DV ) : DV = {
    nextStep ( DenseMatrix(gt)).apply(0,::).t
  }
}

