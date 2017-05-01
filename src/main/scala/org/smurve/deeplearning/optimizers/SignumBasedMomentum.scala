package org.smurve.deeplearning.optimizers

import breeze.linalg.{DenseMatrix, min, sum}
import org.smurve.deeplearning.stats.NNStats
import org.smurve.deeplearning.{DM, DV}

/**
  * Adaptive Moment optimizer self-made
  * @param acc acceleration factor
  * @param dec deceleration factor
  * @param maxMomentum max momentum
  */
class SignumBasedMomentum(acc: Double = 1.2, dec: Double =0.3, maxMomentum: Double = 100.0,
                          epsilon:Double = 1e-8, var eta: Double
                         ) extends Optimizer {

  private var one: DM = _
  private var m: DM = _
  private var width: Int = _

  /** the number of parameters */
  private var ND: Double = _

  private var ps: DM = _

  private var epsilon_s: DM = _

  private def initialize (firstGradient: DM ) = {
    width = firstGradient.cols
    val height = firstGradient.rows
    ps = breeze.numerics.signum(firstGradient)
    one = DenseMatrix.fill( height, width) {1.0}
    m = one
    ND = sum(one)
    epsilon_s = one * 1e-16
  }

  def nextStep (gt: DM ) : DM = {

    if ( m == null ) {
      initialize(gt)
    }

    val cs: DM = breeze.numerics.signum(gt)
    //val pscs: DM = cs :* ps
    //val fwd: DM = ( pscs :+ one ) * 0.5
    //val bwd: DM = one - fwd
    val (fwd,bwd) = fb( ps, cs )
    ps = cs
    m = m :* ((fwd * acc) + (bwd * dec))
    m = min ( m, one * maxMomentum )
    val res = cs :* m * eta
    res
  }

  /**
    * @param ps previous signum vector
    * @param cs currenct signum vector
    * @return vectors for fwd and bwd momentum
    */
  private def fb ( ps: DM, cs: DM ): (DenseMatrix[Double], DenseMatrix[Double]) = {

    ( DenseMatrix.tabulate(ps.rows, ps.cols)((i,j)=> { forward(ps(i,j), cs(i,j))}),
    DenseMatrix.tabulate(ps.rows, ps.cols)((i,j)=> { backward(ps(i,j), cs(i,j))}))
  }


  private def forward ( ps: Double, cs: Double ): Double = {
    if ( cs == 0 ) 0.0 else if ( ps == 0) 1.0 else (1+ps*cs)/2
  }
  private def backward ( ps: Double, cs: Double ): Double = {
    if ( cs == 0 ) 0.0 else if ( ps == 0) 0.0 else (1-ps*cs)/2
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

