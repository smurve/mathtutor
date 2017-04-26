package org.smurve.deeplearning

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest.{FlatSpec, ShouldMatchers}

/**
  * Created by wgi on 25.04.2017.
  */
class AdamTest extends FlatSpec with ShouldMatchers {

  def f(x: Double): Double = (x-1)*(x-1)
  def fp(x: Double): Double = 2 * (x - 1)

  "An Adam Optimizer" should "find the minimum of a simple function" in {

    val adam = new Adam ( width = 1, alpha = 2.0 )

    var xm: DM = DenseMatrix.fill(1,1){2.0}
    var x = xm(0,0)
    var n = 0
    while ( math.abs(x-1.0) > 1e-6 && n < 1000 ) {
      val gradient = DenseMatrix.fill(1,1){fp(x)}
      val nextStep: DM = adam.nextStep(gradient)
      xm = xm + nextStep
      x = xm(0,0)
      n+=1
    }

    n should be < 1000

  }
}
