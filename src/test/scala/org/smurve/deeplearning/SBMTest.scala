package org.smurve.deeplearning

import breeze.linalg.DenseVector
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.deeplearning.optimizers.SignumBasedMomentum

/**
  * Created by wgi on 25.04.2017.
  */
class SBMTest extends FlatSpec with ShouldMatchers {

  def f(x: Double): Double = (x - 1) * (x - 1)

  def fp(x: Double): Double = 2 * (x - 1)

  "An SBM Optimizer" should "find the minimum of a simple function" in {

    val o = new SignumBasedMomentum(maxMomentum = 10, eta = 0.05)
    var xm = DenseVector(2.1)
    var x = xm(0)
    var n = 0
    while (math.abs(x - 1.0) > 1e-6 && n < 1000) {
      val gradient = DenseVector(fp(x))
      val nextStep = o.nextStep(gradient)
      xm = xm - nextStep
      x = xm(0)
      n += 1
    }

    n should be < 1000
    println(s"Good enough after: $n steps")
  }

}
