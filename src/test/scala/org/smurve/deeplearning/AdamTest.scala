package org.smurve.deeplearning

import breeze.linalg.DenseVector
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.deeplearning.optimizers.Adam

/**
  * Created by wgi on 25.04.2017.
  */
class AdamTest extends FlatSpec with ShouldMatchers {

  def f(x: Double): Double = (x-1)*(x-1)
  def fp(x: Double): Double = 2 * (x - 1)

  "An Adam Optimizer" should "find the minimum of a simple function" in {

    val adam = new Adam ( height = 1, alpha = 2.0 )

    var xm = DenseVector.fill(1){2.0}
    var x = xm(0)
    var n = 0
    while ( math.abs(x-1.0) > 1e-6 && n < 1000 ) {
      val gradient = DenseVector.fill(1){fp(x)}
      val nextStep = adam.nextStep(gradient)
      xm = xm + nextStep
      x = xm(0)
      n+=1
    }

    println(s"Good enough after: $n steps")
    n should be < 1000

  }
}
