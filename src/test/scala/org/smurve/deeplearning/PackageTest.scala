package org.smurve.deeplearning

import breeze.linalg.DenseVector
import org.scalatest.{FlatSpec, ShouldMatchers}

/**
  * Created by wgiersche on 08/04/17.
  */
class PackageTest extends FlatSpec with ShouldMatchers {

  private val d = 1E-7

  "the cross-entropy" should "be close to zero at a=y" in {
    val a = DenseVector(1.0-d, d)
    val y = DenseVector(1.0, 0.0)
    val ce = crossEntropyCost(a, y)
    ce < d * 10 should be ( true )
  }

  "The cross-entropy's derivative" should "be approximated by a difference" in {

    val da1 = DenseVector(d,0)
    val da2 = DenseVector(0,d)

    val a = DenseVector(0.8, 0.2)
    val y = DenseVector(1.0, 0.0)

    val dCda1_approx = (crossEntropyCost(a + da1, y) - crossEntropyCost(a, y)) / d
    val dCda2_approx = (crossEntropyCost(a + da2, y) - crossEntropyCost(a, y)) / d

    val nablaC = crossEntropyCostDerivative(a, y)

    (nablaC - dCda1_approx).data(0) < 10 * d should be (true)
    (nablaC - dCda2_approx).data(1) < 10 * d should be (true)

    println(nablaC)
  }
}
