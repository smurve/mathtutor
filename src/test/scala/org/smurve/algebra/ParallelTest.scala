package org.smurve.algebra

import breeze.linalg.DenseVector
import org.scalatest.{FlatSpec, ShouldMatchers}

/**
  * Created by wgi on 15.03.2017.
  */
class ParallelTest extends FlatSpec with ShouldMatchers {

  def now = System.currentTimeMillis

  "a parallel matrix" should "be much faster than a sequential one" in {

    val r = DenseVector.rand[Double](20000000)
    val m = r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: Nil
    val m1 = m.par

    val t0 = now
    for ( i <- 0 to 100 ) {
      val res = m1.map(_.t * r)
    }
    val d1 = now - t0

    println ( s"parralel: $d1")

    val t1 = now
    for ( i <- 0 to 100 ) {
      val res = m.map(_.t * r)
    }
    val d2 = now - t1

    println ( s"sequential: $d2")


  }
}
