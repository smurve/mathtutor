package org.smurve.algebra

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.mnist._

/**
  *   Testing various parameters. Running this with or without netlib on the classpath,
  *   calculation times in seconds for 10'000'000 columns times 20 rows, on a 4-core machine
  *
  *         with netlib   without netlib
  *   par:    6             4
  *   seq:    25            13
  *   mat:    14            20
  *
  *   Only matrix calculations actually benefit from netlib, still they perform way below the approach
  *   with parallel computations for the rows
  */
class ParallelTest extends FlatSpec with ShouldMatchers {

  def now = System.currentTimeMillis


  "parmul(m, v) " should "be a correct matrix * vector mulitplication" in {

    val m = DenseMatrix( (1.0, 2.0, 3.0), (3.0, 2.0, 1.0 ), (1.0, 1.0, 1.0))
    val v = DenseVector( 1.0, -1.0, 1.0 )
    val res = parMul(m, v)
    val vgl = m * v
    res should be (vgl)

  }

  "a parallel matrix" should "be much faster than a sequential one" in {

    val r = DenseVector.rand[Double](5000) // go for at least 5 million to have reasonable load
    val m =
      r :: r :: r :: r :: r ::
      r :: r :: r :: r :: r ::
        r :: r :: r :: r :: r ::
        r :: r :: r :: r :: r :: Nil

    val m1 = m.par

    val m2 = DenseMatrix(
      r,r,r,r,r,
      r,r,r,r,r,
      r,r,r,r,r,
      r,r,r,r,r)

    val batchSize = 100

    // Warm up: loading optimized system libs happens here
    for ( _ <- 0 to 10 ) {
      m2 * r
    }


    val t0 = now
    for ( _ <- 0 to batchSize ) {
      val res = m1.map(_.t * r)
    }
    val d0 = now - t0
    println ( s"parallel without conversion: $d0")


    val t1 = now
    for ( _ <- 0 to batchSize ) {
      val res = parMul(m2,r)
    }
    val d1 = now - t1

    println ( s"parallel with conversion: $d1")


    val t2 = now
    for ( _ <- 0 to batchSize ) {
      val res = m.map(_.t * r)
    }
    val d2 = now - t2

    println ( s"sequential: $d2")


    val t3 = now
    for ( _ <- 0 to batchSize ) {
      val res = m2 * r
    }
    val d3 = now - t3

    println ( s"dense matrix: $d3")


  }
}
