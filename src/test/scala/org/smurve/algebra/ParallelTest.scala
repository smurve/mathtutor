package org.smurve.algebra

import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest.{FlatSpec, ShouldMatchers}

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

  "a parallel matrix" should "be much faster than a sequential one" in {

    val r = DenseVector.rand[Double](50000) // go for at least 5 million to have reasonable load
    val m = r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: r :: Nil
    val m1 = m.par
    val m2 = DenseMatrix(r,r,r  ,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r)

    val batchSize = 1000

    // Warm up
    for ( _ <- 0 to 10 ) {
      val res = m1.map(_.t * r)
    }


    val t0 = now
    for ( _ <- 0 to batchSize ) {
      val res = m1.map(_.t * r)
    }
    val d1 = now - t0

    println ( s"parallel: $d1")

    val t1 = now
    for ( _ <- 0 to batchSize ) {
      val res = m.map(_.t * r)
    }
    val d2 = now - t1

    println ( s"sequential: $d2")


    val t2 = now
    for ( _ <- 0 to batchSize ) {
      val res = m2 * r
    }
    val d3 = now - t2

    println ( s"dense matrix: $d3")


  }
}
