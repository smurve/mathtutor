package org.smurve.algebra

import breeze.linalg.{DenseMatrix, DenseVector}
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure

import scala.util.Random


class VectorTest extends Specification {

  override def is: SpecStructure = {

    s2"""
       |       (AB).t  == B.t A.t $transpose
       |       A + B == B + A     $sumOfMatrices
       |       A.t*A works        $transpose3
      """
  }


  private def transpose = {
    val a = DenseMatrix((1, 2, 3), (0, 1, 2))
    val b = DenseMatrix((1, 2), (2, 1), (3, 2))
    (a * b).t must equalTo(b.t * a.t)
  }

  private def transpose3 = {
    val a = DenseMatrix((1, 2, 3), (0, 1, 2))
    val b = DenseMatrix((1, 2), (2, 1), (3, 2))
    val c = DenseMatrix((1, 2, 3), (0, 1, 2))
    b.t * b must equalTo(DenseMatrix((14, 10), (10, 9)))
  }

  private def sumOfMatrices = {
    val a = DenseMatrix((1, 2, 3), (0, 1, 2))
    val b = DenseMatrix((1, 2, 3), (0, 1, 2))
    a + b must equalTo(b + a)
  }


}

