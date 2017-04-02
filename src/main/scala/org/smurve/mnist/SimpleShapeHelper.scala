package org.smurve.mnist

import breeze.linalg.DenseVector
import org.smurve.deeplearning._

/**
  * A simple problem: the largest sum of any two weights determines, whether the "image" is somewhat vertical,
  * diagonal or horizontal. E.g. if UL + LR is the largest sum of the 6 possible combinations, then it's a diag.
  *
  *    --------------
  *   |       |      |
  *   |   UL  |  UR  |
  *   |       |      |
  *   |-------|------|
  *   |       |      |
  *   |   LL  |  LR  |
  *   |       |      |
  *    --------------
  */
object SimpleShapeHelper {

  def nextSample : (DV, DV) = {

    val x: DV = DenseVector.rand(4)
    (x, classify(x))
  }

  private def classify ( x: DV ) : DV = {
    val ul = x.data(0)
    val ur = x.data(1)
    val ll = x.data(2)
    val lr = x.data(3)

    Array (
      ("D", ul + lr),
      ("D", ur + ll),
      ("H", ur + ul),
      ("H", lr + ll),
      ("V", ul + ll),
      ("V", ur + lr)
    ).maxBy(_._2)._1 match {
      case "D" => DenseVector(1.0, 0.0, 0.0)
      case "H" => DenseVector(0.0, 1.0, 0.0)
      case "V" => DenseVector(0.0, 0.0, 1.0)
    }
  }

}
