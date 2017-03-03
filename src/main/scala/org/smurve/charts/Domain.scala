package org.smurve.charts

import scala.collection.immutable.IndexedSeq

case class Domain(left: Double, right: Double, lower: Double, upper: Double, numX: Int, numY: Int) {

  val dx: Double = (right - left) / numX
  val dy: Double = (upper - lower) / numY

  val values: IndexedSeq[(Double, Double)] = {
    for {
      nx <- 0 until numX
      ny <- 0 until numY
    } yield
      (left + nx * dx, lower + ny * dy)
  }
}

