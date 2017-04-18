package org.smurve.deeplearning.utilities

import breeze.linalg.DenseVector
import org.smurve.deeplearning.DV

class SingleSymbolImage(val width: Int, val height: Int, val img: Array[Double], val symbol: Int ) {

  def apply(x: Int, y: Int ): Double = {
    assert(x < width && x >= 0 )
    assert(y < height && y >= 0 )
    img(width * y + x)
  }

  override def toString: String = {
    img.zipWithIndex.map(v_i => {
      charFor(v_i._1) + (if ( v_i._2 % width == width - 1 ) "\n" else "")
    }).mkString("")
  }

  private def charFor(v: Double) = if ( v == 1.0 ) "O " else ". "

  def asDV: DV = new DenseVector[Double](img)
}
