package org.smurve

/**
  * Created by wgiersche on 25/02/17.
  */
package object complex {

  implicit def toDouble ( n: Int ): Double = n.toDouble

  implicit def toComplex ( n: Int ): Cpx = Cpx ( n,  0)
  implicit def toComplex ( d: Double ): Cpx = Cpx ( d,  0)
  //implicit def toComplex ( p: (Double, Double)): Cpx = Cpx(p._1, p._2)
  implicit def toComplex ( p: (Int, Int)): Cpx = Cpx(p._1, p._2)

  val i = Cpx ( 0, 1 )
}
