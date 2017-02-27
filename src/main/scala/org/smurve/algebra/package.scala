package org.smurve

import org.smurve.algebra.fun.{Const, Fun}
import org.smurve.complex.Cpx
import breeze.linalg.{ DenseVector => DV}

package object algebra {

  implicit def toFun(x: Double): Fun = new Const( Cpx(x, 0))
  implicit def toFun(n: Int): Fun = new Const( Cpx(n, 0))
  implicit def toFun(z: Cpx): Fun = new Const( z )
  implicit def toDouble(n: Int): Double = n.toDouble
  implicit def toDVDouble(dvi: DV[Int]): DV[Double] = new DV(dvi.data.map(_.toDouble))

}
