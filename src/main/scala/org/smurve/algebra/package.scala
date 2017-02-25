package org.smurve

import org.smurve.algebra.fun.{Const, Fun}
import org.smurve.complex.Cpx

package object algebra {

  implicit def toFun(x: Double): Fun = new Const( Cpx(x, 0))
  implicit def toFun(n: Int): Fun = new Const( Cpx(n, 0))
  implicit def toFun(z: Cpx): Fun = new Const( z )

}
