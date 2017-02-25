package org.smurve

import org.smurve.complex.Cpx

package object algebra {

  implicit def toFun(n: Int): Fun = Fun( _ => Cpx(n, 0))
  implicit def toFun(z: Cpx): Fun = Fun( _ => z )

  val x = Fun(x=>x)
}
