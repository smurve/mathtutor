package org.smurve.algebra

import org.smurve.algebra.fun.{Fun, x}
import org.smurve.complex.Cpx
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure

/**
  * Created by wgiersche on 25/02/17.
  */
class FunTest extends Specification {
  override def is: SpecStructure =
    s2"""
          $complexFunction
    """

  private def complexFunction  = {

    val f: Fun = x ° 3 + 2 * x°2 - x - 1
    val g = 2 * f

    g(1) must equalTo(Cpx(2,0))
  }
}
