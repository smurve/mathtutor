package org.smurve.algebra

import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.algebra.fun.{Fun, x}
import org.smurve.complex.Cpx

class AlgebraTest extends FlatSpec with ShouldMatchers {

  "The expression 2 * ( x + 1 )" should " create the expected function " in {
    val f: Fun = 2 * ( x + 1 )
    f(3) should be ( Cpx(8, 0))
  }
}
