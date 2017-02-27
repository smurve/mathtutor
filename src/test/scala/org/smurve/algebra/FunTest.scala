package org.smurve.algebra

import org.scalatest._
import org.smurve.algebra.fun.x
import org.smurve.complex.Cpx

/**
  * Created by wgiersche on 25/02/17.
  */
class FunTest extends FlatSpec with ShouldMatchers {

  "(°) " should "create an exponential function like in x°2" in {
    val f = (x + 1) ° 2
    f(0) should equal( Cpx(1,0))
  }


}
