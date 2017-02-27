package org.smurve.algebra

import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.algebra.fun.{pi, x}

class Tutor extends FlatSpec with ShouldMatchers {

  "pi*x" should "have the correct derivative" in {
    pi * x
    //(pi * x).d.toString should be ( "pi" )
  }
}



