package org.smurve.algebra

import org.smurve.algebra.fun.{Fun, x, zero}
import org.specs2.Specification
import org.specs2.matcher.MatchResult
import org.specs2.specification.core.SpecStructure

class Tutor extends Specification {
  override def is: SpecStructure =
    s2"""
         $nextTest
      """


  private def nextTest: MatchResult[Any] = {
    val f = 3*(2*(x+1))

    f.toString must equalTo ( "6*(x+1)" )
  }
}



