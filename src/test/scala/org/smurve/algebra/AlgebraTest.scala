package org.smurve.algebra

import org.smurve.algebra.fun.{Fun, x}
import org.smurve.complex.Cpx
import org.specs2.Specification
import org.specs2.matcher.MatchResult
import org.specs2.specification.core.SpecStructure

class AlgebraTest extends Specification {
  override def is: SpecStructure =
    s2"""
         declaring and combining functions $combineFunctions
      """


  private def combineFunctions: MatchResult[Cpx] = {
    val f: Fun = 2 * ( x + 1 )
    f(3) must equalTo ( Cpx(8, 0))

  }
}
