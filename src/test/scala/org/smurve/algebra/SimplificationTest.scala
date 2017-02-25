package org.smurve.algebra

import org.smurve.algebra.fun.{Fun, x, zero}
import org.smurve.complex.Cpx
import org.specs2.Specification
import org.specs2.matcher.MatchResult
import org.specs2.specification.core.SpecStructure

class SimplificationTest extends Specification {
  override def is: SpecStructure =
    s2"""
         x + 0 = x x_plus_0
         x - 0 = x  x_minus_0
         x - x = 0  x_minus_x
         f - f = 0  $f_minus_f
      """

  def others = """
  x + 2 - 1 = x + 1 $x_plus_2_minus_1
  """

  private def x_plus_0: MatchResult[Fun] = {
    (x + 0) must equalTo ( x )
  }
  private def x_minus_0: MatchResult[Fun] = {
    (x - 0) must equalTo ( x )
  }
  private def x_minus_x: MatchResult[Fun] = {
    (x - x) must equalTo ( zero )
  }
  private def f_minus_f: MatchResult[Fun] = {
    val f = 2*x
    val g = 2*x
    (f - g) must equalTo ( zero )
  }










  // others for later treatment
  private def x_plus_2_minus_1: MatchResult[Fun] = {
    (x + 2 - 1).simplified must equalTo ( x + 1 )
  }
}
