package org.smurve.complex

import org.specs2.Specification
import org.specs2.specification.core.SpecStructure

/**
  * Created by wgiersche on 25/02/17.
  */
class ComplexTest extends Specification {

  override def is : SpecStructure =
    s2"""
      Algebratic sum (+) ${algebraicSum()}
      Multiplication (*) ${multiplication()}
      Division (/)       ${division()}
      Minus (-)          ${minus()}
      Unary Minus (-)    ${unaryMinus()}
      Exponentiation     ${exponentiation()}
    """

    private def algebraicSum() = {
      val z1: Cpx = (4, 2)
      val z2: Cpx = 5 + 3 * i
      z1 + z2 must equalTo ( Cpx(9, 5))
      1 + z1 must equalTo ( Cpx(5,2))
    }

  private def multiplication() = {
    val z1: Cpx = (4, 2)
    val z2: Cpx = (5, 3)
    z1 * z2 must equalTo ( Cpx(14, 22))
    2 * z1 must equalTo ( Cpx(8,4))
  }

  private def division() = {
    val z1: Cpx = (10, 15)
    val z2: Cpx = (2,1)
    z1 / 5 must equalTo ( Cpx(2,3))
    z1 / z2 must equalTo ( Cpx(7,4))
    z1 / (1,2) must equalTo ( Cpx ( 8, -1))
  }

  private def unaryMinus () = {
    val z2 = Cpx ( 2,1 )
    -z2 must equalTo( Cpx (-2, -1 ))
  }

  private def minus () = {
    val z1 = Cpx ( 3,4 )
    val z2: Cpx = (2,1)
    z1 - z2 must equalTo ( Cpx ( 1, 3))
  }

  private def exponentiation() = {
    val z1 = Cpx ( 2, 0 )
    2 to 9 foreach { n => z1 ^ n must equalTo ( Cpx(2^n, 0))}
    z1 ^ 0 must equalTo ( Cpx ( 1, 0 ))
    z1 ^ 1 must equalTo ( z1 )
  }
}
