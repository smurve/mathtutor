package org.smurve.algebra

import org.smurve.algebra.fun.{Const, x}
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure

class EqualsTest extends Specification {
  override def is: SpecStructure =
    s2"""
        Constant 1 = Constant 1 $constantEquals
        Constant 1 != Constant 2 $constantNotEquals
        x == x                   $xEqualsX
        2*x == 2*x               $prodEqualsProd
        -f == -f                 $minusF_minusF
        x/2 == x/2               $divEqualsDiv
        x+2 == x+2               $sumEqualsSum
        x-2 == x-2               $minusEqualsMinus
        x°2 == x°2               $expEqualsExp
      """


  private def constantEquals = {

    new Const(1) == new Const(1) must beTrue
  }
  private def constantNotEquals = {

    new Const(2) !=  new Const(1) must beTrue
  }

  private def xEqualsX = {

    val f1 = x
    val f2 = x
    f1.equals(f2) must beTrue
  }

  private def prodEqualsProd = {

    val f1 = 2*x
    val f2 = 2*x
    f1.equals(f2) must beTrue
  }

  private def divEqualsDiv = {

    val f1 = 2/x
    val f2 = 2/x
    f1.equals(f2) must beTrue
  }

  private def sumEqualsSum = {

    val f1 = 2+x
    val f2 = 2+x
    f1.equals(f2) must beTrue
  }

  private def minusEqualsMinus = {

    val f1 = 2-x
    val f2 = 2-x
    f1.equals(f2) must beTrue
  }

  private def minusF_minusF = {

    val f1 = -(2*x)
    val f2 = -(2*x)
    f1.equals(f2) must beTrue
  }

  private def expEqualsExp = {

    val f1 = x°2
    val f2 = x°2
    f1.equals(f2) must beTrue
  }
}


