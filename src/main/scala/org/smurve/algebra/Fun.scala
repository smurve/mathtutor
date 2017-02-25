package org.smurve.algebra

import org.smurve.complex.Cpx

abstract class Fun(val evalF: Cpx => Cpx) {

  def apply(x: Cpx): Cpx = evalF(x)

  def apply(inner: Fun) = new InFun(this, inner)

  def +(other: Fun): Fun = new AddFun(this, other)

  def *(other: Fun): Fun = new MulFun(this, other)

  def -(other: Fun): Fun = new MinFun(this, other)

  def /(other: Fun): Fun = new DivFun(this, other)

  def °(exponent: Integer): Fun = new ExpFun(this, exponent)

  def unary_- : Fun = new NegFun(this)

  def d: Fun

  def toContextString(context: String ): String
}

object zero extends Fun(_ => 0) {
  def d: Fun = this

  override def toString = "0"

  def toContextString(context: String): String = toString
}

class Const(z: Cpx) extends Fun(_ => z) {
  def d: Fun = zero
  def toContextString(context: String): String = toString
  override def toString: String = z.toString
}

object x extends Fun(x => x) {
  def d = new Const(1)

  def toContextString(context: String): String = toString
  override def toString: String = "x"
}