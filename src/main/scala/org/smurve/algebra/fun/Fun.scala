package org.smurve.algebra.fun

import org.smurve.complex.Cpx

abstract class Fun(val evalF: Cpx => Cpx) {

  private val PRECISION = 10000000000.0

  def apply(x: Cpx): Cpx = { val r = evalF(x); Cpx( math.round(r.r * PRECISION )/PRECISION, math.round(r.i*PRECISION) / PRECISION)}
  //def apply: (Cpx) => Cpx = evalF

  def apply(inner: Fun): Fun = Fun.simplify(new CompFun(this, inner))

  def +(other: Fun): Fun = Fun.simplify(new SumFun(this, other))

  def *(other: Fun): Fun =
    {
      Fun.simplify(
        new ProdFun(this, other))
    }

  def -(other: Fun): Fun = Fun.simplify(new DiffFun(this, other))

  def /(other: Fun): Fun = Fun.simplify(new FracFun(this, other))

  def °(exponent: Integer): Fun = Fun.simplify(new PowFun(this, exponent))

  def unary_- : Fun = Fun.simplify(new NegFun(this))

  def d: Fun

  def toContextString(context: String): String

  def simplified: Fun

  val context: String
}


object pi extends Const(math.Pi) {
  override def toString = "pi"
}

object e extends Const(math.E) {
  override def toString = "e"
}

object zero extends Const(0) {
  override def toString = "0"
}

object one extends Const(1) {
  override def toString = "1"
}

class Const(z: Cpx) extends Fun(_ => z) {
  override def d: Fun = zero

  override def toContextString(context: String): String = toString

  override def toString: String = z.toString
  override val context = "c"

  override def simplified: Fun = {
    if (z == Cpx(1, 0))
      one
    else if (z == Cpx(0, 0))
      zero
    else
      this
  }

  override def equals(other: Any): Boolean = other match {
    case other: Const =>
      z == other.evalF(0)
    case _ =>
      false
  }
}

object x extends Fun(x => x) {
  def d = one

  def toContextString(context: String): String = toString

  override def toString: String = "x"

  override def simplified: Fun = this

  override val context = "x"
}




object Fun {
  def simplify(f: Fun): Fun = {
    val s: Fun = f.simplified
    var s1: Fun = 0
    while (!s.equals(s1)) {
      s1 = s.simplified
      //println(s, s1, "Done? " + s.equals(s1))
      //println("press F9 to continue...")
      val test = s == s1
    }
    s
  }


  def findAssociativityOption[L <: Fun](f: Fun, g: Fun, binOp: String, leaf: String): Option[(L, L, Fun)] = {

    if (f.context == binOp && g.context == leaf) {
      val fbin = f.asInstanceOf[BiFun]

      if (fbin.g.context == leaf) {
        Some(g.asInstanceOf[L], fbin.g.asInstanceOf[L], fbin.f)
      } else if (fbin.f.context == leaf && g.context == leaf) {
        Some(g.asInstanceOf[L], fbin.f.asInstanceOf[L], fbin.g)
      } else None

    } else if (g.context == binOp && f.context == leaf) {
      val gbin = g.asInstanceOf[BiFun]

      if (gbin.g.context == leaf ) {
        Some(f.asInstanceOf[L], gbin.g.asInstanceOf[L], gbin.f)
      } else if (gbin.f.context == leaf && f.context == leaf) {
        Some(f.asInstanceOf[L], gbin.f.asInstanceOf[L], gbin.g)
      } else None

    } else None
  }


}


