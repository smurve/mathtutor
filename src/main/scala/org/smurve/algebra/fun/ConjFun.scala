package org.smurve.algebra.fun

import org.smurve.complex.Cpx

/**
  * the conjugate x - iy
  * @param f the inner function
  */
class ConjFun(val f: Fun) extends Fun(x => Cpx(x.r, -x.i)) {

  // the brackets are for IntelliJ, not for the compiler
  override def d: Fun = throw new NoSuchMethodError("Can't differentiate conjugate")

  override def toString: String = "conj(" + f.toString + ")"

  override val context = "conj"

  override def toContextString(context: String): String = toString

  override def simplified: Fun = this

  override def equals ( other: Any ) : Boolean = other match {
    case e: ConjFun => e.f == f
    case _=> false
  }
}

object conj {
  def apply( f: Fun ) : Fun = new ConjFun(f)
}