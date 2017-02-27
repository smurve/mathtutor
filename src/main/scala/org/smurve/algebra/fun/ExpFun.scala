package org.smurve.algebra.fun

import org.smurve.complex.Cpx

/**
  *
  * the additive inverse -f of a function f
  *
  * @param f the function
  */
class ExpFun(val f: Fun) extends Fun(y => { val z=f(y); math.exp(z.r) * Cpx( math.cos(z.i), math.sin(z.i))}) {

  override def d: Fun = f.d * this

  // the brackets are for IntelliJ, not for the compiler
  override def toString: String = s"exp(${f.toString})"

  override def toContextString(context: String): String = toString

  override def simplified: Fun = new ExpFun(f.simplified)

  override val context = "exp"

  override def equals ( other: Any ): Boolean = other match {
    case o: ExpFun => f.equals(o.f)
    case _ => false
  }
}

object exp {
  def apply(f:Fun): ExpFun = new ExpFun(f)
}


