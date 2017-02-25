package org.smurve.algebra.fun

/**
  *
  * the additive inverse -f of a function f
  *
  * @param f the function
  */
class NegFun(val f: Fun) extends Fun(x => -f(x)) {
  override def d: Fun = -f.d

  // the brackets are for IntelliJ, not for the compiler
  override def toString: String = "-" + f.toContextString("-")

  override def toContextString(context: String): String = "(" + toString + ")"

  override def simplified: Fun = this

  override val context = "-"

  override def equals ( other: Any ): Boolean = other match {
    case o: NegFun => f.equals(o.f)
    case _ => false
  }
}

