package org.smurve.algebra.fun

/**
  *
  * @param f the base function
  * @param n the exponent
  */
class PowFun(val f: Fun, val n: Int) extends Fun(x => f(x) ^ n) {

  // the brackets are for IntelliJ, not for the compiler
  override def d: Fun = (n * f ° (n - 1)) * f.d


  override def toString: String = f.toContextString("°") + "^" + n

  override val context = "°"

  override def toContextString(context: String): String =
    if (context == "°") "(" + toString + ")" else toString

  override def simplified: Fun = {
    if ( n == 0 ) one
    else {
      if ( n == 1 ) f.simplified
      else
        new PowFun(f.simplified, n)
    }
  }

  override def equals ( other: Any ) : Boolean = other match {
    case e: PowFun => e.n == n && e.f == f
    case _=> false
  }
}

