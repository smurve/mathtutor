package org.smurve.algebra.fun

/**
  *
  * the algebraic difference of two functions
  *
  * @param f the left function
  * @param g the right function
  */
class DiffFun(f: Fun, g: Fun) extends BiFun(f, g, x => f(x) - g(x)) {
  override def d: Fun = f.d - g.d

  override def toString: String = f.toString + "-" + g.toContextString("-")

  override def toContextString(context: String): String =
    if (List("°", "-", "*", "/").contains(context)) "(" + toString + ")" else toString

  override val context = "-"

  override def simplified: Fun = {
    val fn = f.simplified
    val gn = g.simplified

    if (fn == zero && gn == zero)
      zero
    else if (fn == zero)
      -gn.simplified
    else if (gn == zero)
      fn.simplified
    else if (fn == gn)
      zero
    else
      new DiffFun(fn.simplified, gn.simplified)
  }

  override def equals( other: Any ) : Boolean = other match {
    case m: DiffFun => m.f == f
    case _=> false
  }
}

