package org.smurve.algebra.fun

/**
  *
  * @param f the numerator function
  * @param g the denominator function
  */
class FracFun(f: Fun, g: Fun) extends BiFun(f, g, x => f(x) / g(x)) {
  override def d: Fun = f.d / g - f * g.d / g / g

  override def toString: String = f.toContextString("/") + "/" + g.toContextString("/")

  override def toContextString(context: String): String =
    if (context == "°") "(" + toString + ")" else toString

  override val context = "/"

  override def simplified: Fun = this

  override def equals( other: Any ) : Boolean = other match {
    case o: FracFun =>
      f.equals(o.f) && g.equals(o.g)
    case _ =>
      false
  }

}

