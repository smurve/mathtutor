package org.smurve.algebra.fun

/**
  * a composite function
 *
  * @param f: the outer function
  * @param g: the inner function
  */
class CompFun(f: Fun, g: Fun) extends BiFun(f, g, x => f(g(x))) {

  override def d: Fun = f.d(g) * g.d

  override def toString: String = f.toString.replaceAll("x", g.toContextString(context))

  override def toContextString(context: String): String = toString

  override def simplified: Fun = this

  override val context: String = f.context

  def equals( other: Fun ) : Boolean = other match {
    case o: CompFun =>
      f.equals(o.f) && g.equals(o.g)
    case _ =>
      false
  }
}