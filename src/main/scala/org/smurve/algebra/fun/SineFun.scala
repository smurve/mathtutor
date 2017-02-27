package org.smurve.algebra.fun

import org.smurve.complex.Cpx

/**
  *
  * the additive inverse -f of a function f
  *
  * @param f the function
  */
class SineFun(val f: Fun) extends Fun(y => { val z=f(y); Cpx(math.sin(z.r)*math.cosh(z.i),math.cos(z.r)*math.sinh(z.i))}) {
  override def d: Fun = f.d * new CosineFun(f)

  // the brackets are for IntelliJ, not for the compiler
  override def toString: String = s"sin(${f.toString})"

  override def toContextString(context: String): String = toString

  override def simplified: Fun = new SineFun(f.simplified)

  override val context = "sin"

  override def equals ( other: Any ): Boolean = other match {
    case o: SineFun => f.equals(o.f)
    case _ => false
  }
}
object sin {
  def apply(f:Fun): SineFun = new SineFun(f)
}


