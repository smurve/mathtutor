package org.smurve.algebra.fun

import org.smurve.complex.Cpx

class CosineFun(val f: Fun) extends Fun(y => { val z=f(y); Cpx(math.cos(z.r)*math.cosh(z.i),-math.sin(z.r)*math.sinh(z.i))}) {
  override def d: Fun = f.d * new NegFun(new SineFun(f))

  // the brackets are for IntelliJ, not for the compiler
  override def toString: String = s"cos(${f.toString})"

  override def toContextString(context: String): String = toString

  override def simplified: Fun = new CosineFun(f.simplified)

  override val context = "cos"

  override def equals ( other: Any ): Boolean = other match {
    case o: CosineFun => f.equals(o.f)
    case _ => false
  }
}
object cos {
  def apply(f:Fun): CosineFun = new CosineFun(f)
}

