package org.smurve.algebra

import org.smurve.complex.Cpx

abstract class BiFun(f: Fun, g: Fun, evalF: Cpx => Cpx) extends Fun(evalF) {
}

class AddFun(f: Fun, g: Fun) extends BiFun(f, g, x=>f(x) + g(x)) {
  override def d: Fun = f.d + g.d

  override def toString: String = f.toString+"+"+g.toString
  override def toContextString(context: String): String =
    if (List("°", "-", "*", "/").contains(context) ) "("+toString+")" else toString
}

class MinFun(f: Fun, g: Fun) extends BiFun(f, g, x=>f(x) - g(x)) {
  override def d: Fun = f.d - g.d
  override def toString: String = f.toString + "-" + g.toContextString ("-")
  override def toContextString(context: String): String =
    if (List("°", "-", "*", "/").contains(context) ) "("+toString + ")" else toString
}

class MulFun(f: Fun, g: Fun) extends BiFun(f, g, x=>f(x) * g(x)) {
  override def d: Fun = f.d * g + f * g.d
  override def toString: String = f.toContextString("*") + "*" + g.toContextString("*")
  override def toContextString(context: String): String =
    if (context == "°" ) "(" + toString + ")" else toString
}

class DivFun(f: Fun, g: Fun) extends BiFun(f, g, x=>f(x) / g(x)) {
  override def d: Fun = f.d / g - f * g.d / g / g
  override def toString: String = f.toContextString("/") + "/" + g.toContextString("/")
  override def toContextString(context: String): String =
    if (context == "°" ) "("+toString + ")" else toString
}

class ExpFun(f: Fun, n: Int) extends Fun (x=>f(x) ^ n) {

  // the brackets are for IntelliJ, not for the compiler
  override def d: Fun = (n * f°(n-1)) * f.d


  override def toString: String = f.toContextString("°") +"°" + n

  override def toContextString(context: String): String =
    if (context == "°" ) "("+toString + ")" else toString
}

class NegFun(f: Fun) extends Fun (x => -f(x)) {
  override def d: Fun = -f.d

  // the brackets are for IntelliJ, not for the compiler
  override def toString: String = "-"+f.toContextString("-")

  override def toContextString(context: String): String = "("+toString + ")"
}

class InFun (f: Fun, g: Fun) extends BiFun (f, g, x=> f(g(x))) {
  override def d: Fun = f.d(g)*g.d

  override def toString: String = f.toString.replaceAll("x", g.toString)
  override def toContextString(context: String): String = toString
}