package org.smurve.algebra.fun

/**
  *
  * the sum of two functions
  *
  * @param f the left function
  * @param g the right function
  */
class SumFun(f: Fun, g: Fun) extends BiFun(f, g, x => f(x) + g(x)) {

  override def d: Fun = f.d + g.d

  override def toString: String = f.toString + "+" + g.toString

  override def toContextString(context: String): String =
    if (List("°", "-", "*", "/").contains(context)) "(" + toString + ")" else toString

  override val context = "+"

  override def simplified: Fun = {
    val fn = f.simplified
    val gn = g.simplified

    val r1 = if (fn == zero && gn == zero) zero else {
      if (fn == zero)
        gn
      else if (gn == zero)
        fn
      else
        fn match {
          case l: Const => gn match {
            case r: Const => return new Const(r.evalF(0) + l.evalF(0))
            case _ => new SumFun(fn, gn)
          }
          case _ => new SumFun(fn, gn)
        }
    }


    /*
      * Check weather any of our two functions can potentially be associatively collapsed: like a * (b * c)
      * returns an Option of a pair of Consts and the ProdFun, if there is such a much
      */
    val localOption: Option[(Const, Const, Fun)] = Fun.findAssociativityOption[Const](fn, gn, "+", "c")

    // if there's an option it will contain the bits for a simplification, otherwise we go with what we have
    localOption.map(lo => new SumFun(f = new Const(lo._1.evalF(0) + lo._2.evalF(0)), g = lo._3)).getOrElse(r1)


  }

  override def equals(other: Any): Boolean = other match {
    case o: SumFun =>
      f.equals(o.f) && g.equals(o.g)
    case _ =>
      false
  }

}

