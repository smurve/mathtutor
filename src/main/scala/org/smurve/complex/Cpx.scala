package org.smurve.complex

import scala.language.implicitConversions

/**
  * Created by wgiersche on 25/02/17.
  */
case class Cpx(r: Double, i: Double) {

  def c: Cpx = Cpx(r, -i)

  // the conjugate
  val abs: Double = math.sqrt(r * r + i * i)
  val abs2: Double = r * r + i * i

  def +(other: Cpx): Cpx = Cpx(r + other.r, i + other.i)

  def *(other: Cpx): Cpx = Cpx(r * other.r - i * other.i, r * other.i + i * other.r)

  def -(other: Cpx): Cpx = Cpx(r - other.r, i - other.i)

  def /(other: Double) = Cpx(r / other, i / other)

  def /(other: Cpx): Cpx = this * other.c / other.abs2

  def ^(e: Int): Cpx = if (e == 0) Cpx(1, 0) else {
    if (e == 1) this else {
      if (e % 2 == 0) {
        val s = this.^(e / 2)
        s * s
      } else this * this.^(e - 1)
    }
  }

  def unary_- : Cpx = Cpx(-r, -i)

  override def toString: String = {

    if (r == 0 && i == 0) "0" else {

      // use multiplier explicitly for non-integers
      val s1 = if (r == 0) "" else if (r == r.toInt) r.toInt.toString else r

      val s2 = if (i == 0) "" else {
        if (i < 0) '-' else {
          if (r == 0) "" else "+"
        }
      }
      val s3 = {
        val (m,v) = if (i == i.toInt) ("",math.abs(i).toInt.toString)  else ("*",math.abs(i).toString)

        if (math.abs(i) == 1) s"${m}i" else {
          if (i != 0)  v + s"${m}i" else ""
        }
      }
      s"$s1$s2$s3"
    }
  }
}
