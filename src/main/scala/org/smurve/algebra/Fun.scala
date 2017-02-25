package org.smurve.algebra

import org.smurve.complex.Cpx

/**
  * Created by wgiersche on 25/02/17.
  */
case class Fun ( f: Cpx => Cpx ){

  def apply( x: Cpx ): Cpx = f(x)

  def apply(inner: Fun ) = Fun (x=> f ( inner.f(x) ))

  def + ( other: Fun ) : Fun = Fun ( x => f(x) + other.f(x))
  def * ( other: Fun ) : Fun = Fun ( x => f(x) * other.f(x))
  def - ( other: Fun ) : Fun = Fun ( x => f(x) - other.f(x))
  def / ( other: Fun ) : Fun = Fun ( x => f(x) / other.f(x))
  def ° ( exponent: Integer ) : Fun = Fun ( x => f(x) ^ exponent )
  def unary_- : Fun = Fun ( -_ )
}
