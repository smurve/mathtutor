package org.smurve

import org.smurve.algebra.fun.{Const, Fun}
import org.smurve.complex.Cpx
import breeze.linalg.DenseVector

import scala.language.implicitConversions

package object algebra {

  implicit def toFun(x: Double): Fun = new Const( Cpx(x, 0))
  implicit def toFun(n: Int): Fun = new Const( Cpx(n, 0))
  implicit def toFun(z: Cpx): Fun = new Const( z )
  implicit def toDouble(n: Int): Double = n.toDouble
  implicit def toDVDouble(dvi: DenseVector[Int]): DenseVector[Double] = new DenseVector(dvi.data.map(_.toDouble))

  class Composable ( f: DenseVector[Double] => DenseVector[Double] ) {
    def apply (x:DenseVector[Double]): DenseVector[Double] = f(x)
    def ° ( other: Composable ): (DenseVector[Double]) => DenseVector[Double] = (x: DenseVector[Double]) => this.apply(other.apply(x))
    def || ( other: Composable ): (DenseVector[Double]) => DenseVector[Double] = other°this
  }



  implicit def toComposable(f: (DenseVector[Double]) => DenseVector[Double]): Composable = new Composable(f)
}
