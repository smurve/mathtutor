package org.smurve

import breeze.linalg.{DenseMatrix, DenseVector}
import scala.language.implicitConversions

/**
  * Created by wgiersche on 11.03.17.
  */
package object mnist {

  case class Activation ( fn : DenseVector[Double] => DenseVector[Double], deriv: DenseVector[Double] => DenseVector[Double]) {}

  val SIGMOID : Activation = Activation ( sigmoid, sigmoid_prime )

  def sigmoid (v: DenseVector[Double]) : DenseVector[Double] = {
    DenseVector(v.map(xi => 1 / ( 1 + math.exp(-xi))).toArray)
  }

  def sigmoid_prime ( v: DenseVector[Double]) : DenseVector[Double] = {
    val s = sigmoid(v)
    DenseVector(s.map(si => si * ( 1 - si )).toArray)
  }

  def odamard (v1: DenseVector[Double], v2: DenseVector[Double]): DenseVector[Double] = {
    DenseVector(  (v1.toArray zip v2.toArray).map(p => p._1 * p._2))
  }

  type DV = DenseVector[Double]
  type DM = DenseMatrix[Double]

  type InitWith = String
  val INIT_WITH_RANDOM : InitWith = "RANDOM"
  val INIT_WITH_CONST : InitWith = "CONST"

  implicit def toDouble ( n: Int ) : Double = n.toDouble
}
