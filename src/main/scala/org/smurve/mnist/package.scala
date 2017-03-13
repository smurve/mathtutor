package org.smurve

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by wgiersche on 11.03.17.
  */
package object mnist {

  def sigmoid ( v: DenseVector[Double]) : DenseVector[Double] = {
    DenseVector(v.map(xi => 1 / ( 1 + math.exp(-xi))).toArray)
  }

  def sigmoid_prime ( v: DenseVector[Double]) : DenseVector[Double] = {
    val s = sigmoid(v)
    DenseVector(s.map(si => si * ( 1 - si )).toArray)
  }

  def adamard (v1: DenseVector[Double], v2: DenseVector[Double]) = {
    DenseVector(  (v1.toArray zip v2.toArray).map(p => p._1 * p._2))
  }

  type Activation = DenseVector[Double] => DenseVector[Double]

  type DV = DenseVector[Double]
  type DM = DenseMatrix[Double]

  type InitWith = String
  val INIT_WITH_RANDOM : InitWith = "RANDOM"
  val INIT_WITH_CONST : InitWith = "CONST"
}
