package org.smurve

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.log

package object deeplearning {

  type DV = DenseVector[Double]
  type DM = DenseMatrix[Double]

  type InitWith = String
  val INIT_WITH_RANDOM: InitWith = "RANDOM"
  val INIT_WITH_CONST: InitWith = "CONST"


  val EUCLIDEAN = CostFunction ( euclideanCost, euclideanCostDerivative)
  val CROSS_ENTROPY = CostFunction (crossEntropyCost, crossEntropyCostDerivative)

  // Activation functions and their derivatives
  val a_sigmoid: Activation = Activation("sigmoid", sigmoid, sigmoid_prime)
  def a_scale(scale: Double): Activation = Activation("scale",
    x => x * scale , x => DenseVector.fill(x.length){scale})
  val a_relu: Activation = Activation( "ReLU", relu, relu_prime )

  def sigmoid(v: DenseVector[Double]): DenseVector[Double] = {
    DenseVector(v.map(xi => 1 / (1 + math.exp(-xi))).toArray)
  }

  def sigmoid_prime(v: DenseVector[Double]): DenseVector[Double] = {
    val s = sigmoid(v)
    DenseVector(s.map(si => si * (1 - si)).toArray)
  }

  def relu ( v: DV ) : DV = v.map(_.max(0))

  def relu_prime ( v: DV ) : DV = v.map(x=>if(x<=0.0) 0.0 else 1.0 )

  def euclideanCost(finalActivation: DV, desired: DV): Double = {
    val diff = finalActivation - desired
    (.5 * diff.t * diff).toArray.apply(0)
  }

  def euclideanCostDerivative(finalActivation: DV, desired: DV): DV = finalActivation - desired

  def crossEntropyCost(a: DV, y: DV): Double = {
    val ones = DenseVector.ones[Double](y.length)
    val res = -(y.t * log(a) + ( ones - y ).t * log(ones - a))
    res
  }

  def crossEntropyCostDerivative(a: DV, y: DV) : DV = {
    val ones = DenseVector.ones[Double](y.length)
    val res = ( a - y ) :/ ( a :* ( ones - a ))
    res
  }

}
