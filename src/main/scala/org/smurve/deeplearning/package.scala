package org.smurve

import breeze.linalg.{DenseMatrix, DenseVector}

package object deeplearning {

  type DV = DenseVector[Double]
  type DM = DenseMatrix[Double]

  type InitWith = String
  val INIT_WITH_RANDOM: InitWith = "RANDOM"
  val INIT_WITH_CONST: InitWith = "CONST"


  val EUCLIDEAN = CostFunction ( euclideanCost, euclideanCostDerivative)

  // Activation functions and their derivatives
  val SIGMOID: Activation = Activation(sigmoid, sigmoid_prime)
  val IDENTITY: Activation = Activation(x => x, x => DenseVector.ones(x.length))
  val RELU: Activation = Activation( relu, relu_prime )

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



}
