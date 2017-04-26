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

  val a_tau: Activation = Activation ("tau", tau, tau_prime)

  // Activation functions and their derivatives
  val a_sigmoid: Activation = Activation("sigmoid", sigmoid, sigmoid_prime)
  def a_scale(scale: Double): Activation = Activation("scale",
    x => x * scale , x => DenseVector.fill(x.length){scale})
  val a_relu: Activation = Activation( "ReLU", relu, relu_prime )


  def sigmoid(v: DV): DV = {
    DenseVector(v.map(xi => 1 / (1 + math.exp(-xi))).toArray)
  }

  def sigmoid_prime(v: DV): DV = {
    val s = sigmoid(v)
    DenseVector(s.map(si => si * (1 - si)).toArray)
  }

  /**
    * Wolfie's tau function
    * @param x the incoming value
    * @return sigmoid for negative values and linear continuation for positive values
    */
  def tau(x: DV): DV = {
    x.map(xi => if (xi < 0) 1 / (1 + math.exp(-xi)) else .5 + .25 * xi )
  }

  def tau_prime(x: DV): DV = {
    x.map(xi => if ( xi < 0 ){
      val s = breeze.numerics.sigmoid(xi)
      s * (1 - s)
    } else
      0.25
    )
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
