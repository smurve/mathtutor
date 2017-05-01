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

  /**
    * cross entropy with a tolerance range around 0 and 1
    * 0.05 appears to be doing a good job with sigmoid activation
    * @param b
    * @return
    */
  def CROSS_ENTROPY_B(b: Double = 0.05) = CostFunction (crossEntropyCost_b(b), crossEntropyCostDerivative_b(b))



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

  /**
    * this helps to cope with the numerical instability at x_i = 1.0. Note that the sigmoid function
    * returns exactly 1.0 for inputs x >= 37
    */
  private val ALMOST_ONE = 1-1E-16

  /**
    * cross-entropy cost function with a buffer.
    * With activations like e.g. sigmoid(x) the "minimum" would be at x-> \infty,
    * thus the back prop would eventually run flat on the other side of the activation layer
    * @param a: the activation with values in the range (0-1)
    * @param y: the desired vector with values (0,1) and (1,0)
    * @param b: the buffer value. 0.05 does a good job.
    * @return the distance from the tolerance range of y
    */
  def crossEntropyCost_b(b: Double)(a: DV, y: DV): Double =
    crossEntropyCost(a, buffer(y, b))

  def crossEntropyCostDerivative_b(b: Double)(a: DV, y: DV) : DV =
    crossEntropyCostDerivative(a, buffer(y, b) )

  def buffer ( y: DV, b: Double) : DV = y.map(ya=>
       if ( ya == 1.0 ) 1.0 - b else if ( ya == 0.0 ) b else
         throw new IllegalArgumentException("Only 0 and 1!"))


  def crossEntropyCost(a: DV, y: DV): Double = {
    a.toArray.zipWithIndex.foreach(p=>{
      if (p._1 == 1.0 && y(p._2) == 1.0 )
        a(p._2) = ALMOST_ONE
    })
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
