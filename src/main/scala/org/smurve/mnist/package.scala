package org.smurve

import breeze.linalg._

import scala.language.implicitConversions


/**
  * Created by wgiersche on 11.03.17.
  */
package object mnist {

  case class Activation
  (
    fn: DenseVector[Double] => DenseVector[Double],
    deriv: DenseVector[Double] => DenseVector[Double]) {
  }

  case class CostFunction
  (
    fn: (DV, DV) => Double,
    deriv: ( DV, DV ) => DV
  )

  val EUCLIDEAN = CostFunction ( euclideanCost, euclideanCostDerivative)

  val SIGMOID: Activation = Activation(sigmoid, sigmoid_prime)

  val UNIT: Activation = Activation(x => x, x => DenseVector.ones(x.length))

  def sigmoid(v: DenseVector[Double]): DenseVector[Double] = {
    DenseVector(v.map(xi => 1 / (1 + math.exp(-xi))).toArray)
  }

  def sigmoid_prime(v: DenseVector[Double]): DenseVector[Double] = {
    val s = sigmoid(v)
    DenseVector(s.map(si => si * (1 - si)).toArray)
  }

  def euclideanCost(finalActivation: DV, desired: DV): Double = {
    val diff = finalActivation - desired
    (.5 * diff.t * diff).toArray.apply(0)
  }

  def euclideanCostDerivative(finalActivation: DV, desired: DV): DV = finalActivation - desired


  type DV = DenseVector[Double]
  type DM = DenseMatrix[Double]

  type InitWith = String
  val INIT_WITH_RANDOM: InitWith = "RANDOM"
  val INIT_WITH_CONST: InitWith = "CONST"

  implicit def toDouble(n: Int): Double = n.toDouble

  /**
    * Use parallel array to improve CPU usage with matrix multiplications
    */
  def parMul(m: DM, v: DV): DV = {
    var a = Array[DV]()
    m(*, ::).foreach(r => a = a :+ r)
    DenseVector(a.par.map(x => x.t * v).toArray)
  }


  def proj(start: Int, wWidth: Int, wHeight: Int, input: DV, inputWidth: Int): DV = {
    val seq = (0 until wHeight).flatMap(y => (0 until wWidth).map({
      x => input.toArray.apply(start + x + y * inputWidth)
    }))


    DenseVector(seq.toArray)
  }

  /**
    * determine the number with the max confidence
    *
    * @param y the result vector of the network
    * @return the index with the largest value
    */
  def asNumber(y: DV): Int = {
    y.data.zipWithIndex.fold((0.0, 1))((l, r) => if (l._1 < r._1) r else l)._2
  }


}
