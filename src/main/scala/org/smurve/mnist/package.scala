package org.smurve

import breeze.linalg._

import scala.collection.immutable.Seq
import scala.language.implicitConversions


/**
  * Created by wgiersche on 11.03.17.
  */
package object mnist {

  case class Activation(fn: DenseVector[Double] => DenseVector[Double], deriv: DenseVector[Double] => DenseVector[Double]) {}

  val SIGMOID: Activation = Activation(sigmoid, sigmoid_prime)

  def sigmoid(v: DenseVector[Double]): DenseVector[Double] = {
    DenseVector(v.map(xi => 1 / (1 + math.exp(-xi))).toArray)
  }

  def sigmoid_prime(v: DenseVector[Double]): DenseVector[Double] = {
    val s = sigmoid(v)
    DenseVector(s.map(si => si * (1 - si)).toArray)
  }


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

  /*
  def convolute(input: DV, imgWidth: Int, imgHeight: Int, winWidth: Int, winHeight: Int, featureMatrix: DM, fweights: DV ): DV = {

    val inputXRange = 0 until imgWidth - winWidth + 1
    val inputYRange = 0 until imgHeight - winHeight + 1
    val inputArray = input.toArray

    // seq is the sequence of index pairs mapping from the input image to the convoluted image
    val seq = for {
      y <- inputYRange
      x <- inputXRange
    } yield (x + imgWidth * y, x + inputXRange * y)


    val resArray = seq.map(p => {
      val i_inp = p._1
      val i_out = p._2
      val features = featureMatrix * proj(i_inp, winWidth, winHeight, input, imgWidth)
      fweights.t * features
    }).toArray

    DenseVector(resArray)

  }*/



  def proj(start: Int, wWidth: Int, wHeight: Int, input: DV, inputWidth: Int ) : DV = {
    val seq = (0 until wHeight).flatMap(y=> (0 until wWidth).map({
      x=>input.toArray.apply(start + x+y*inputWidth)
    }))


    DenseVector(seq.toArray)
  }
}
