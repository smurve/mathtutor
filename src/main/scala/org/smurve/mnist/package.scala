package org.smurve

import breeze.linalg._

import scala.language.implicitConversions
import org.smurve.deeplearning._


/**
  * Created by wgiersche on 11.03.17.
  */
package object mnist {

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
