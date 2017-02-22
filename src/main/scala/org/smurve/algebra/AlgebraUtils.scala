package org.smurve.algebra

import breeze.linalg.{DenseVector => DV}


/**
  * A Box represents a rectangle that can grow in size as it includes (method <<<()) more points
  * Used to determine the boundary box of large data sets
  * @param lowerLeft the lower left corner expressed by a DenseVector
  * @param upperRight the upper right corner expressed by a DenseVector
  */
case class Box(lowerLeft: DV[Double], upperRight: DV[Double]) {

  /**
    * calculate the box that's just large enough to also include the given vector
    */
  def <<<(newV: DV[Double]): Box = AlgebraUtils.include(this, newV)

  /**
    * @return the area of the box = width * height
    */
  def area : Double = math.abs((lowerLeft.data(0)-upperRight.data(0)) * (lowerLeft.data(1) - upperRight.data(1)))
}


/**
  * Some utilities I use regularly
  */
object AlgebraUtils {

  /**
    * fold samples into a growing box starting with the first sample
    *
    * @param samples the sample set to determine the boundary box of
    * @return the smallest rectangle that contains all the samples given.
    */
  def boundaryBox(samples: Array[DV[Double]]): Box = {
    assert(samples.nonEmpty)
    samples./:(Box(samples(0), samples(0)))(_ <<< _)
  }

  /**
    * @param box  a box made of two DenseVectors representing lower left and upper right corners
    * @param newV a new DenseVector to put into the box
    * @return the smallest box that contains the input box and the new Vector
    */
  def include(box: Box, newV: DV[Double]): Box = {
    val xLowerLeft = if (box.lowerLeft.data(0) < newV.data(0)) box.lowerLeft.data(0) else newV.data(0)
    val yLowerLeft = if (box.lowerLeft.data(1) < newV.data(1)) box.lowerLeft.data(1) else newV.data(1)
    val xUpperRight = if (box.upperRight.data(0) > newV.data(0)) box.upperRight.data(0) else newV.data(0)
    val yUpperRight = if (box.upperRight.data(1) > newV.data(1)) box.upperRight.data(1) else newV.data(1)
    Box(DV[Double](xLowerLeft, yLowerLeft), DV[Double](xUpperRight, yUpperRight))
  } // */

}


