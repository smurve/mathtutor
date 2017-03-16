package org.smurve.mnist

import breeze.linalg.DenseVector


/**
  * A file containing MNIST Labels
  * see: http://yann.lecun.com/exdb/mnist/
  *
  * @param fileName: the name of the file
  */
class MNISTLabelFile(fileName: String ) extends MNISTFile ( fileName ) {

  val headerSize = 8

  assert(magicNumber == LABELS )

  val numLabels: Int = asInt(bytes.slice(4,8))

  def lv ( index: Int ) : Int = bytes(headerSize+index)

  def lblForVal ( lval: Int ) : DV = {
    DenseVector((0 to 9).map( i=> if (lval == i) 1.0 else 0.0).toArray)
  }

  def lblAtPos( index: Int ) : DV = {
    lblForVal(lv(index))
  }
}

