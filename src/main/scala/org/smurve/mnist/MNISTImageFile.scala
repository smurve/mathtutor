package org.smurve.mnist

import scala.collection.immutable.IndexedSeq


/**
  * A file containing MNIST Images
  * see: http://yann.lecun.com/exdb/mnist/
  * @param fileName: the name of the file
  */
class MNISTImageFile ( fileName: String ) extends MNISTFile ( fileName ) {

  val headerSize = 16

  assert(magicNumber == IMAGES )

  val numImgs: Int = asInt(bytes.slice(4,8))
  val numRows: Int = asInt(bytes.slice(8,12))
  val numCols: Int = asInt(bytes.slice(12,16))

  val imgSize: Int = numRows * numCols

  def img ( index: Int ) : MNISTImage =
    MNISTImage(bytes.slice( headerSize + imgSize * index, headerSize + imgSize * ( index + 1 )), numCols, numRows)

  def imgs: IndexedSeq[MNISTImage] = ( 0 until numImgs).map(img)
}

