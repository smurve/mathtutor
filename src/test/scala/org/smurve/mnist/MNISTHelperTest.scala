package org.smurve.mnist

import breeze.linalg.DenseVector
import org.scalatest.{FlatSpec, ShouldMatchers}

class MNISTHelperTest extends FlatSpec with ShouldMatchers {

  val h = MNISTHelper

  def b(orig: Int) = orig.toByte


  "shrink()" should "shrink a byte array correctly" in {

    val shrunk = h.shrink( Array(
      4,6,5,7,9,11,
      5,1,4,4,3,5,
      5,7,4,6,2,6,
      7,3,1,3,5,3

    ).map(_.toByte), 6, 4)

    shrunk should be (Array(4, 5, 7, 5, 3, 4).map(_.toByte))

    h.shrink( Array(
      240,240,240,240
    ).map(_.toByte), 2, 2)(0) should be ( b(240) )

    h.shrink( Array(
      70,70,70,70
    ).map(_.toByte), 2, 2)(0) should be ( b(70) )

    h.shrink( Array(
      130,130,130,130
    ).map(_.toByte), 2, 2)(0) should be ( b(130) )

    h.shrink( Array(
      129,127,65,63
    ).map(_.toByte), 2, 2)(0) should be ( b(96) )

  }

  "Shearing" should "change the image a little" in {

    val file = new MNISTImageFile("train")


    val image = file.img(2)

    val sheared = h.shearHorizontal(image)

    println(image)
    println(sheared)

  }

  "Sharpening" should "sharpen the image" in {

    val file = new MNISTImageFile("train")

    val image = file.img(2)

    val sharpened = h.sharpen(image, 127)

    println(image)
    println(sharpened)

  }

  "Squeezing" should "squeeze the image" in {

    val file = new MNISTImageFile("train")

    val image = file.img(2)

    val squeezed = h.squeeze(image)

    println(image)
    println(squeezed)

  }

  "An image file" should "be created with its filename" in {

    val file = new MNISTImageFile("train")

    file.magicNumber should be ( 2051 )
    file.numImgs should be ( 60000 )
    file.numRows should be ( 28 )
    file.numCols should be ( 28 )
    file.headerSize should be ( 16 )
  }

  "A label file" should "provide Vectors as labels" in {

    val file = new MNISTLabelFile("train-labels")

    file.lblAtPos(0) should be ( DenseVector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
  }


}
