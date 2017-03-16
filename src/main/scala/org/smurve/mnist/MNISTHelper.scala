package org.smurve.mnist


/**
  * Created by wgiersche on 11.03.17.
  */
object MNISTHelper {

  /**
    * averages over each set of 4 neighboured pixels (bytes)
    *
    * @param bytes  the original image
    * @param width  the width of the original image
    * @param height the height of the original image
    * @return a shrunk image of size = width / 2 * height / 2
    */
  def shrink(bytes: Array[Byte], width: Int, height: Int): Array[Byte] = {
    (0 until height / 2).flatMap(i => {
      val row1 = bytes.slice(2 * i * width, (2 * i + 1) * width)
      val row2 = bytes.slice((2 * i + 1) * width, (2 * i + 2) * width)
      val row = (row1 zip row2).map(p => (p._1 & 0xFF) + (p._2 & 0xFF))
      (0 until width / 2).map(i => ((row(2 * i) + row(2 * i + 1)) / 4).toByte)
    }).toArray
  }

  def byteSum(l: Byte, r: Byte): Byte = {
    ((l & 0xFF) + (r & 0xFF)).toByte
  }


  def shrink(orig: MNISTImage): MNISTImage = {
    MNISTImage(shrink(orig.bytes, orig.width, orig.height), orig.width / 2, orig.height / 2)
  }


  def main(args: Array[String]): Unit = {

    val file = new MNISTImageFile("train")

    val image = file.img(0)

    println(image)
    println(shrink(image))
  }

}
