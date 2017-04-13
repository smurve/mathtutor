package org.smurve.mnist
import scala.collection.immutable.Seq


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

  def shearHorizontal(orig: MNISTImage): MNISTImage = {
    assert ( orig.width == 28 )
    val array = orig.bytes
    val width = orig.width
    var newArray = Array[Byte]()
    for ( block <- 0 until 4 ) {
      for ( row <- 0 until 7 ) {
        val startAt = ( block * 7 + row ) * width
        val offset = 3 - block * 2
        val padding : Array[Byte] = Array.fill[Byte](math.abs(offset))(0.toByte)
        val res =
          if ( offset > 0) {
            val sheared = array.slice(startAt, startAt + 28 - offset)
            padding ++ sheared
          } else {
            val sheared = array.slice(startAt - offset, startAt + 28)
            sheared ++ padding
          }

        newArray = newArray ++ res
      }
    }
    MNISTImage(newArray, 28, 28)
  }

  def pos ( byte : Byte ) : Int = (byte + 256 ) % 256

  def sharpen ( orig: MNISTImage, threshold: Int ) : MNISTImage =
    MNISTImage ( orig.bytes.map(b => if ( pos(b) > threshold ) 255.toByte else 0.toByte ), orig.width, orig.height)

  def squeeze ( orig: MNISTImage ) : MNISTImage = {
    val newBytes =
    ( 0 until 28 ).filter(!Seq(10,20).contains(_)).flatMap(
      row => orig.bytes.slice ( 28 * row, 28 * (row + 1))).
      toArray ++ Array.fill[Byte](56)(0.toByte)



      //zipWithIndex.filter(p=>Seq(10, 20).contains(p._2)).flatMap(p=>p._1)

    MNISTImage(newBytes, 28, 28)
  }


}
