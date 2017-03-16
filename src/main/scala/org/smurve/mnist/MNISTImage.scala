package org.smurve.mnist

import breeze.linalg.DenseVector

/**
  * A single MNIST image
  */
case class MNISTImage(bytes: Array[Byte], width: Int, height: Int ) {

  override def toString: String =     ( 0 to height ).map (i => {
    val row = bytes.slice(i * width, (i+1)*width)
    rowAsString(row)
  }).mkString("\n")


  private def rowAsString ( bytes: Array[Byte]) : String = {
    bytes.map(b=>{
      val n  = b & 0xFF
      val c = if (n == 0) 0 else n / 32 + 1
      c match {
        case 0 => "  "
        case 1 => "' "
        case 2 => "''"
        case 3 => "::"
        case 4 => ";;"
        case 5 => "cc"
        case 6 => "OO"
        case 7 => "00"
        case 8 => "@@"
      }
    }).mkString("")
  }

  /**
    * provide the image data as a DenseVector of values between 0 and 1
    * @return
    */
  def dv = DenseVector(bytes.map (_ & 0xFF).map(_ / 256.0))
}
