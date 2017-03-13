package org.smurve.mnist

import java.nio.file.{Files, Paths}

import breeze.linalg.DenseVector


/**
  * Created by wgiersche on 11.03.17.
  */
object MNISTHelper {


  def read ( fileName: String ) : Array[Byte] = {
    Files.readAllBytes(Paths.get(fileName))
  }

  def readTraining ( samples: String, labelSuffix: String = "-labels") : Array[TrainingVector] = {

    val allImages = Files.readAllBytes(Paths.get(samples))
    val labels = Files.readAllBytes(Paths.get(samples+labelSuffix))
    val size = 65536 * (0xFF & allImages(5)) + 256 * ( 0xFF & allImages(6)) + ( 0xFF & allImages(7))

    (0 until size).map(index => {
      val image = allImages.slice(16 + index * 784, 16+ ( index + 1 ) * 784 )
      TrainingVector(image, labels(index+8) & 0xFF)
    }).toArray
  }


  def disp ( bytes: Array[Byte], height: Int = 28, width: Int = 28 ): String = {

    ( 0 to height ).map (i => {
      val row = bytes.slice(i * width, (i+1)*width)
      asString(row)
    }).mkString("\n")
  }

  def byteAvg ( l: Byte, r: Byte ) : Byte = {
    (((l & 0xFF) + (r & 0xFF)) / 2).toByte
  }

  def disp2 ( bytes: Array[Byte], height: Int = 28, width: Int = 28 ): String = {

    ( 0 until height/2 ).map (i => {
      val row1 = bytes.slice(2*i * width, (2*i+1)*width)
      val row2 = bytes.slice((2*i+1) * width, (2*i+2)*width)
      val row = (row1 zip row2).map(p=>byteAvg(p._1, p._2) )
      val row3 = (0 until width / 2 ).map(i=> byteAvg(row(2*i), row(2*i+1))).toArray
      asString(row3)
    }).mkString("\n")
  }

  def img ( bytes: Array[Byte], index: Int ) : String  = {
    disp ( bytes.slice(16 + index * 784, 16+ ( index + 1 ) * 784 ))
  }

  def img2 ( bytes: Array[Byte], index: Int ) : String  = {
    disp2 ( bytes.slice(16 + index * 784, 16+ ( index + 1 ) * 784 ))
  }

  //".:ox$"
  def asString ( bytes: Array[Byte]) : String = {
    bytes.map(b=>{
      val n  = (b & 0xFF) / 64
      n match {
        case 0 => " "
        case 1 => "."
        case 2 => "o"
        case 3 => "$"
      }
    }).mkString("")
  }


  def main(args: Array[String]): Unit = {
    val b = read("train")
    val image = img(b, 1)
    println(image)
  }

}
