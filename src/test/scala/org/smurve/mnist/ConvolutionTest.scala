package org.smurve.mnist

import breeze.linalg.{DenseMatrix, DenseVector, max, min}
import org.scalatest.{FlatSpec, ShouldMatchers}

/**
  * Created by wgiersche on 19/03/17.
  */
class ConvolutionTest extends FlatSpec with ShouldMatchers {

  "Convolution" should "work like a dream" in {

    val image = new MNISTImageFile("train").img(0)

    println(image)

    val features = DenseMatrix((
      -1.0,-1.0,-1.0,-1.0,
      -1.0,-1.0,-1.0,-1.0,
      +1.0,+1.0,+1.0,+1.0,
      +1.0,+1.0,+1.0,+1.0
    ))

    val fweights = DenseVector(1.0)

    val res = convolute(image.dv, 28, 28, 4,4, features, fweights = fweights)
    val resImage = MNISTImage(toByteArray(res), 25, 25)

    println (resImage)
  }


  "proj()" should "return a subimage of the given image" in {
    val input = Array(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250)
    val img = MNISTImage(input.map(_.toByte), 5, 5)
    val prj = proj(0,3,3,img)
    prj should be (DenseVector(10.0, 20.0, 30.0, 60.0, 70.0, 80.0, 110.0, 120.0, -126.0))
    proj(7,3,3,img) should be (DenseVector(80.0, 90.0, 100.0, -126.0, -116.0, -106.0, -76.0, -66.0, -56.0))
  }


  "toByteArray" should "work" in {
    val res = toByteArray(DenseVector(-4, -2, -1, 4.0))
    res should be ( Array(0, 63, 95, -1))
  }

  def toByteArray ( v: DV ) : Array[Byte] = {
    val maxV = max(v)
    val minV = min(v)
    def scale = 255 / (maxV - minV)
    v.toArray.map(x=>((x - minV) * scale).toByte )
  }
}
