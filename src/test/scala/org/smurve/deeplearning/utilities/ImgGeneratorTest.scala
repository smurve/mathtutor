package org.smurve.deeplearning.utilities

import org.scalatest.{FlatSpec, ShouldMatchers}

/**
  * Created by wgiersche on 17.04.17.
  */
class ImgGeneratorTest extends FlatSpec with ShouldMatchers {

  "An Image Generator" should "produce Vectors representing an image with a symbol just somewhere on it" in {

    val ig = new ImageGenerator(5, 5)
    val img: SingleSymbolImage = ig.singleSymbolImage(ig.DOWN, 2, 2)

    img(2,2) should be (1.0)
    img(3,3) should be (1.0)
    img(4,2) should be (1.0)
  }

  "An Image Generator" should "produce random images " in {

    val ig = new ImageGenerator(5, 5)
    for (_ <- 0 to 10 ) {
      val img: SingleSymbolImage = ig.rndImage(ig.DOWN, ig.UP)
      println ( img.toString )
    }

  }



}
