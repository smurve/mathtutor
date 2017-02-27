package org.smurve.algebra

import breeze.linalg.{DenseVector => DV}
import org.scalatest.{FlatSpec, ShouldMatchers}

import scala.util.Random


/**
  * Created by wgiersche on 22/02/17.
  */
class DisplayTest extends FlatSpec with ShouldMatchers {

  "A screen" should "display a function and the given data points" in {

    def rnd() = math.floor(math.random * 7) - 3

    val center = DV(0.0,0, 0)
    val epi1 = center + DV(3.0, 3, 1)
    val epi2 = center + DV(-3.0, -3.0, 0)

    var samples : List[DV[Double]]= (1 to 20).flatMap(_ =>
      List(
        epi1 + DV(rnd(), rnd(), 0),
        epi2 + DV(rnd(), rnd(), 0))
    ).toList

    samples = Random.shuffle(samples)

    val screen: Screen = new Screen(30, 0.1, AlgebraUtils.boundaryBox(samples.toArray))

    val s = screen.frame(samples, 0, 1)

    println(s)
    s.contains(".") should be(true)
    s.contains("x") should be(true)
    s.contains("o") should be(true)
  }



}
