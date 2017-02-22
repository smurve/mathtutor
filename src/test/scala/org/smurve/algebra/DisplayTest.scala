package org.smurve.algebra

import breeze.linalg.{ DenseVector => DV }
import org.specs2.Specification
import org.specs2.specification.core.SpecStructure

import scala.util.Random


/**
  * Created by wgiersche on 22/02/17.
  */
class DisplayTest extends Specification {
  override def is: SpecStructure =
    s2"""
        plotting some values works $createSamples
      """

  private def createSamples = {

    def dx = math.floor(math.random * 7) - 3

    val center = DV(0.0,0, 0)
    val epi1 = center + DV(3.0, 3, 1)
    val epi2 = center + DV(-3.0, -3.0, 0)
    var samples : List[DV[Double]]= (1 to 20).flatMap(_ => List(epi1 + DV(dx, dx, 0), epi2 + DV(dx, dx, 0))).toList
    samples = Random.shuffle(samples)

    val screen: Screen = new Screen(30, 0.1, AlgebraUtils.boundaryBox(samples.toArray))

    println(screen.frame(samples, 0, 1))
    "Truth" must equalTo("Truth")
  }



}
