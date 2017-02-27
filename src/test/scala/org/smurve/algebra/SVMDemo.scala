package org.smurve.algebra

import breeze.linalg.{DenseVector => DV}
import org.specs2.Specification
import org.specs2.matcher.MatchResult
import org.specs2.specification.core.SpecStructure

import scala.util.Random

/**
  * Created by wgiersche on 22/02/17.
  */
class SVMDemo extends Specification {
  override def is: SpecStructure =
        s2"""
        (dollar) training
    """


  private def training: MatchResult[String] = {

    val gamma = 0.2
    var screen: Option[Screen] = None

    val svm = new SVM( gamma )

    var samples: List[DV[Double]] = Nil
    createSamples.foreach{s=>
      samples = s :: samples

      if ( samples.size > 1 && AlgebraUtils.boundaryBox(samples.toArray).area > 0) {
        if ( screen.isEmpty ) {
          screen = Some(new Screen(30, 0.1, AlgebraUtils.boundaryBox(samples.toArray)))
        }
        screen.get.display(plane(svm.weights, s))
        println(screen.get.frame(samples, 0, 1))
        println ( s"Weights:     ${svm.weights.toString()}")
        val p = svm.learn(s)
        println ( s"Sample: ${s.toString} - predicted: $p")
        println ( s"New weights: ${svm.weights.toString()}")
        println ()
        println ()
      }
    }

    "Training" must equalTo ("Training")
  }

  private def plane ( w: DV[Double], s: DV[Double]): (Double)=>Double = {
    x => w.data(0) / w.data(2) - w.data(1)/w.data(2) * x
  }


  private def createSamples = {

    def dx = math.floor(math.random * 7) - 3

    val center = DV(0.0,0, 0)
    val epi1 = center + DV(3.0, 3, 1)
    val epi2 = center + DV(-3.0, -3.0, 0)
    var samples : List[DV[Double]]=(1 to 20).flatMap(_ => List(epi1 + DV(dx, dx, 0), epi2 + DV(dx, dx, 0))).toList
    Random.shuffle(samples)
  }
}

class SVM ( gamma: Double ) {

  var weights: DV[Double] = DV[Double](math.random -0.5, math.random - 0.5, math.random -0.5 )

  def f_a ( s: DV[Double]) : Double = {
    val p: Double = weights.t * s
    if ( p > 0 ) 1.0 else 0.0
  }

  def learn ( sample: DV[Double]) : Double = {

    val d = sample.data(2)

    val copy = DV[Double](-1.0, sample.data(0), sample.data(1))

    val prediction = f_a(copy)

    weights = weights + gamma * (d - prediction) * copy

    prediction
  }

}
