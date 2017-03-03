package org.smurve.charts

import org.scalatest.{FlatSpec, ShouldMatchers}

/**
  * Created by wgiersche on 03.03.17.
  */
class ProjectorTest extends FlatSpec with ShouldMatchers{

  private val sin45 = 70711
  private val prj = new PlaneProjector(45,0)

  "the projected coordinates for 45, 0" should "be correct" in {
    val cs = prj.normalVectors
    first_five(cs._1.data(0)) should be (-sin45)
    first_five(cs._1.data(1)) should be (sin45)
    cs._1.data(2) should be ( 0 )

    cs._2.data(0) should be ( 0 )
    cs._2.data(1) should be ( 0 )
    cs._2.data(2) should be ( 1 )
  }


  // first five digits
  private def first_five ( d: Double ) : Double = math.round(100000 * d)
}
