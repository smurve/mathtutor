package org.smurve.charts

import org.scalatest.{FlatSpec, ShouldMatchers}

/**
  * Created by wgiersche on 04.03.17.
  */
class CornerIteratorTest extends FlatSpec with ShouldMatchers{

  "A corner iterator with both axis in positive direction" should "work" in {

    val pos_pos = CornerIterator(0,2,0,2)
    val c0 = pos_pos.corner(0)
    c0 should be ( List((0,0), (1,0), (2,0), (2,1), (2,2)))
    pos_pos.corner(1) should be ( List((0,1), (1,1), (1,2)))
    pos_pos.corner(2) should be ( List((0,2)))
  }

  "A corner iterator with left axis forward and right axis backwards" should "work" in {

    val pos_pos = CornerIterator(0,2,2,0)
    val c0 = pos_pos.corner(0)
    c0 should be ( List((0,2),(1,2), (2,2), (2,1), (2,0)))
    pos_pos.corner(1) should be ( List((0,1),(1,1),(1,0)))
    pos_pos.corner(2) should be ( List((0,0)))
  }

  "A corner iterator with both axes backward" should "work" in {

    val pos_pos = CornerIterator(3,0,4,0)
    pos_pos.corner(0) should be ( List( (3,4), (2,4), (1,4), (0,4), (0,3), (0,2), (0,1), (0,0)))
    pos_pos.corner(1) should be ( List( (3,3), (2,3), (1,3), (1,2), (1,1), (1,0)))
    pos_pos.corner(2) should be ( List( (3,2), (2,2), (2,1), (2,0)))
    pos_pos.corner(3) should be ( List( (3,1), (3,0)))
  }
}
