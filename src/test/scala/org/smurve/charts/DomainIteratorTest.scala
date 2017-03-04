package org.smurve.charts

import org.scalatest.{FlatSpec, ShouldMatchers}

/**
  *
  *
  *            |
  *       3    |    2
  *            |
  *   _________|________
  *            |
  *            |
  *       4    |    1
  *            |
  *
  **
  *
  *
  */
class DomainIteratorTest extends FlatSpec with ShouldMatchers {

  val xAxis = DomainAxis(-3, 3, 7)
  val yAxis = DomainAxis(-4, 2, 7)

  "An iterator for case 1" should "produce elements x then y" in {

    /*

            (-3,2)  -  (3,2)

            (-3,-4) -  (3,-4)


                                observer
     */

    // Case 1: Observer in lower right quarter, theta doesn't matter
    val proj = new PlaneProjector(-45, 45)
    val xpx = proj.normalVectors._1.data(0)
    val xpy = proj.normalVectors._1.data(1)

    val iterator = new DomainIterator(xpx, xpy, xAxis, yAxis)

    val c0 = iterator.corner(0)
    c0.length should be ( 13)
    c0.head should be ( (-3, -4))
    c0(7) should be ( (3, -3))
    c0(12) should be ((3,2))

    val c1 = iterator.corner(1)
    c1.length should be ( 11)
    c1.head should be ( (-3, -3))
    c1(5) should be ( (2, -3))
    c1(10) should be ((2,2))

    val c6 = iterator.corner(6)
    c6.length should be ( 1 )
    c6.head should be ( (-3, 2))

    val c7 = iterator.corner(7)
    c7 should be ( Nil )
  }




  "An iterator for case 2" should "produce y then backwards x" in {

    /*
                                observer

            (-3,2)  -  (3,2)

            (-3,-4) -  (3,-4)
     */


    // Case 1: Observer in lower right quarter, theta doesn't matter
    val prj = new PlaneProjector(45, 45)
    val xpx = prj.normalVectors._1.data(0)
    val xpy = prj.normalVectors._1.data(1)

    val iterator = new DomainIterator(xpx, xpy, xAxis, yAxis)

    val c0 = iterator.corner(0)
    c0.length should be ( 13)
    c0.head should be ( (3, -4))
    c0(12) should be ((-3,2))

    val c1 = iterator.corner(1)
    c1.length should be ( 11)
    c1.head should be ( (2, -4))
    c1(10) should be ( (-3, 1))

    val c6 = iterator.corner(6)
    c6.length should be ( 1 )
    c6.head should be ( (-3, -4))
  }

  "An iterator for case 3" should "produce x and y backwards" in {

    /*
        observer


                (-3,2)  -  (3,2)

                (-3,-4) -  (3,-4)
     */


    // Case 1: Observer in lower right quarter, theta doesn't matter
    val prj = new PlaneProjector(135, 45)
    val xpx = prj.normalVectors._1.data(0)
    val xpy = prj.normalVectors._1.data(1)

    val iterator = new DomainIterator(xpx, xpy, xAxis, yAxis)

    val c0 = iterator.corner(0)
    c0.length should be ( 13)
    c0.head should be ( (3, 2))
    c0(12) should be ((-3,-4))

    val c1 = iterator.corner(1)
    c1.length should be (11)
    c1.head should be ((3,1))
    c1(10) should be ((-2, -4))

    val c6 = iterator.corner(6)
    c6.length should be ( 1 )
    c6.head should be ( (3, -4))
  }




  "An iterator for case 4" should "produce backwards y and then x" in {

    /*
                (-3,2)  -  (3,2)

                (-3,-4) -  (3,-4)


        observer
     */


    // Case 1: Observer in lower right quarter, theta doesn't matter
    val prj = new PlaneProjector( -135, 45)
    val xpx = prj.normalVectors._1.data(0)
    val xpy = prj.normalVectors._1.data(1)

    val iterator = new DomainIterator(xpx, xpy, xAxis, yAxis)

    val c0 = iterator.corner(0)
    c0.length should be ( 13)
    c0.head should be ( (-3, 2))
    c0(12) should be ((3,-4))

    val c1 = iterator.corner(1)
    c1.length should be (11)
    c1.head should be ((-2,2))
    c1(10) should be ((3, -3))

    val c6 = iterator.corner(6)
    c6.length should be ( 1 )
    c6.head should be ( (3, 2))
  }

}
