package org.smurve.charts

/**
  * Iterates the two-dimensional domain, spun by the given axis in a corner fashion.
  * The coordinates xpx, xpy represent the projection normal vector in projected x direction.
  * In the picture below, the numbers indicate the order of the resulting list of domain points
  * In this approach a point x is shadowed if any of the previously calculated values had the
  * same projected x and higher or equal projected y.
  *
  *
  *           01 11 19 25 29 30
  *           02 12 20 26 27 28
  *           03 13 21 22 23 24
  *           04 14 15 16 17 18
  *           05 06 07 08 09 10
  *
  *
  *   _
  *   /\
  *    (Viewer)
  */
class DomainIterator (xpx: Double, xpy: Double, bdx: DomainAxis, bdy: DomainAxis) {

  private val swap: ((Int, Int)) => (Int, Int) = p => (p._2, p._1)
  private val dontSwap: ((Int, Int)) => (Int, Int) = p => (p._1, p._2)

  val (left, lDir, right, rDir, maybeSwap) =
    if ( xpx > 0 && xpy > 0 ) {
      (bdx, 1, bdy, 1, dontSwap)
    } else if ( xpx < 0 && xpy > 0 ) {
      (bdy, 1, bdx, -1, swap)
    } else if ( xpx > 0 && xpy < 0 ) {
      (bdy, -1, bdy, 1, swap)
    } else {
      (bdx, -1, bdy, -1, dontSwap)
    }

  private val la_lb = ((left.N -1 ) - lDir * (left.N -1 ) ) / 2
  private val la_rb = ((left.N -1 )  + lDir * (left.N -1 ) ) / 2
  private val ra_lb = ((right.N -1 )  - rDir * (right.N -1 ) ) / 2
  private val ra_rb = ((right.N -1 )  + rDir * (right.N -1 ) ) / 2
  private val iterator = CornerIterator ( la_lb, la_rb, ra_lb, ra_rb)

  def corner(k: Int) : List[(Double, Double)] = {

    val vals = iterator.corner(k)

    vals.map(p=>
      (bdx.lb + maybeSwap(p)._1 * bdx.d ,   bdy.lb + maybeSwap(p)._2 * bdy.d)
    )
  }

}


/**
  * one-dimensional domain, specified by
  * @param lb: lower boundary
  * @param ub: upper boundary
  * @param N: Number of points in the domain
  */
case class DomainAxis(lb: Double, ub: Double, N: Int) {
  val d: Double = (ub - lb)/(N-1)
}
