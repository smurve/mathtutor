package org.smurve.charts

/**
  * corner iterator for two axis ranging from lb to and including rb respectively
  */
case class CornerIterator(la_lb: Int,
                          la_rb: Int, // left axis right boundary
                          ra_lb: Int,
                          ra_rb: Int) {

  val d_la: Int = math.signum(la_rb - la_lb)
  val d_ra: Int = math.signum(ra_rb - ra_lb)

  def corner(k: Int): List[(Int, Int)] = {
    val la_rb_k = la_rb - k * d_la
    // left axis right boundary off by k
    val ra_lb_k1 = ra_lb + d_ra * (k + 1) // right axis

    (la_lb to la_rb_k by d_la).map(m =>

      (m, ra_lb + k * d_ra)

    ).toList ::: (ra_lb_k1 to ra_rb by d_ra).map(n =>

      (la_rb_k, n)

    ).toList
  }
}

