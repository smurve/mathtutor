package org.smurve.charts

import breeze.linalg.{DenseVector=>DV}
import org.smurve.complex.Cpx

class PlaneProjector (phi_deg: Double, theta_deg: Double) {

  /**
    * deg to rad conversion
    *
    * @param deg the angle in 360° coordinates
    */
  def rad(deg: Double): Double = deg * (2 * math.Pi / 360)

  private val (xpx, xpy, xpz) = xpnv(rad(phi_deg))
  private val (ypx, ypy, ypz) = ypnv(rad(phi_deg), rad(theta_deg))
  private val _zp = zpnv(rad(phi_deg), rad(theta_deg))

  /** projection normal vector in projected x direction
    */
  private def xpnv(phi: Double): (Double, Double, Double) = (-math.sin(phi), math.cos(phi), 0)

  /** projection normal vector in projected y direction
    */
  private def ypnv(phi: Double, theta: Double): (Double, Double, Double) = (
    -math.cos(phi) * math.sin(theta),
    -math.sin(phi) * math.sin(theta),
    math.cos(theta)
  )

  /** projection normal vector towards the center of the domain
    */
  private def zpnv(phi: Double, theta: Double): DV[Double] = {
    DV(-math.cos(phi) * math.cos(theta), -math.sin(phi) * math.cos(theta), -math.sin(theta))
  }

  /**
    * @return the z normal of the projection plane
    */
  def zp: DV[Double] = _zp

  def normalVectors: (DV[Double], DV[Double], DV[Double]) = (DV(xpx, xpy, xpz), DV(ypx, ypy, ypz), _zp)

  /**
    * @param xo the original x coordinate
    * @param yo the original y coordinate
    * @param zo the original z coordinate
    * @return the coordinates of the projection of the given point
    */
  def project(xo: Double, yo: Double, zo: Double): (Double, Double) =
    (xo * xpx + yo * xpy, xo * ypx + yo * ypy + zo * ypz)

  /**
    * @param zo : the complex number to be projected
    * @param fz : the real function value at zo
    * @return the coordinates of the projection of the function value
    */
  def project(zo: Cpx, fz: Double): (Double, Double) = project(zo.r, zo.i, fz)
}
