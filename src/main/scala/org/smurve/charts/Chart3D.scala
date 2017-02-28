package org.smurve.charts

import org.smurve.algebra.fun.Fun
import org.smurve.complex.Cpx

/**
  * parallel projection for an observing plane from default direction:
  * phi =   30° from y-axis,
  * theta = 30° from z-plane
  */
class Chart3D ( phi_deg: Double = 30, theta_deg: Double = 30 ) ( implicit spec: PlotterSpec ){

  /**
    * deg to rad conversion
    * @param deg the angle in 360° coordinates
    */
  private def rad ( deg: Double ) = deg * ( 2 * math.Pi / 360 )

  private def fromAngle ( phi_deg: Double, theta_deg: Double ) : Unit = {
    (xpx, xpy, xpz) = xp ( rad(phi_deg ))
    (ypx, ypy, ypz) = yp ( rad(phi_deg), rad(theta_deg))
  }

  def showFun ( fun: Fun ): Unit = {

  }

  def display () : Unit = {
  }

  // projection normal vector in projected x direction
  private def xp(phi: Double ) : (Double, Double, Double) = (
    math.cos(phi), math.sin(phi), 0)

  // projection normal vector in projected y direction
  private def yp( phi: Double, theta: Double ): (Double, Double, Double) = (
    -math.sin(phi) * math.cos(theta),
    -math.cos(phi) * math.cos(theta),
    math.sin(phi)
  )

  private var (xpx, xpy, xpz) = xp ( rad(phi_deg ))
  private var (ypx, ypy, ypz) = yp ( rad(phi_deg), rad(theta_deg))
  /**
    * @param xo the original x coordinate
    * @param yo the original y coordinate
    * @param zo the original z coordinate
    * @return the coordinates of the projection of the given point
    */
  def project ( xo: Double, yo: Double, zo: Double ) : (Double, Double) =
    ( xo * xpx + yo * xpy, xo * ypx + yo * ypy + zo * ypz )

  /**
    * @param zo: the complex number to be projected
    * @param fz: the real function value at zo
    * @return the coordinates of the projection of the function value
    */
  def project ( zo: Cpx, fz: Double ): (Double, Double ) = project ( zo.r, zo.i, fz )
}
