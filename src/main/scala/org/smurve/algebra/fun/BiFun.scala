package org.smurve.algebra.fun

import org.smurve.complex.Cpx

abstract class BiFun(val f: Fun, val g: Fun, evalF: Cpx => Cpx) extends Fun(evalF) {
}

