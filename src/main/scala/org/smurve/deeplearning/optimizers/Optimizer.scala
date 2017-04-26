package org.smurve.deeplearning.optimizers

import org.smurve.deeplearning.{DM, DV}

/**
  * Created by wgiersche on 26.04.17.
  */
trait Optimizer {

  def nextStep ( gt: DM ) : DM

  def nextStep ( gt: DV ) : DV
}
