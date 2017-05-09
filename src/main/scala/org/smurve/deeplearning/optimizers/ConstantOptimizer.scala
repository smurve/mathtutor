package org.smurve.deeplearning.optimizers
import org.smurve.deeplearning.{DM, DV}

/**
  * the "trivial" gradient descend optimizer
  */
class ConstantOptimizer ( val eta: Double ) extends Optimizer {

  override def nextStep(gt: DM): DM = gt * eta

}
