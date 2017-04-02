package org.smurve.deeplearning

/**
  * A cost (loss) function and its derivative
  * @param fn the cost/loss function
  * @param deriv its derivative
  */
case class CostFunction
(
  /**
    * first parameter: Final activation
    * second parameter: Desired output
    */
  fn: (DV, DV) => Double,
  /**
    * first parameter: Final activation
    * second parameter: Desired output
    */
  deriv: ( DV, DV ) => DV
)

