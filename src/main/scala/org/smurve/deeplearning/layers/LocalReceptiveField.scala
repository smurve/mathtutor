package org.smurve.deeplearning.layers

/**
  * A model for the local receptive field for convolutional layers
  *
  * @param input_cols number of columns of the input matrix
  * @param input_rows number of rows of the input matrix
  * @param lrf_cols   number of columns of the local receptive field
  * @param lrf_rows   number of rows of the local receptive field
  */
case class LocalReceptiveField(input_cols: Int, input_rows: Int, lrf_cols: Int, lrf_rows: Int) {

  val input_size: Int = input_rows * input_cols
  val feature_cols: Int = input_cols - lrf_cols + 1
  val feature_rows: Int = input_rows - lrf_rows + 1
  val size_featureMap: Int = feature_cols * feature_rows
  val size: Int = lrf_rows * lrf_cols


  /**
    * calculate the domain's index given target vector index = k
    * and the local receptive field's index j
    */
  @inline
  def tau(k: Int, j: Int): Int = phi(j) + xi(k)

  /**
    * @param k offset
    * @return the tau function applied to the entire window
    */
  @inline
  def tau(k: Int): Array[Int] = (0 until size).map(tau(_, k)).toArray

  @inline
  private def phi(j: Int): Int = j % lrf_cols + (j / lrf_cols) * input_cols

  @inline
  private def xi(k: Int): Int = (k / feature_cols) * input_cols + k % feature_cols
}
