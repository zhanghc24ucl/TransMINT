import csv
import datetime as dt
from typing import Dict, List, Optional, Tuple, Union

import gpflow
import numpy as np
import pandas as pd
import tensorflow as tf
from gpflow.kernels import ChangePoints, Matern32
from sklearn.preprocessing import StandardScaler
from tensorflow_probability import bijectors as tfb

Kernel = gpflow.kernels.base.Kernel

MAX_ITERATIONS = 200


class ChangePointsWithBounds(ChangePoints):
    def __init__(
        self,
        kernels: Tuple[Kernel, Kernel],
        location: float,
        interval: Tuple[float, float],
        steepness: float = 1.0,
        name: Optional[str] = None,
    ):
        """Overwrite the ChangePoints class to
        1) only take a single location
        2) so location is bounded by interval


        Args:
            kernels (Tuple[Kernel, Kernel]): the left hand and right hand kernels
            location (float): changepoint location initialisation, must lie within interval
            interval (Tuple[float, float]): the interval which bounds the changepoint hyperparameter
            steepness (float, optional): initialisation of the steepness parameter. Defaults to 1.0.
            name (Optional[str], optional): class name. Defaults to None.

        Raises:
            ValueError: errors if initial changepoint location is not within interval
        """
        # overwrite the locations variable to enforce bounds
        if location < interval[0] or location > interval[1]:
            raise ValueError(
                "Location {loc} is not in range [{low},{high}]".format(
                    loc=location, low=interval[0], high=interval[1]
                )
            )

        locations = [location]
        super().__init__(
            kernels=kernels, locations=locations, steepness=steepness, name=name
        )

        affine = tfb.Shift(tf.cast(interval[0], tf.float64))(
            tfb.Scale(tf.cast(interval[1] - interval[0], tf.float64))
        )
        self.locations = gpflow.base.Parameter(
            locations, transform=tfb.Chain([affine, tfb.Sigmoid()]), dtype=tf.float64
        )

    def _sigmoids(self, X: tf.Tensor) -> tf.Tensor:
        # overwrite to remove sorting of locations
        locations = tf.reshape(self.locations, (1, 1, -1))
        steepness = tf.reshape(self.steepness, (1, 1, -1))
        return tf.sigmoid(steepness * (X[:, :, None] - locations))

def fit_matern_kernel(
        X, Y,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        likelihood_variance: float = 1.0,) -> Tuple[float, Dict[str, float]]:
    """Fit the Matern 3/2 kernel on a time-series

    Args:
        X (pp.array): indices in shape (n,)
        Y (np.array): series in shape (n,)
        variance (float, optional): variance parameter initialisation. Defaults to 1.0.
        lengthscale (float, optional): length_scale parameter initialisation. Defaults to 1.0.
        likelihood_variance (float, optional): likelihood variance parameter initialisation. Defaults to 1.0.

    Returns:
        Tuple[float, Dict[str, float]]: negative log marginal likelihood and parameters after fitting the GP
    """
    m = gpflow.models.GPR(
        data=(X, Y),
        kernel=Matern32(variance=variance, lengthscales=lengthscale),
        noise_variance=likelihood_variance,
    )
    opt = gpflow.optimizers.Scipy()
    nlml = opt.minimize(
        m.training_loss, m.trainable_variables, options=dict(maxiter=MAX_ITERATIONS)
    ).fun
    params = {
        "kM_variance": m.kernel.variance.numpy(),
        "kM_lengthscales": m.kernel.lengthscales.numpy(),
        "kM_likelihood_variance": m.likelihood.variance.numpy(),
    }
    return nlml, params


def fit_changepoint_kernel(
    time_series_data: pd.DataFrame,
    k1_variance: float = 1.0,
    k1_lengthscale: float = 1.0,
    k2_variance: float = 1.0,
    k2_lengthscale: float = 1.0,
    kC_likelihood_variance=1.0,
    kC_changepoint_location=None,
    kC_steepness=1.0,
) -> Tuple[float, float, Dict[str, float]]:
    """Fit the Changepoint kernel on a time-series

    Args:
        time_series_data (pd.DataFrame): time-series with ciolumns X and Y
        k1_variance (float, optional): variance parameter initialisation for k1. Defaults to 1.0.
        k1_lengthscale (float, optional): lengthscale initialisation for k1. Defaults to 1.0.
        k2_variance (float, optional): variance parameter initialisation for k2. Defaults to 1.0.
        k2_lengthscale (float, optional): lengthscale initialisation for k2. Defaults to 1.0.
        kC_likelihood_variance (float, optional): likelihood variance parameter initialisation. Defaults to 1.0.
        kC_changepoint_location (float, optional): changepoint location initialisation, if None uses midpoint of interval. Defaults to None.
        kC_steepness (float, optional): steepness parameter initialisation. Defaults to 1.0.

    Returns:
        Tuple[float, float, Dict[str, float]]: changepoint location, negative log marginal likelihood and paramters after fitting the GP
    """
    if not kC_changepoint_location:
        kC_changepoint_location = (
            time_series_data["X"].iloc[0] + time_series_data["X"].iloc[-1]
        ) / 2.0

    m = gpflow.models.GPR(
        data=(
            time_series_data.loc[:, ["X"]].to_numpy(),
            time_series_data.loc[:, ["Y"]].to_numpy(),
        ),
        kernel=ChangePointsWithBounds(
            [
                Matern32(variance=k1_variance, lengthscales=k1_lengthscale),
                Matern32(variance=k2_variance, lengthscales=k2_lengthscale),
            ],
            location=kC_changepoint_location,
            interval=(time_series_data["X"].iloc[0], time_series_data["X"].iloc[-1]),
            steepness=kC_steepness,
        ),
    )
    m.likelihood.variance.assign(kC_likelihood_variance)
    opt = gpflow.optimizers.Scipy()
    nlml = opt.minimize(
        m.training_loss, m.trainable_variables, options=dict(maxiter=200)
    ).fun
    changepoint_location = m.kernel.locations[0].numpy()
    params = {
        "k1_variance": m.kernel.kernels[0].variance.numpy().flatten()[0],
        "k1_lengthscale": m.kernel.kernels[0].lengthscales.numpy().flatten()[0],
        "k2_variance": m.kernel.kernels[1].variance.numpy().flatten()[0],
        "k2_lengthscale": m.kernel.kernels[1].lengthscales.numpy().flatten()[0],
        "kC_likelihood_variance": m.likelihood.variance.numpy().flatten()[0],
        "kC_changepoint_location": changepoint_location,
        "kC_steepness": m.kernel.steepness.numpy(),
    }
    return changepoint_location, nlml, params

