import gtda.diagrams
import numpy as np
from typing import List, Dict, Any



class TDA_Featurizer:


    """

    A class to extract TDA features from a persistence diagram list.

    """

    def __init__(self, PH_list: List[List[np.ndarray]]):
        self.PH_list = PH_list

    def preprocess_diagram_list_for_gtda(self):

        """

        Output:
        bdq_ordered_diagram_list - List[np.ndarray] - (num_diagrams, num_features, 3) list of diagrams, where each
        point in the output diagram is of the form [birth_time, death_time, homology_group]

        """

        diagram_list = self.PH_list

        bdq_ordered_diagram_list = []
        for diagram in diagram_list:
            ones_H0 = np.ones((len(diagram[0]), 1))
            ones_H1 = np.ones((len(diagram[1]), 1))
            H0 = np.hstack((np.array(diagram[0]), ones_H0 * 0))
            H1 = np.hstack((np.array(diagram[1]), ones_H1 * 1))

            combined_array = np.array([np.vstack((H0, H1))])

            bdq_ordered_diagram_list.append(combined_array)

        return bdq_ordered_diagram_list

    def amplitude_feature_from_diagram_list(
            self,
            amplitude_metric: str,
            metric_params: Dict[str, Any] | None = None,
            fit_params: Dict[str, Any] | None = None
            ):

        """

        Input:

        amplitude_metric - str - the metric to use for the amplitude calculation. Refer
        to the link below for a list of available metrics.

        metric_params - Dict[str, Any] | None - *optional, parameters for the gtda metric

        fit_params - Dict[str, Any] | None - *optional, parameters for the gtda fit_transform

        Refer to www.giotto-ai.github.io/gtda-docs/latest/modules/generated/diagrams/features/
        gtda.diagrams.Amplitude.html#gtda.diagrams.Amplitude for details on amplitude_metric,
        metric_params, and fit_params.

        Output:

        bdq_ordered_diagram_list - List[np.ndarray] - (num_diagrams, homology_group) list of
        amplitude metrics of each diagram in the list with the trivial diagonal diagram

        """

        diagam_list = self.PH_list

        if metric_params is not None:
            amplitude = gtda.diagrams.Amplitude(metric=amplitude_metric, metric_params=metric_params)
        else:
            amplitude = gtda.diagrams.Amplitude(metric=amplitude_metric)

        # preprocessing
        bdq_ordered_diagram_list = []
        for diagram in diagram_list:
            ones_H0 = np.ones((len(diagram[0]), 1))
            ones_H1 = np.ones((len(diagram[1]), 1))
            H0 = np.hstack((np.array(diagram[0]), ones_H0 * 0))
            H1 = np.hstack((np.array(diagram[1]), ones_H1 * 1))

            combined_array = np.array([np.vstack((H0, H1))])

            if fit_params is not None:
                metric = amplitude.fit_transform(X=combined_array, **fit_params)[0]
            else:
                metric = amplitude.fit_transform(X=combined_array)[0]

            bdq_ordered_diagram_list.append(metric)

        return bdq_ordered_diagram_list

    def diagram_list_to_betti_curves(
            self, n_bins: int = 100, n_jobs: int | None = None):

        """

        Input:

        n_bins - int - number of bins to use for the betti curve

        n_jobs - int | None - number of jobs to run in parallel. None means 1 job.


        Output:
        (betti_curve, fit_betti_curve)

        betti_curve - np.ndarray - (num_features, 2) betti curve

        fit_betti_curve - a gtda.diagrams.BettiCurve object fit to the input diagrams

        """

        diagram_list = self.PH_list
        betti_curve = gtda.diagrams.BettiCurve(n_bins=n_bins, n_jobs=n_jobs)

        # preprocessing
        bdq_ordered_diagram_list = []
        for diagram in diagram_list:
            ones_H0 = np.ones((len(diagram[0]), 1))
            ones_H1 = np.ones((len(diagram[1]), 1))
            H0 = np.hstack((np.array(diagram[0]), ones_H0 * 0))
            H1 = np.hstack((np.array(diagram[1]), ones_H1 * 1))
            combined_array = np.array([np.vstack((H0, H1))])

            result = betti_curve.fit_transform(combined_array)
            bdq_ordered_diagram_list.append(result)

        return bdq_ordered_diagram_list, betti_curve

    def plot_betti_curve(self, betti_curve: List[np.ndarray], fitBettiCurve: gtda.diagrams.BettiCurve):

        """
        Input:

        betti_curve - np.ndarray - (num_features, 2) betti curve

        fitBettiCurve - a gtda.diagrams.BettiCurve object fit to the input diagrams (used for plotting),
        obtained from diagram_list_to_betti_curves

        Output:

        plot - plotly.graph_objs..Figure - plot of the betti curve

        optionally, save the plot as an image using plot.write_image("betti_curve.png")

        """
        plot = fitBettiCurve.plot(betti_curve)

        # uncomment to save the plot
        plot.write_image("betti_curve.png")

        return plot


    def plot_persistence_image(self, diagram: np.ndarray, homology_group: int = 0):

        """

        Input:

        persistence_image - np.ndarray - (num_features, 3) persistence image

        Output:

        None - plots the persistence image

        """

        persistence_image = gtda.diagrams.PersistenceImage()
        result = persistence_image.fit_transform(diagram)
        a = persistence_image.plot(result, homology_dimension_idx=homology_group)

        # uncomment to save the plot
        a.write_image("persistence_image.png")

    def persistence_entropies_from_diagram_list(
            self,
            normalize: bool = False,
            nan_fill_value: float = -1.0,
            n_jobs: int | None = None
            ):

        """

        Input:

        normalize - When True, the persistence entropy of each diagram is normalized by
        the logarithm of the sum of lifetimes of all points in the diagram.

        nan_fill_value - float - If a float, (normalized) persistence entropies initially computed
        as numpy.nan are replaced with this value. If None, these values are left as numpy.nan

        n_jobs - int | None - The number of jobs to use for the computation. None
        means 1 unless in a joblib.parallel_backend context. -1 means using all processors.


        Output:

        bdq_ordered_diagram_list - List[np.ndarray] - (num_diagrams, homology_group) list of
        persistence entropies of each diagram in the list
        """

        diagram_list = self.PH_list

        persistence_entropy = gtda.diagrams.PersistenceEntropy(normalize=normalize, nan_fill_value=nan_fill_value, n_jobs=n_jobs)

        # preprocessing
        bdq_ordered_diagram_list = []
        for diagram in diagram_list:
            ones_H0 = np.ones((len(diagram[0]), 1))
            ones_H1 = np.ones((len(diagram[1]), 1))
            H0 = np.hstack((np.array(diagram[0]), ones_H0 * 0))
            H1 = np.hstack((np.array(diagram[1]), ones_H1 * 1))

            combined_array = np.array([np.vstack((H0, H1))])
            entropy = persistence_entropy.fit_transform(combined_array)
            bdq_ordered_diagram_list.append(entropy)

        return bdq_ordered_diagram_list
