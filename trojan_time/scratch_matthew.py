# compile persistence diagrams
ph_list = [x.PH_list for x in models]

def preprocess_diagram_list_for_gtda(diagram_list: List[List[np.ndarray]]):

    """

    Input:
    diagram_list - List[ List[ np.ndarray, np.ndarray] ] - list of persistance diagrams corresponding to
    a single model. A persistence diagram is a list of np.ndarrays of shape (num_features, 2), where the ith
    list is the ith homology group. Assumes two homology groups, H0 and H1.

    Output:
    bdq_ordered_diagram_list - List[np.ndarray] - (num_diagrams, num_features, 3) list of diagrams, where each
    point in the output diagram is of the form [birth_time, death_time, homology_group]

    """

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
        diagram_list: List[List[np.ndarray]],
        amplitude_metric: str,
        metric_params: Dict[str, Any] | None = None,
        fit_params: Dict[str, Any] | None = None
        ):

    """

    Input:

    diagram_list - List[ List[ np.ndarray, np.ndarray] ] - list of persistance diagrams
    corresponding to a single model. A persistence diagram is a list of np.ndarrays
    of shape (num_features, 2), where the ith list is the ith homology group. Assumes
    two homology groups, H0 and H1.

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

# ph_list = [x.PH_list for x in models]
# wasserstein_amplitude = amplitude_feature_from_diagram_list(
#         diagram_list=ph_list[0], amplitude_metric="wasserstein", metric_params={"p": 3}, fit_params=None)
# print(wasserstein_amplitude)
# exit()
# persistence_image = amplitude_feature_from_diagram_list(
#         diagram_list=ph_list[0], amplitude_metric="persistence_image"
#         )
# print(persistence_image)

def diagram_list_to_betti_curves(
        diagram_list: List[np.ndarray], n_bins: int = 100, n_jobs: int | None = None):

    """

    Input:

    diagram_list - List[ List[ np.ndarray, np.ndarray] ] - list of persistance diagrams corresponding to
    a single model. A persistence diagram is a list of np.ndarrays of shape (num_features, 2), where the ith
    list is the ith homology group. Assumes two homology groups, H0 and H1.

    n_bins - int - number of bins to use for the betti curve

    n_jobs - int | None - number of jobs to run in parallel. None means 1 job.

    Output:

    (betti_curve, fit_betti_curve)

    betti_curve - np.ndarray - (num_features, 2) betti curve

    fit_betti_curve - a gtda.diagrams.BettiCurve object fit to the input diagrams

    """

    betti_curve = gtda.diagrams.BettiCurve(n_bins=n_bins, n_jobs=n_jobs)
    # result = betti_curve.fit_transform(diagram, n_bins=n_bins, n_jobs=n_jobs)

    # preprocessing
    bdq_ordered_diagram_list = []
    for diagram in diagram_list:
        ones_H0 = np.ones((len(diagram[0]), 1))
        ones_H1 = np.ones((len(diagram[1]), 1))
        H0 = np.hstack((np.array(diagram[0]), ones_H0 * 0))
        H1 = np.hstack((np.array(diagram[1]), ones_H1 * 1))

        combined_array = np.array([np.vstack((H0, H1))])

        result = betti_curve.fit_transform(combined_array)
        # plot = betti_curve.plot(result)
        # plot.write_image("betti_curve.png")
        # exit()
        # if fit_params is not None:
            # metric = amplitude.fit_transform(X=combined_array, **fit_params)[0]
        # else:
            # metric = amplitude.fit_transform(X=combined_array)[0]

        bdq_ordered_diagram_list.append(result)

    return bdq_ordered_diagram_list, betti_curve

def plot_betti_curve(betti_curve: List[np.ndarray], fitBettiCurve: gtda.diagrams.BettiCurve):

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

# curves, fitBettiCurve = diagram_list_to_betti_curves(ph_list[0], n_bins=100, n_jobs=None)
# plot_betti_curve(betti_curve=curves[0], fitBettiCurve=fitBettiCurve)


def plot_persistence_image(diagram: np.ndarray, homology_group: int = 0):

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

# diagram = pickle.load(open("persistence_diagram.pkl", "rb"))
# plot_persistence_image(diagram[0], homology_group=1)

def wasserstein_distance_from_diagram_list(diagram_list: List[List[np.ndarray]]):
    """

    Input:

        diagram_list - List[ List[ np.ndarray, np.ndarray] ] - list of persistance diagrams
    corresponding to a single model. A persistence diagram is a list of np.ndarrays
    of shape (num_features, 2), where the ith list is the ith homology group. Assumes
    two homology groups, H0 and H1.

    Output:

    bdq_ordered_diagram_list - List[np.ndarray] - (num_diagrams, homology_group) list of
    wasserstein distances of each diagram in the list with the trivial diagonal diagram

    """

    wasserstein_amplitude = gtda.diagrams.Amplitude(metric="wasserstein")

    # preprocessing
    bdq_ordered_diagram_list = []
    for diagram in diagram_list:
        ones_H0 = np.ones((len(diagram[0]), 1))
        ones_H1 = np.ones((len(diagram[1]), 1))
        H0 = np.hstack((np.array(diagram[0]), ones_H0 * 0))
        H1 = np.hstack((np.array(diagram[1]), ones_H1 * 1))

        combined_array = np.array([np.vstack((H0, H1))])
        wasserstein_distance = wasserstein_amplitude.fit_transform(combined_array)[0]
        bdq_ordered_diagram_list.append(wasserstein_distance)

    return bdq_ordered_diagram_list


# ph_list = [x.PH_list for x in models]
# print(ph_list[0][0])
# wasserstein_distances = wasserstein_distance_from_diagram_list(ph_list[0])
# print(wasserstein_distances)

def persistence_entropies_from_diagram_list(
        diagram_list: List[List[np.ndarray]],
        normalize: bool = False,
        nan_fill_value: float = -1.0,
        n_jobs: int | None = None
        ):

    """

    Input:

    diagram_list - List[ List[ np.ndarray, np.ndarray] ] - list of persistance diagrams
    corresponding to a single model. A persistence diagram is a list of np.ndarrays
    of shape (num_features, 2), where the ith list is the ith homology group. Assumes
    two homology groups, H0 and H1.

    Output:

    bdq_ordered_diagram_list - List[np.ndarray] - (num_diagrams, homology_group) list of
    persistence entropies of each diagram in the list
    """

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

# entropies = persistence_entropies_from_diagram_list(ph_list[0])
# print(entropies)

