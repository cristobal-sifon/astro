def plot_radec(plot_func, ra, dec, *args, wrap=180, **kwargs):
    """
    Plot catalog of objects on the sky.

    Parameters
    ----------
    plot_func : callable
        plotting function
    ra, dec : float arrays
        RA and Dec (or x and y) of catalog
    wrap : float, optional
        RA (or x) at which to wrap the plot
    args, kwargs
        arguments of ``plot_func``
    """
    if 'label' in kwargs:
        label = kwargs['label']
        kwargs.pop('label')
    else:
        label = '_none_'
    highra = (ra > wrap)
    plot_func(ra[~highra], dec[~highra], *args, label='_none_', **kwargs)
    plot_func(ra[highra], dec[highra], *args, label='_none_', **kwargs)
    plot_func([], [], *args, label=label, **kwargs)
    return
