import numpy as np

from scipy.fft import fftn
from scipy.fft import ifftn
from scipy.fft import fftfreq


def phase_cross_correlation(
    image, 
    template, 
    upsample_factor = 1,
    max_shift = (10, 10),
    space = "real"
):
    """
    A slightly modified version of the skimage `phase_cross_correlation`
    function for efficient sub-pixel image registration by a cross-correlation.

    See their documentation and notes here: 
     - https://scikit-image.org/docs/stable/api/skimage.registration.html

    Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT [1]_.

    Parameters
    ----------
    image : np.ndarray
        The image to register. 

    template : np.ndarray
        The template to register 'image' against.

    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered within
        1/20th of a pixel. Default is 1 (no upsampling).  Not used if any of
        ``reference_mask`` or ``moving_mask`` is not None.

    max_shift : tuple, list, optional
        The maximum shift allowed when registering.

    space : string, one of "real" or "fourier", optional
        Defines how the algorithm interprets input data. "real" means data will
        be FFT'd to compute the correlation, while "fourier" data will bypass
        FFT of input data. Case insensitive. Not used if any of
        ``reference_mask`` or ``moving_mask`` is not None.

    Returns
    -------
    shift : ndarray
        Shift vector (in pixels) required to register ``moving_image``
        with ``reference_image``. Axis ordering is consistent with
        the axis order of the input array.

    Notes
    -----
    The use of cross-correlation to estimate image translation has a long
    history dating back to at least [2]_. The "phase correlation"
    method (selected by ``normalization="phase"``) was first proposed in [3]_.
    Publications [1]_ and [2]_ use an unnormalized cross-correlation
    (``normalization=None``). Which form of normalization is better is
    application-dependent. For example, the phase correlation method works
    well in registering images under different illumination, but is not very
    robust to noise. In a high noise scenario, the unnormalized method may be
    preferable.

    When masks are provided, a masked normalized cross-correlation algorithm is
    used [5]_, [6]_.

    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008). :DOI:`10.1364/OL.33.000156`
    .. [2] P. Anuta, Spatial registration of multispectral and multitemporal
           digital imagery using fast Fourier transform techniques, IEEE Trans.
           Geosci. Electron., vol. 8, no. 4, pp. 353–368, Oct. 1970.
           :DOI:`10.1109/TGE.1970.271435`.
    .. [3] C. D. Kuglin D. C. Hines. The phase correlation image alignment
           method, Proceeding of IEEE International Conference on Cybernetics
           and Society, pp. 163-165, New York, NY, USA, 1975, pp. 163–165.
    .. [4] James R. Fienup, "Invariant error metrics for image reconstruction"
           Optics Letters 36, 8352-8357 (1997). :DOI:`10.1364/AO.36.008352`
    .. [5] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [6] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`

    """ 
    # Images must be the same shape.
    if image.shape != template.shape:
        raise ValueError(
            "The images must be same size for `phase_cross_correlation`."
        )

    # Make sure upsample_factor >= 1.
    if upsample_factor < 1:
        raise ValueError(
            "'upsample_factor' must be greater than or equal to 1."
        )

    # Only 2D data makes sense right now.
    if image.ndim != 2 and upsample_factor > 1:
        raise NotImplementedError(
            "'phase_cross_correlation' only supports subpixel registration "
            "for 2D images."
        )

    # Assume complex data is already in Fourier space.
    if space.lower() == "fourier":
        src_freq = image 
        target_freq = template 
    # Real data needs to be fft'd.
    elif space.lower() == "real":
        src_freq = fftn(image)
        target_freq = fftn(template)
    else:
        raise ValueError("The 'space' argument must be 'real' of 'fourier'.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT.
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()

    cross_correlation = ifftn(image_product)
    cross_correlation = np.abs(cross_correlation)

    # Remove any information outside of the maximum allowed shift.
    cross_correlation[   max_shift[0] : -max_shift[0]] = 0
    cross_correlation[:, max_shift[1] : -max_shift[1]] = 0

    # Locate maximum.
    maxima = np.unravel_index(
        np.argmax(cross_correlation), 
        cross_correlation.shape
    )

    # Explicit stacking does not work with some backends?
    #shift = np.array(maxima, dtype = np.float64)
    shift = np.stack(maxima).astype(np.float64, copy = False)

    midpoints = np.array([np.fix(axis_size // 2) for axis_size in shape])
    shift[shift > midpoints] -= np.array(shape)[shift > midpoints]

    if upsample_factor > 1:
        upsample_factor = np.array(upsample_factor, dtype = np.float64)

        # Initial shift estimate in upsampled grid.
        shift = np.round(shift * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)

        # Center of output array at dftshift + 1.
        dftshift = np.fix(upsampled_region_size / 2.0)

        # Matrix multiply DFT around the current shift estimate.
        sample_region_offset = dftshift - shift * upsample_factor

        cross_correlation = _upsampled_dft(
            image_product.conj(),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset
        ).conj()

        # See reasoning for removing the normalization here:
        #  - https://github.com/scikit-image/scikit-image/pull/4901
        #normalization = (src_freq.size * upsample_factor ** 2)
        #cross_correlation /= normalization

        # Locate maximum and map back to original pixel grid.
        maxima = np.unravel_index(
            np.argmax(np.abs(cross_correlation)), 
            cross_correlation.shape
        )
        maxima = np.stack(maxima).astype(np.float64, copy = False)
        maxima = maxima - dftshift
        shift = shift + (maxima / upsample_factor)

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shift[dim] = 0

    return shift


def _upsampled_dft(
    data,
    upsampled_region_size,
    upsample_factor = 1, 
    axis_offsets = None
) -> np.ndarray:
    """
    This is a copy of the skimage '_upsample_dft()' function. More information
    can be found here:
     - https://scikit-image.org/docs/stable/api/skimage.registration.html

    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension. ifftshift to bring the center of the image 
          to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Parameters
    ----------
    data : np.ndarray
        The input data array (DFT of original data) to upsample.

    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled. If one integer is provided, it is
        duplicated up to the dimensionality of ``data``.

    upsample_factor : integer, optional
        The upsampling factor. Defaults to 1.

    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses image
        center).

    Returns 
    ----------
    output : np.ndarray
        The upsampled DFT of the specified region.

    """
    # If people pass in an integer, expand it to a list of equal-sized sections.
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError(
                "The shape of upsampled region sizes must be equal to input "
                "data's number of dimensions."
            )

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError(
                "The number of axis offsets must be equal to input data's "
                "number of dimensions."
            )

    im2pi = 1j * 2 * np.pi

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))
    for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
        kernel = (
            (np.arange(ups_size) - ax_offset)[:, None] * fftfreq(n_items, upsample_factor)
        )
        kernel = np.exp(-im2pi * kernel)

        # Use kernel with same precision as the data.
        kernel = kernel.astype(data.dtype, copy = False)

        # Equivalent to: `data[i, j, k] = kernel[i, :] @ data[j, k].T`
        data = np.tensordot(kernel, data, axes = (1, -1))

    return data


