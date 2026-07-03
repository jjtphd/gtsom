"""
utils.py — Utility classes and functions for the gtsom package.
"""

import numpy as np


import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------

class DataValidator:
    """
    Validates a data matrix X (and optionally observation labels) and stores
    lightweight fingerprints for consistency checking across subsequent calls.

    Construction validates X (must be 2-D) and records its fingerprint, along
    with an optional label fingerprint derived from ``labels``.  The
    :meth:`check` method warns whenever a later ``(X, labels)`` pair does not
    match what was seen at construction.

    Parameters
    ----------
    X : array-like
        Data matrix to validate and fingerprint.
    labels : array-like or None, optional
        Observation labels.  If provided, a label fingerprint is stored as a
        ``{str_label: count}`` dict so that label-universe and
        per-class-count mismatches can be detected at :meth:`check` time.

    Raises
    ------
    ValueError
        If X is not 2-D after conversion to ndarray.

    Examples
    --------
    >>> import numpy as np
    >>> validator = DataValidator(np.ones((100, 5)))
    >>> validator.shape
    (100, 5)
    >>> validator.check(np.ones((100, 5)))              # silent — matches
    >>> validator.check(np.ones((200, 5)), context="fit")  # warns: X mismatch

    With labels:

    >>> y = np.array([0, 0, 1, 1, 2, 2])
    >>> validator = DataValidator(np.ones((6, 3)), labels=y)
    >>> validator.check(np.ones((6, 3)), labels=y)      # silent — matches
    >>> validator.check(np.ones((6, 3)), labels=None, context="fit")  # warns: labels missing
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, X, labels=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        self._fingerprint = (
            X.shape,
            X.dtype,
            float(X.mean()),
            float(X.std()),
            float(X.min()),
            float(X.max()),
        )
        self._label_fingerprint = self._build_label_fingerprint(labels)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self):
        """Shape (N, d) of the validated dataset."""
        return self._fingerprint[0]

    @property
    def dtype(self):
        """dtype of the validated dataset."""
        return self._fingerprint[1]

    @property
    def has_labels(self):
        """True if labels were provided at construction."""
        return self._label_fingerprint is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, X, labels=None, context=""):
        """
        Warn if X or labels do not match the stored fingerprints.

        Four distinct warning cases are handled:

        1. **X mismatch** — the incoming ``X`` has a different shape, dtype,
           or summary statistics from the dataset seen at construction.
        2. **Labels missing at check time** — labels were provided at
           construction but ``labels=None`` here; ``recaller.WL`` will be
           stale after this call.
        3. **Label universe / count mismatch** — both construction and this
           call provided labels, but the set of unique values or their per-class
           counts differ; ``ctab`` and snapshot ``WL`` may be inconsistent.
        4. **Labels provided only at check time** — labels were not provided at
           construction; re-initialize the SOM with labels to enable label
           coloring.

        Parameters
        ----------
        X : array-like
            Incoming data matrix to check.
        labels : array-like or None, optional
            Incoming observation labels to check.
        context : str, optional
            Caller name included in warning messages for traceability
            (e.g. ``"fit"``).
        """
        prefix = f"{context}: " if context else ""

        # --- Case 1: X fingerprint ---
        X = np.asarray(X)
        current_fp = (
            X.shape,
            X.dtype,
            float(X.mean()),
            float(X.std()),
            float(X.min()),
            float(X.max()),
        )
        if current_fp != self._fingerprint:
            fp = self._fingerprint
            warnings.warn(
                f"{prefix}X does not match the dataset used at initialization. "
                f"Expected shape={fp[0]}, dtype={fp[1]}, "
                f"mean={fp[2]:.4g}, std={fp[3]:.4g}, "
                f"min={fp[4]:.4g}, max={fp[5]:.4g}. "
                f"Got shape={current_fp[0]}, dtype={current_fp[1]}, "
                f"mean={current_fp[2]:.4g}, std={current_fp[3]:.4g}, "
                f"min={current_fp[4]:.4g}, max={current_fp[5]:.4g}.",
                UserWarning,
                stacklevel=3,
            )

        # --- Cases 2 / 3 / 4: label fingerprint ---
        init_had_labels  = self._label_fingerprint is not None
        check_has_labels = labels is not None

        if init_had_labels and not check_has_labels:
            # Case 2
            warnings.warn(
                f"{prefix}Labels were provided at initialization but not passed "
                f"here. recaller.WL will not be updated and will reflect "
                f"prototype–label assignments from the previous epoch.",
                UserWarning,
                stacklevel=3,
            )
        elif init_had_labels and check_has_labels:
            # Case 3: compare fingerprints
            current_lf = self._build_label_fingerprint(labels)
            if current_lf != self._label_fingerprint:
                warnings.warn(
                    f"{prefix}Labels passed here differ from those provided at "
                    f"initialization (different label universe or per-class "
                    f"counts). ctab and snapshot WL values may be inconsistent. "
                    f"Expected label counts: {self._label_fingerprint}. "
                    f"Got: {current_lf}.",
                    UserWarning,
                    stacklevel=3,
                )
        elif not init_had_labels and check_has_labels:
            # Case 4
            warnings.warn(
                f"{prefix}Labels were not provided at initialization. "
                f"Re-initialize the SOM via from_grid() or from_data() with "
                f"labels= to enable label coloring and consistent snapshot WL.",
                UserWarning,
                stacklevel=3,
            )
        # else: neither had labels — silent, nothing to check.

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        label_info = (
            f", labels={list(self._label_fingerprint.keys())}"
            if self._label_fingerprint is not None
            else ""
        )
        return (
            f"DataValidator(shape={self._fingerprint[0]}, "
            f"dtype={self._fingerprint[1]}{label_info})"
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self):
        """
        Return a plain dict representation of this validator's fingerprints.

        The training data ``X`` is not stored — only the summary statistics
        needed to detect mismatches in subsequent :meth:`check` calls.
        The dtype is stored as a string and cast back to ``np.dtype`` on load.

        Returns
        -------
        dict with keys:
            shape   list of int
            dtype   str
            mean    float
            std     float
            min     float
            max     float
            has_labels  bool
            label_keys    np.ndarray of object (str labels) or None
            label_counts  np.ndarray of int64 or None
        """
        fp = self._fingerprint
        d = {
            "shape": list(fp[0]),
            "dtype": str(fp[1]),
            "mean":  float(fp[2]),
            "std":   float(fp[3]),
            "min":   float(fp[4]),
            "max":   float(fp[5]),
            "has_labels": self._label_fingerprint is not None,
        }
        if self._label_fingerprint is not None:
            keys   = list(self._label_fingerprint.keys())
            counts = list(self._label_fingerprint.values())
            d["label_keys"]   = np.array(keys,   dtype=object)
            d["label_counts"] = np.array(counts, dtype=np.int64)
        else:
            d["label_keys"]   = None
            d["label_counts"] = None
        return d

    @classmethod
    def from_dict(cls, d):
        """
        Reconstruct a DataValidator from a dict produced by :meth:`to_dict`.

        Bypasses ``__init__`` so that the stored fingerprint is restored
        exactly without requiring the original data ``X``.

        Parameters
        ----------
        d : dict
            As returned by :meth:`to_dict`.

        Returns
        -------
        DataValidator
        """
        obj = cls.__new__(cls)
        obj._fingerprint = (
            tuple(int(x) for x in d["shape"]),
            np.dtype(d["dtype"]),
            float(d["mean"]),
            float(d["std"]),
            float(d["min"]),
            float(d["max"]),
        )
        if d.get("has_labels") and d.get("label_keys") is not None:
            keys   = [str(k) for k in d["label_keys"]]
            counts = [int(c) for c in d["label_counts"]]
            obj._label_fingerprint = dict(zip(keys, counts))
        else:
            obj._label_fingerprint = None
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_label_fingerprint(labels):
        """
        Build a ``{str_label: count}`` dict from ``labels``, or return
        ``None`` if ``labels`` is ``None``.

        Labels are str-coerced before counting so that, for example,
        ``np.int64(0)`` and ``int(0)`` both map to ``'0'`` and compare
        equal across sessions.  None/NaN entries are silently excluded.
        """
        if labels is None:
            return None
        arr = np.asarray(labels).ravel()
        # Exclude missing values; coerce survivors to str
        str_labels = [
            str(v) for v in arr
            if not (v is None or (isinstance(v, float) and np.isnan(v)))
        ]
        if not str_labels:
            return {}
        uniq, counts = np.unique(str_labels, return_counts=True)
        return dict(zip(uniq.tolist(), counts.tolist()))


# ---------------------------------------------------------------------------
# Dimensionality reduction helpers (used by GTSOM.from_data)
# ---------------------------------------------------------------------------

def reduce_coords_pca(W, coord_dim, random_state=None):
    """
    Reduce prototype matrix W to coord_dim dimensions via randomised PCA.

    Parameters
    ----------
    W : np.ndarray, shape (M, d)
        Prototype matrix.
    coord_dim : int
        Target dimensionality (2 or 3).
    random_state : int or None, default None

    Returns
    -------
    coords : np.ndarray, shape (M, coord_dim), float32
    """
    pca = PCA(n_components=coord_dim, svd_solver='randomized',
              random_state=random_state)
    coords = pca.fit_transform(W)
    return coords.astype(np.float32)


def reduce_coords_le(W, coord_dim, random_state=None):
    """
    Reduce prototype matrix W to coord_dim dimensions via Laplacian Eigenmaps.

    Uses sklearn SpectralEmbedding with a nearest-neighbors affinity graph
    built from the prototype vectors.

    Parameters
    ----------
    W : np.ndarray, shape (M, d)
        Prototype matrix.
    coord_dim : int
        Target dimensionality (2 or 3).
    random_state : int or None, default None

    Returns
    -------
    coords : np.ndarray, shape (M, coord_dim), float32
    """
    se = SpectralEmbedding(
        n_components=coord_dim,
        affinity='nearest_neighbors',
        random_state=random_state,
    )
    coords = se.fit_transform(W)
    return coords.astype(np.float32)


def reduce_coords_random(M, coord_dim, random_state=None):
    """
    Generate M neuron coordinates sampled uniformly from the unit hypercube.

    Produces a completely unstructured initial layout with no relationship to
    the prototype vectors. Useful for exposition — starting from a worst-case
    random state makes the organising effect of SOM training most visible.

    Parameters
    ----------
    M : int
        Number of prototypes (neurons).
    coord_dim : int
        Dimensionality of the output space (2 or 3).
    random_state : int or None, default None

    Returns
    -------
    coords : np.ndarray, shape (M, coord_dim), float32
        Coordinates drawn uniformly from [0, 1)^coord_dim.
    """
    rng = np.random.default_rng(random_state)
    return rng.random((M, coord_dim)).astype(np.float32)


def reduce_coords_random_proj(W, coord_dim, random_state=None):
    """
    Reduce prototype matrix W to coord_dim dimensions via random projection.

    Multiplies W by a random Gaussian matrix scaled by ``1 / sqrt(coord_dim)``
    (Johnson-Lindenstrauss style). Preserves approximate pairwise distances in
    expectation at zero computational cost — a structurally grounded starting
    layout that is nonetheless random and requires no decomposition.

    Parameters
    ----------
    W : np.ndarray, shape (M, d)
        Prototype matrix.
    coord_dim : int
        Target dimensionality (2 or 3).
    random_state : int or None, default None

    Returns
    -------
    coords : np.ndarray, shape (M, coord_dim), float32
    """
    rng = np.random.default_rng(random_state)
    d = W.shape[1]
    R = rng.standard_normal((d, coord_dim)).astype(np.float32) / np.sqrt(coord_dim)
    return (W.astype(np.float32) @ R)


_ANNEAL_EPS = 1e-8
# Small positive floor applied to initial and final when either is zero.
# Replaces the old additive-shift approach (which distorted the curve shape)
# with input clipping: zero values are replaced by _ANNEAL_EPS before the
# formula is applied, keeping the exponential a true exponential in value
# space rather than in (value + constant) space.


class ExponentialAnneal:
    """
    Exponential annealing schedule for a scalar parameter.

    Computes the annealed value at a given age (epoch count) as::

        value(age) = initial * (final / initial) ** (age / tau)

    The result is clipped at ``final`` once the schedule reaches it,
    ensuring the final value is respected exactly regardless of how many
    epochs have elapsed.

    Supports both decreasing schedules (``final < initial``, e.g. annealing
    a neighborhood bandwidth from broad to narrow) and increasing schedules
    (``final > initial``, e.g. shifting neighborhood influence from
    lattice-dominated toward CONN-dominated over training). The direction
    is inferred automatically from the relationship between ``initial``
    and ``final``; the formula and clipping behavior adapt accordingly.

    ``initial`` and ``final`` may be zero. If either is zero it is replaced
    internally by a small positive floor (``_ANNEAL_EPS = 1e-8``) before
    the formula is applied, keeping the exponential well-defined. The
    reported ``self.initial`` / ``self.final`` always reflect the values
    as supplied by the caller; only the internal computation is clipped.
    The resulting schedule is a true exponential in value space — not
    in a translated space — so the curve shape is never distorted by
    the zero-handling.

    Intended for use with any parameter that requires monotonic change
    during training — neighborhood bandwidth (rho), topology-blending
    weight (nbr_topo_alpha), learning rate, etc.

    Parameters
    ----------
    initial : float
        Starting value of the parameter (at age=0). Must be >= 0.
    final : float
        Target value of the parameter. Must be >= 0. If ``final <
        initial``, the schedule decays; if ``final > initial``, the
        schedule increases; if ``final == initial``, the schedule is
        flat and returns ``initial`` at every age.
    tau : float
        Time constant in epochs. Controls how quickly the parameter moves
        from ``initial`` toward ``final``. Smaller values change faster.
        Computed internally from ``halflife_epochs`` when that convenience
        constructor is used instead.

    Examples
    --------
    Decreasing schedule (decay):

    >>> schedule = ExponentialAnneal(initial=5.0, final=0.3, tau=20)
    >>> schedule(0)
    5.0
    >>> schedule(10)   # half a tau — midway through the decay
    1.2247...
    >>> schedule(20)   # one full tau — reaches final exactly
    0.3
    >>> schedule(1000)  # beyond tau — clipped at final
    0.3

    Decreasing to zero:

    >>> schedule = ExponentialAnneal(initial=1.0, final=0.0, tau=20)
    >>> schedule(0)
    1.0
    >>> schedule(1000)  # clipped at final
    0.0

    Increasing schedule (ramp):

    >>> schedule = ExponentialAnneal(initial=0.0, final=1.0, tau=20)
    >>> schedule(0)
    0.0
    >>> schedule(1000)  # clipped at final
    1.0

    Construction from a half-life epoch count:

    >>> schedule = ExponentialAnneal.from_halflife(initial=5.0, final=0.3,
    ...                                             halflife_epochs=50)
    >>> round(schedule(50), 6)  # geometric midpoint of 5.0 and 0.3
    1.2247...
    >>> round(schedule(100), 6)  # clipped at final
    0.3
    """

    def __init__(self, initial, final, tau):
        if initial < 0:
            raise ValueError(f"initial must be >= 0, got {initial}")
        if final < 0:
            raise ValueError(f"final must be >= 0, got {final}")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")

        self.initial = float(initial)
        self.final   = float(final)
        self.tau     = float(tau)

        # Internal clipped values: replace zero with _ANNEAL_EPS so that
        # the ratio (sf/si) and log(sf/si) are always well-defined.
        # Unlike the old additive-shift approach, this does not distort
        # the curve shape for non-zero parameters.
        self._si = max(self.initial, _ANNEAL_EPS)
        self._sf = max(self.final,   _ANNEAL_EPS)

        # Detect flat schedule (initial == final)
        self._flat = self.initial == self.final
        if self._flat:
            self._increasing = False
        else:
            self._increasing = self.final > self.initial

    @classmethod
    def from_halflife(cls, initial, final, halflife_epochs):
        """
        Construct a schedule where ``halflife_epochs`` is the geometric half-life.

        ``halflife_epochs`` is the epoch at which the parameter reaches the
        **geometric mean** of ``initial`` and ``final``::

            value(halflife_epochs) = sqrt(initial * final)

        This is a half-life in **log space** (the schedule is linear in
        log space, and the geometric mean is the arithmetic midpoint of
        ``log(initial)`` and ``log(final)``), not in linear space.

        .. note:: **Geometric vs arithmetic midpoint**

            The geometric midpoint ``sqrt(initial * final)`` is always
            closer to ``final`` than the arithmetic midpoint
            ``(initial + final) / 2`` when ``initial != final``. For
            example, with ``initial=10, final=0.1``:

            - Arithmetic midpoint: 5.05  (barely moved from initial)
            - Geometric midpoint:  1.0   (already close to final)

            This means the schedule will feel "fast then slow" in linear
            space — the parameter drops quickly early on and then settles
            gradually. If you expect to be "halfway there" in the ordinary
            (arithmetic) sense at ``halflife_epochs``, you will find the
            parameter has already moved further than that.

        .. note:: **Relationship to tau and total training epochs**

            ``tau = 2 * halflife_epochs``, and the schedule is clipped at
            ``final`` at age ``2 * halflife_epochs``. This means:

            - Setting ``halflife_epochs ≈ n_epochs / 4`` places the
              geometric midpoint at the first quarter of training, giving
              a steep initial drop followed by a long flat tail — the
              classic exponential shape.
            - Setting ``halflife_epochs ≈ n_epochs / 2`` places the
              geometric midpoint at the halfway point, so the schedule is
              still actively decaying at the end of training and the curve
              looks nearly linear over the training window.

            As a rule of thumb, set ``halflife_epochs`` to roughly a
            quarter of your total training epochs to get a visually
            exponential decay with a clear knee and flat tail.

        Parameters
        ----------
        initial : float
            Starting value at age=0.
        final : float
            Floor value; clipped at this value from age
            ``2 * halflife_epochs`` onward.
        halflife_epochs : float
            The epoch at which the parameter reaches the geometric mean
            ``sqrt(initial * final)``. Must be positive.

        Returns
        -------
        ExponentialAnneal

        Examples
        --------
        >>> s = ExponentialAnneal.from_halflife(initial=3.0, final=0.3,
        ...                                      halflife_epochs=50)
        >>> round(s(0), 4)
        3.0
        >>> round(s(50), 4)   # geometric midpoint sqrt(3.0 * 0.3) ≈ 0.9487
        0.9487
        >>> round(s(100), 4)  # clipped at final
        0.3

        With halflife_epochs = n_epochs / 4 for a classic exponential shape:

        >>> s = ExponentialAnneal.from_halflife(initial=10.0, final=0.1,
        ...                                      halflife_epochs=25)
        >>> round(s(25), 4)   # geometric midpoint sqrt(10 * 0.1) = 1.0
        1.0
        >>> round(s(50), 4)   # clipped at final well before end of training
        0.1
        """
        if halflife_epochs <= 0:
            raise ValueError(
                f"halflife_epochs must be positive, got {halflife_epochs}"
            )
        # tau such that age=halflife_epochs gives the geometric midpoint:
        # initial * (final/initial)^(halflife_epochs/tau) = sqrt(initial*final)
        # => (final/initial)^(halflife_epochs/tau) = sqrt(final/initial)
        # => halflife_epochs/tau = 0.5
        # => tau = 2 * halflife_epochs
        tau = 2.0 * halflife_epochs
        return cls(initial=initial, final=final, tau=tau)

    def __call__(self, age):
        """
        Return the annealed parameter value at the given age.

        Parameters
        ----------
        age : int or float
            Current epoch count (0-based).

        Returns
        -------
        float
        """
        if self._flat:
            return self.final
        raw = self._si * (self._sf / self._si) ** (age / self.tau)
        if self._increasing:
            return float(min(raw, self.final))
        else:
            return float(max(raw, self.final))

    def values(self, n_epochs):
        """
        Return the annealed values for epochs 0 through n_epochs - 1.

        Useful for inspecting or plotting the planned schedule before
        training begins.

        Parameters
        ----------
        n_epochs : int

        Returns
        -------
        np.ndarray, shape (n_epochs,)
        """
        if self._flat:
            return np.full(n_epochs, self.final)
        ages = np.arange(n_epochs, dtype=float)
        raw  = self._si * (self._sf / self._si) ** (ages / self.tau)
        if self._increasing:
            return np.minimum(raw, self.final).astype(float)
        else:
            return np.maximum(raw, self.final).astype(float)

    def __repr__(self):
        if self._flat:
            return (
                f"ExponentialAnneal("
                f"initial={self.initial}, final={self.final}, "
                f"tau={self.tau:.4g}, flat=True)"
            )
        return (
            f"ExponentialAnneal("
            f"initial={self.initial}, "
            f"final={self.final}, "
            f"tau={self.tau:.4g})"
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self):
        """
        Return a plain dict representation of this schedule.

        Returns
        -------
        dict with keys: initial (float), final (float), tau (float)
        """
        return {
            "initial": self.initial,
            "final":   self.final,
            "tau":     self.tau,
        }

    @classmethod
    def from_dict(cls, d):
        """
        Reconstruct an ExponentialAnneal from a dict produced by
        :meth:`to_dict`.

        Parameters
        ----------
        d : dict
            As returned by :meth:`to_dict`.

        Returns
        -------
        ExponentialAnneal
        """
        return cls(initial=d["initial"], final=d["final"], tau=d["tau"])