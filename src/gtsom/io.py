"""
io.py — HDF5 persistence layer for the gtsom package.

Provides two module-level recursive helpers (_write_dict, _read_dict) that
serialize/deserialize arbitrary nested dicts whose values may be dense arrays,
sparse CSR matrices, ragged list-of-lists, label arrays (object arrays with
None entries), scalars, None, or further nested dicts.

The GTSOMio class uses these helpers to save and load fully fitted GTSOM
instances. It is the single owner of the HDF5 file structure; no other module
in gtsom imports h5py or knows about the file layout.

Typical usage
-------------
Save a fitted GTSOM::

    GTSOMio(som).save("som.h5")

Or via the convenience wrapper on GTSOM itself::

    som.save("som.h5")

Load back::

    som = GTSOMio.load("som.h5")
    # or equivalently:
    som = GTSOM.load("som.h5")

Type dispatch
-------------
_write_dict inspects each value and tags it with a ``_type`` HDF5 attribute
so that _read_dict can reconstruct the correct Python type without any
external schema:

    dense        np.ndarray            → dataset   + _type='dense'
    sparse       scipy.sparse.csr      → subgroup  + _type='sparse'
    ragged       list of lists         → subgroup  + _type='ragged'
    label_array  object ndarray/Nones  → dataset   + _type='label_array'
    dict         nested dict           → subgroup  + _type='dict'
    none         None                  → attr       _type='none'
    scalar       int/float/str/bool    → attr  (no _type needed; HDF5 preserves type)
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import issparse, csr_matrix

__all__ = ["GTSOMio"]

# ---------------------------------------------------------------------------
# Type tag constants
# ---------------------------------------------------------------------------

_T_DENSE  = "dense"
_T_SPARSE = "sparse"
_T_RAGGED = "ragged"
_T_LABEL  = "label_array"
_T_DICT   = "dict"
_T_NONE   = "none"


# ---------------------------------------------------------------------------
# Module-level recursive helpers
# ---------------------------------------------------------------------------

def _write_dict(group, d):
    """
    Recursively write a dict ``d`` into an open h5py Group ``group``.

    Each value is inspected and tagged with a ``_type`` attribute so that
    _read_dict can reconstruct the correct Python type on load.  Keys must
    be strings.

    Supported value types
    ---------------------
    np.ndarray          → HDF5 dataset, _type='dense'
    scipy sparse CSR    → HDF5 subgroup, _type='sparse'
    list of lists       → HDF5 subgroup, _type='ragged'
    object ndarray      → HDF5 dataset,  _type='label_array'  (WL-style arrays
                          where entries are scalars or None)
    dict                → HDF5 subgroup, _type='dict'  (recursive)
    None                → HDF5 attr '_type'='none'  on a small subgroup
    scalar              → HDF5 attr (int, float, str, bool; no _type needed)

    Parameters
    ----------
    group : h5py.Group
        Open, writable HDF5 group to write into.
    d : dict
        Dict whose values are of the supported types above.
    """
    import h5py

    for key, value in d.items():
        key = str(key)

        # ---- None --------------------------------------------------------
        if value is None:
            sg = group.create_group(key)
            sg.attrs["_type"] = _T_NONE

        # ---- nested dict -------------------------------------------------
        elif isinstance(value, dict):
            sg = group.create_group(key)
            sg.attrs["_type"] = _T_DICT
            _write_dict(sg, value)

        # ---- sparse CSR --------------------------------------------------
        elif issparse(value):
            sg = group.create_group(key)
            sg.attrs["_type"] = _T_SPARSE
            csr = value.tocsr()
            sg.create_dataset("data",    data=csr.data)
            sg.create_dataset("indices", data=csr.indices)
            sg.create_dataset("indptr",  data=csr.indptr)
            sg.attrs["shape"] = list(csr.shape)
            sg.attrs["dtype"] = str(csr.dtype)

        # ---- object ndarray (label array: entries are scalars or None) ---
        elif isinstance(value, np.ndarray) and value.dtype == object:
            _write_label_array(group, key, value)

        # ---- dense ndarray -----------------------------------------------
        elif isinstance(value, np.ndarray):
            ds = group.create_dataset(key, data=value)
            ds.attrs["_type"] = _T_DENSE

        # ---- ragged list-of-lists ----------------------------------------
        elif (
            isinstance(value, list)
            and len(value) > 0
            and isinstance(value[0], (list, np.ndarray))
        ):
            sg = group.create_group(key)
            sg.attrs["_type"] = _T_RAGGED
            flat = (
                np.concatenate([np.asarray(row, dtype=np.int64) for row in value])
                if any(len(r) > 0 for r in value)
                else np.array([], dtype=np.int64)
            )
            lengths = np.array([len(r) for r in value], dtype=np.int64)
            indptr  = np.concatenate([[0], np.cumsum(lengths)])
            sg.create_dataset("flat",   data=flat)
            sg.create_dataset("indptr", data=indptr)

        # ---- scalar (int, float, str, bool, tuple used as shape, etc.) ---
        else:
            # Tuples (e.g. _shape) and lists of scalars become attrs directly.
            # h5py stores Python int/float/str/bool natively; numpy scalars
            # are cast to Python builtins first to avoid h5py dtype surprises.
            if isinstance(value, (tuple, list)):
                group.attrs[key] = list(value)
            elif isinstance(value, (np.integer,)):
                group.attrs[key] = int(value)
            elif isinstance(value, (np.floating,)):
                group.attrs[key] = float(value)
            elif isinstance(value, (np.bool_,)):
                group.attrs[key] = bool(value)
            else:
                group.attrs[key] = value


def _write_label_array(group, key, arr):
    """
    Write an object ndarray whose entries are scalars or None as a
    vlen-string HDF5 dataset, recording the element type so it can be
    cast back correctly on load.

    None entries are stored as the empty string "".

    Parameters
    ----------
    group : h5py.Group
    key : str
    arr : np.ndarray, dtype=object
    """
    import h5py

    non_none = [v for v in arr if v is not None]
    elem_type = type(non_none[0]).__name__ if non_none else "str"

    str_arr = np.array(
        ["" if v is None else str(v) for v in arr],
        dtype=h5py.special_dtype(vlen=str),
    )
    ds = group.create_dataset(key, data=str_arr)
    ds.attrs["_type"]     = _T_LABEL
    ds.attrs["elem_type"] = elem_type


def _read_dict(group):
    """
    Recursively read an h5py Group written by _write_dict and return a dict
    with values restored to their original Python/numpy types.

    Parameters
    ----------
    group : h5py.Group
        Open HDF5 group previously written by _write_dict.

    Returns
    -------
    dict
    """
    import h5py

    result = {}

    # --- attrs (scalars written directly, no _type tag) -------------------
    for key, value in group.attrs.items():
        if key == "_type":
            continue   # group-level type tag, not a data field
        # h5py may return numpy scalars; cast to Python builtins
        if isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, np.floating):
            result[key] = float(value)
        elif isinstance(value, np.bool_):
            result[key] = bool(value)
        elif isinstance(value, np.ndarray):
            # attrs that are lists (e.g. _shape stored as list)
            result[key] = value.tolist()
        elif isinstance(value, bytes):
            result[key] = value.decode()
        else:
            result[key] = value

    # --- datasets and subgroups -------------------------------------------
    for key in group.keys():
        item = group[key]

        if isinstance(item, h5py.Dataset):
            _type = item.attrs.get("_type", _T_DENSE)

            if _type == _T_DENSE:
                result[key] = item[:]

            elif _type == _T_LABEL:
                result[key] = _read_label_array(item)

        elif isinstance(item, h5py.Group):
            _type = item.attrs.get("_type", _T_DICT)

            if _type == _T_NONE:
                result[key] = None

            elif _type == _T_DICT:
                result[key] = _read_dict(item)

            elif _type == _T_SPARSE:
                result[key] = csr_matrix(
                    (item["data"][:], item["indices"][:], item["indptr"][:]),
                    shape=tuple(item.attrs["shape"]),
                )

            elif _type == _T_RAGGED:
                flat   = item["flat"][:]
                indptr = item["indptr"][:]
                result[key] = [
                    flat[indptr[i]:indptr[i + 1]].tolist()
                    for i in range(len(indptr) - 1)
                ]

    return result


def _read_label_array(dataset):
    """
    Reconstruct an object ndarray from a vlen-string HDF5 dataset written
    by _write_label_array.  Empty strings are restored as None; other
    entries are cast to the recorded elem_type.

    Parameters
    ----------
    dataset : h5py.Dataset

    Returns
    -------
    np.ndarray, dtype=object
    """
    raw       = dataset[:]
    elem_type = str(dataset.attrs.get("elem_type", "str"))
    result    = np.empty(len(raw), dtype=object)

    for i, v in enumerate(raw):
        v_str = v.decode() if isinstance(v, bytes) else str(v)
        if v_str == "":
            result[i] = None
        elif "int" in elem_type:
            result[i] = int(v_str)
        elif "float" in elem_type:
            result[i] = float(v_str)
        else:
            result[i] = v_str

    return result


# ---------------------------------------------------------------------------
# GTSOMio
# ---------------------------------------------------------------------------

class GTSOMio:
    """
    HDF5 save/load manager for GTSOM instances.

    Owns the entire HDF5 file structure.  No other module in gtsom imports
    h5py or knows about the file layout.

    Parameters
    ----------
    som : GTSOM
        A fitted (or partially fitted) GTSOM instance.  Only required for
        :meth:`save`; use :meth:`load` as a classmethod to reconstruct.

    Examples
    --------
    Save::

        GTSOMio(som).save("som.h5")

    Load::

        som = GTSOMio.load("som.h5")
    """

    # HDF5 top-level group names — single source of truth for file layout
    _GRP_META      = "meta"
    _GRP_PARAMS    = "params"
    _GRP_STATE     = "state"
    _GRP_ARCH      = "arch"
    _GRP_RECALLER  = "recaller"
    _GRP_CTAB      = "ctab"
    _GRP_HISTORY   = "history"
    _EPOCH_FMT     = "epoch_{:06d}"

    def __init__(self, som):
        self._som = som

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, path):
        """
        Save the GTSOM instance to an HDF5 file at ``path``.

        File layout
        -----------
        ::

            som.h5
            ├── meta/          version string and class tag
            ├── params/        learning hyperparameters (schedules, etc.)
            ├── state/         mutable training state (age, train_time)
            ├── arch/
            │   ├── W, W0, coords0, prevBMU, nbr_W  (arrays)
            │   ├── embedding/     Embedding.to_dict() contents
            │   ├── kernel/        NeighborKernel scalars
            │   ├── validator/     DataValidator.to_dict() contents (if present)
            │   └── recaller/      VQRecaller.to_dict() contents
            ├── ctab/          color table (if present)
            └── history/
                ├── (attr) n_snapshots
                ├── epoch_000000/  snapshot dict contents
                ├── epoch_000001/
                └── ...

        Parameters
        ----------
        path : str
            Destination file path.  Created or overwritten.
        """
        import h5py
        with h5py.File(path, "w") as f:
            self._write_to_group(f)

    @classmethod
    def load(cls, path):
        """
        Load a GTSOM from an HDF5 file written by :meth:`save`.

        Parameters
        ----------
        path : str
            Path to the HDF5 file.

        Returns
        -------
        GTSOM
            Fully reconstructed instance, ready for :meth:`~GTSOM.transform`,
            :meth:`~GTSOM.plot`, :meth:`~GTSOM.get_snapshot_fig`, or
            :meth:`~GTSOM.fit` (to resume training).
        """
        import h5py
        with h5py.File(path, "r") as f:
            return cls._load_from_group(f)

    # ------------------------------------------------------------------
    # Group-level API (for embedding into a larger HDF5 file)
    # ------------------------------------------------------------------

    def _write_to_group(self, group):
        """
        Write the GTSOM into an already-open h5py Group.

        Intended for use by external packages (e.g. soumap) that need to
        embed a GTSOM as a subgroup of a larger HDF5 file without going
        through a temporary file.  ``save()`` delegates to this method
        after opening the file.

        Parameters
        ----------
        group : h5py.Group
            Open, writable group to write into.  All GTSOM subgroups
            (meta, params, state, arch, ctab, history) are created
            as children of this group.
        """
        from importlib.metadata import version as _pkg_version

        som = self._som

        try:
            _version = _pkg_version("gtsom")
        except Exception:
            _version = "unknown"

        # --- meta ---------------------------------------------------------
        mg = group.create_group(self._GRP_META)
        mg.attrs["gtsom_version"] = _version
        mg.attrs["class"]         = "GTSOM"

        # --- params -------------------------------------------------------
        _write_dict(group.create_group(self._GRP_PARAMS),
                    self._build_params_dict(som))

        # --- state --------------------------------------------------------
        _write_dict(group.create_group(self._GRP_STATE),
                    self._build_state_dict(som))

        # --- arch ---------------------------------------------------------
        ag = group.create_group(self._GRP_ARCH)
        _write_dict(ag, self._build_arch_dict(som))

        # VQRecaller: written separately so its group structure is self-contained
        if som.recaller is not None:
            rg = ag.create_group(self._GRP_RECALLER)
            _write_dict(rg, som.recaller.to_dict())

        # --- ctab ---------------------------------------------------------
        if som.ctab is not None:
            _write_dict(group.create_group(self._GRP_CTAB),
                        self._build_ctab_dict(som.ctab))

        # --- history ------------------------------------------------------
        hg = group.create_group(self._GRP_HISTORY)
        hg.attrs["n_snapshots"] = len(som.learn_history_)
        for i, snap in enumerate(som.learn_history_):
            sg = hg.create_group(self._EPOCH_FMT.format(i))
            _write_dict(sg, self._build_snapshot_dict(snap))

    @classmethod
    def _load_from_group(cls, group):
        """
        Reconstruct a GTSOM from an already-open h5py Group.

        Intended for use by external packages (e.g. soumap) that embed a
        GTSOM as a subgroup of a larger HDF5 file.  ``load()`` delegates
        to this method after opening the file.

        Parameters
        ----------
        group : h5py.Group
            Open, readable group previously written by :meth:`_write_to_group`.

        Returns
        -------
        GTSOM
        """
        from gtsom.gtsom import GTSOM
        from gtsom.embedding import Embedding
        from gtsom.utils import ExponentialAnneal, DataValidator
        from gtsom.kernel import NeighborKernel
        from gtsom.dr_metrics import DRMetricsResult
        from vqlp import VQRecaller

        # --- params -------------------------------------------------------
        params = _read_dict(group[cls._GRP_PARAMS])
        state  = _read_dict(group[cls._GRP_STATE])
        arch   = _read_dict(group[cls._GRP_ARCH])

        # --- assemble GTSOM without calling __init__ ----------------------
        som = GTSOM.__new__(GTSOM)

        # Schedules
        som.rho_schedule   = ExponentialAnneal.from_dict(params["rho_schedule"])
        som.alpha_schedule = ExponentialAnneal.from_dict(params["alpha_schedule"])

        # Scalar hyperparameters
        som.nbr_influence_min  = params["nbr_influence_min"]
        som.proto_topo         = params["proto_topo"]
        som.compute_dr_metrics = params["compute_dr_metrics"]
        som.n_jobs             = params["n_jobs"]
        som.random_state       = params["random_state"]

        # Training state
        som.age        = state["age"]
        som.train_time = state["train_time"]

        # Architecture arrays
        som.W       = arch.get("W")
        som.W0      = arch.get("W0")
        som.coords0 = arch.get("coords0")
        som.prevBMU = arch.get("prevBMU")
        som.nbr_W   = arch.get("nbr_W")

        # Embedding
        som.embed = (Embedding.from_dict(arch["embedding"])
                     if "embedding" in arch else None)

        # NeighborKernel — reconstruct from saved scalars
        k = arch.get("kernel", {})
        som.kernel = (NeighborKernel(kind=k["kind"], h_min=k["h_min"])
                      if k else None)

        # DataValidator
        som._validator = (DataValidator.from_dict(arch["validator"])
                          if "validator" in arch else None)

        # VQRecaller
        arch_grp = group[cls._GRP_ARCH]
        som.recaller = (VQRecaller.from_dict(_read_dict(arch_grp[cls._GRP_RECALLER]))
                        if cls._GRP_RECALLER in arch_grp else None)

        # ctab
        som.ctab = None
        if cls._GRP_CTAB in group:
            ctab_d   = _read_dict(group[cls._GRP_CTAB])
            som.ctab = dict(zip(ctab_d["keys"], ctab_d["values"]))

        # history
        hg      = group[cls._GRP_HISTORY]
        n_snaps = int(hg.attrs.get("n_snapshots", 0))
        som.learn_history_ = [
            cls._restore_snapshot(
                _read_dict(hg[cls._EPOCH_FMT.format(i)]),
                DRMetricsResult,
            )
            for i in range(n_snaps)
        ]

        return som

    # ------------------------------------------------------------------
    # Private dict builders (save path)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_params_dict(som):
        return {
            "rho_schedule":   som.rho_schedule.to_dict(),
            "alpha_schedule": som.alpha_schedule.to_dict(),
            "nbr_influence_min":  som.nbr_influence_min,
            "proto_topo":         som.proto_topo,
            "compute_dr_metrics": som.compute_dr_metrics,
            "n_jobs":             som.n_jobs,
            "random_state":       som.random_state,
        }

    @staticmethod
    def _build_state_dict(som):
        return {
            "age":        som.age,
            "train_time": som.train_time,
        }

    @staticmethod
    def _build_arch_dict(som):
        d = {}
        for name in ("W", "W0", "coords0", "prevBMU", "nbr_W"):
            val = getattr(som, name)
            if val is not None:
                d[name] = val

        if som.embed is not None:
            d["embedding"] = som.embed.to_dict()

        if som.kernel is not None:
            d["kernel"] = {"kind": som.kernel.kind, "h_min": som.kernel.h_min}

        if som._validator is not None:
            d["validator"] = som._validator.to_dict()

        # recaller is written separately (own subgroup in save())
        return d

    @staticmethod
    def _build_ctab_dict(ctab):
        return {
            "keys":   np.array(list(ctab.keys()),   dtype=object),
            "values": np.array(list(ctab.values()), dtype=object),
        }

    @staticmethod
    def _build_snapshot_dict(snap):
        """
        Convert one learn_history_ snapshot dict into a write-ready dict.

        DRMetricsResult is serialized via its own to_dict(); all other
        values are passed through as-is (arrays, sparse matrices, scalars).
        """
        d = {
            "age":      snap["age"],
            "mqe":      snap["mqe"],
            "delBMU":   snap["delBMU"],
            "color_by": snap["color_by"],
            "W_mqe":    snap["W_mqe"],
            "coords":   snap["coords"],
            "RFSize":   snap["RFSize"],
            "adjacency": snap["adjacency"],
        }
        if snap["WL"] is not None:
            d["WL"] = snap["WL"]
        dm = snap.get("dr_metrics")
        if dm is not None:
            d["dr_metrics"] = dm.to_dict()
        return d

    # ------------------------------------------------------------------
    # Private snapshot restorer (load path)
    # ------------------------------------------------------------------

    @staticmethod
    def _restore_snapshot(d, DRMetricsResult):
        """
        Convert a raw dict (from _read_dict) back into a learn_history_
        snapshot dict, reconstructing DRMetricsResult if present.
        """
        snap = {
            "age":       d["age"],
            "mqe":       d["mqe"],
            "delBMU":    d["delBMU"],
            "color_by":  d["color_by"],
            "W_mqe":     d["W_mqe"],
            "coords":    d["coords"],
            "RFSize":    d["RFSize"],
            "adjacency": d["adjacency"],
            "WL":        d.get("WL"),
            "dr_metrics": (DRMetricsResult.from_dict(d["dr_metrics"])
                           if "dr_metrics" in d else None),
        }
        return snap
