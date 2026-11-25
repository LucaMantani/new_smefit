import copy
import importlib
import inspect
import logging
import numbers
import pathlib
import sys

from reportengine.configparser import ConfigError

from smefit.chi2 import Chi2

log = logging.getLogger(__name__)

# Keys of an `external_chi2` entry that smefit reads itself and does not forward
# to the chi2 class: `path` locates its module, `group` is how reports aggregate
# it (read by produce_data_groups). Every other key is a constructor argument.
_EXT_CHI2_RESERVED_KEYS = frozenset({"path", "group"})


def _accepts_max_scale(chi2_class):
    """Whether the constructor of ``chi2_class`` takes a ``max_scale`` argument."""
    params = inspect.signature(chi2_class).parameters.values()
    return any(
        p.name == "max_scale" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params
    )


def _check_interface(chi2_ext, class_name):
    """Raise unless ``chi2_ext`` exposes what smefit reads from an external chi2.

    Checked here, at load time, so that a plugin missing either attribute fails
    with its name.
    """
    compute = getattr(chi2_ext, "compute_chi2", None)
    if not callable(compute):
        raise ValueError(
            f"External chi2 {class_name} has no callable compute_chi2 method."
        )
    num_data = getattr(chi2_ext, "num_data", None)
    # numbers.Integral admits numpy integers, which a module counting its data
    # with len() of an array may well return; bool is excluded explicitly
    if not isinstance(num_data, numbers.Integral) or isinstance(num_data, bool):
        raise ValueError(
            f"External chi2 {class_name} must set num_data to an integer, "
            f"got {num_data!r}."
        )


def load_external_chi2(external_chi2, coefficients, rge_dict, max_scale=None):
    """
    Loads the external chi2 modules.
    We assume that the external chi2 only need to know which coefficients we want to
    fit and the RGE dictionary, which specifies the reference scale and the theory settings
    for the RGE evolution.

    Parameters
    ----------
    external_chi2: dict
        dict of external chi2s.
        Each key is a dictionary with the name of the chi2 class, specifying the
        path to the module and the parameters to be passed to the chi2 class.
    coefficients: CoefficientManager
        The coefficient manager object.
    rge_dict: dict
        The RGE dictionary, specifying the reference scale and theory settings.
    max_scale: float or None
        The runcard's global energy-scale cut (GeV). An entry's own
        ``max_scale`` overrides it, ``null`` exempting the entry, exactly as on
        ``datasets``. The resolved cut is passed to the constructor as
        ``max_scale=`` -- only when there is one, so a module with no data to
        cut needs no such argument -- and the module applies it to its own
        data, most simply by loading it through ``smefitAPI.data`` and
        ``smefitAPI.theory`` with the cut on its dataset entry. A cut reaching
        a module whose constructor does not take it raises rather than leave
        it uncut.

    Returns
    -------
    ext_chi2_modules: list
         List of external chi2 objects that can be evaluated by passing a coefficients instance

    Raises
    ------
    ValueError
        When a module cannot be imported, does not define the class, or its
        constructor raises, chained to the original exception; and when the
        object built does not expose a callable ``compute_chi2`` and an integer
        ``num_data``.
    """
    # dynamical import
    ext_chi2_modules = []

    for class_name, module in external_chi2.items():
        log.info("Loading external chi2 module: %s", class_name)

        if "path" not in module:
            raise ValueError(
                f"external chi2 {class_name} has no 'path' to the module defining it."
            )
        path = pathlib.Path(module["path"])
        base_path, stem = path.parent, path.stem
        sys.path = [str(base_path)] + sys.path
        try:
            chi2_module = importlib.import_module(stem)
        except Exception as exc:
            raise ValueError(
                f"Could not import the module {stem} of external chi2 "
                f"{class_name} from {base_path}: {exc!r}"
            ) from exc

        try:
            my_chi2_class = getattr(chi2_module, class_name)
        except AttributeError as exc:
            raise ValueError(
                f"The module {path} does not define the external chi2 class "
                f"{class_name}."
            ) from exc

        extra_keys = {
            key: value
            for key, value in module.items()
            if key not in _EXT_CHI2_RESERVED_KEYS
        }
        # the entry's own cut (null: exempt), else the global one; passed on
        # only when there is one
        cut = extra_keys.pop("max_scale", max_scale)
        if cut is not None:
            if not _accepts_max_scale(my_chi2_class):
                raise ConfigError(
                    f"max_scale {cut:g} applies to external chi2 {class_name}, "
                    "whose constructor takes no max_scale argument, so it cannot "
                    "be cut. Implement the cut in the module, or exempt it with "
                    "max_scale: null on its entry."
                )
            extra_keys["max_scale"] = cut

        try:
            chi2_ext = my_chi2_class(
                coefficients=coefficients,
                rge_dict=copy.deepcopy(rge_dict),
                **extra_keys,
            )
        except Exception as exc:
            raise ValueError(
                f"The constructor of external chi2 {class_name} raised {exc!r}."
            ) from exc
        _check_interface(chi2_ext, class_name)

        ext_chi2_modules.append(
            Chi2(
                chi2_ext.compute_chi2,
                param_names=coefficients.free_names,
                num_data=chi2_ext.num_data,
                name=class_name,
            )
        )

    return ext_chi2_modules
