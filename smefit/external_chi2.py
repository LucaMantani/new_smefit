import copy
import importlib
import inspect
import logging
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
    """
    # dynamical import
    ext_chi2_modules = []

    for class_name, module in external_chi2.items():
        log.info("Loading external chi2 module: %s", class_name)

        module_path = module["path"]
        path = pathlib.Path(module_path)
        base_path, stem = path.parent, path.stem
        sys.path = [str(base_path)] + sys.path
        try:
            chi2_module = importlib.import_module(stem)
        except ModuleNotFoundError:
            log.error("Module %s not found in %s. Adjust and rerun.", stem, base_path)
            raise ModuleNotFoundError(
                f"Module {stem} not found in {base_path}. Adjust and rerun."
            )

        my_chi2_class = getattr(chi2_module, class_name)

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

        chi2_ext = my_chi2_class(
            coefficients=coefficients, rge_dict=copy.deepcopy(rge_dict), **extra_keys
        )

        ext_chi2_modules.append(
            Chi2(
                chi2_ext.compute_chi2,
                param_names=coefficients.free_names,
                num_data=chi2_ext.num_data,
                name=class_name,
            )
        )

    return ext_chi2_modules
