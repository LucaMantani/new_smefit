import copy
import importlib
import logging
import pathlib
import sys

from smefit.chi2 import Chi2

log = logging.getLogger(__name__)


def load_external_chi2(external_chi2, coefficients, rge_dict):
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

        extra_keys = {key: value for key, value in module.items() if key != "path"}

        chi2_ext = my_chi2_class(
            coefficients=coefficients, rge_dict=copy.deepcopy(rge_dict), **extra_keys
        )

        ext_chi2_modules.append(
            Chi2(
                chi2_ext.compute_chi2,
                param_names=coefficients.free_names,
                num_data=chi2_ext.num_data,
            )
        )

    return ext_chi2_modules
