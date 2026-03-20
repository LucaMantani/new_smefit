"""
api.py

This module contains the `reportengine` programmatic API, initialized with the
smefit providers, Config and Environment.

"""

import logging

from reportengine import api

from smefit.app import smefit_providers
from smefit.config import smefitConfig
from smefit.environment import smefitEnvironment

log = logging.getLogger(__name__)

smefitAPI = api.API(smefit_providers, smefitConfig, smefitEnvironment)
