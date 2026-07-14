"""Equiformer encoders for contrastive pharmacophore learning."""

from .equiformer_adj import EquiformerAdjContrastive
from .equiformer_official import EquiformerOfficialContrastive

METHODS = {
    "equiformer_adj": EquiformerAdjContrastive,
    "equiformer_official": EquiformerOfficialContrastive,
}
