REGISTRY = {}

from .basic_controller import BasicMAC

REGISTRY["basic_mac"] = BasicMAC

from .basic_controller_interactive import BasicMACInteractive, BasicMACInteractiveRegV1, BasicMACInteractiveRegV2

REGISTRY["basic_mac_interactive"] = BasicMACInteractive
REGISTRY["basic_mac_interactive_regv1"] = BasicMACInteractiveRegV1
REGISTRY["basic_mac_interactive_regv2"] = BasicMACInteractiveRegV2

from .basic_controller_influence import BasicMACInfluence

REGISTRY["basic_mac_influence"] = BasicMACInfluence