#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import tnt


if __name__ == "__main__":
    config_path = Path(__file__).with_name("configuration.yaml")
    with tnt.configuration_session(config_path) as config:
        print("#" * 30 + " Configuration " + "#" * 30)
        config.print()
        print(f"{type(config) = }")
        print(f"{config.data['io_settings'] = }")
        print("#" * 30 + " Configuration as dict " + "#" * 30)
        print(f"{config.as_dict() = }")
        print(f"{type(config.as_dict()) = }")
        print(f"{config.as_dict()['io_settings'] = }")
        print("" + "#" * 30 + " Configuration as portable dict " + "#" * 30)
        print(f"{config.as_portable_dict() = }")
        print(f"{type(config.as_portable_dict()) = }")
        print(f"{config.as_portable_dict()['io_settings'] = }")