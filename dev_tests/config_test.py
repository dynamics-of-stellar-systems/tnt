#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import tnt


if __name__ == "__main__":
    config_path = Path(__file__).with_name("configuration.yaml")
    with tnt.configuration_session(config_path) as config:
        config.print()
