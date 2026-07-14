#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import tnt


def read_config() -> tnt.Configuration:
    config = tnt.Configuration()
    config.read(Path(__file__).with_name("configuration.yaml"))
    return config


if __name__ == "__main__":
    config = read_config()
    config.print()
