# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import os.path as osp
import time

from pandas.core.base import NoNewAttributesMixin

from smartsim.error.errors import SmartSimError

from ..utils import get_logger
from .entity import SmartSimEntity

logger = get_logger(__name__)


class DBNode(SmartSimEntity):
    """DBNode objects are the entities that make up the orchestrator.
    Each database node can be launched in a cluster configuration
    and take launch multiple databases per node.

    To configure how each instance of the database operates, look
    into the smartsimdb.conf.
    """

    def __init__(self, name, path, run_settings, ports, hostname_converter=None):
        """Initialize a database node within an orchestrator."""
        self.ports = ports
        self._host = None
        super().__init__(name, path, run_settings)
        self._multihost = False
        self._shard_ids = None
        self._hosts = None
        self.hostname_converter = hostname_converter

    @property
    def host(self):
        if not self._host:
            self._host = self._parse_db_host()
        return self._host

    @property
    def hosts(self):
        if not self._hosts:
            self._hosts = self._parse_db_hosts()
        return self._hosts

    def set_host(self, host):
        self._host = str(host)

    def set_hosts(self, hosts):
        self._hosts = [str(host) for host in hosts]

    def remove_stale_dbnode_files(self):
        """This function removes the .conf, .err, and .out files that
        have the same names used by this dbnode that may have been
        created from a previous experiment execution.
        """

        for port in self.ports:
            if not self._multihost:
                conf_file = osp.join(self.path, self._get_cluster_conf_filename(port))
                if osp.exists(conf_file):
                    os.remove(conf_file)
            else:
                conf_files = [
                    osp.join(self.path, filename)
                    for filename in self._get_cluster_conf_filenames(port)
                ]
                for conf_file in conf_files:
                    if osp.exists(conf_file):
                        os.remove(conf_file)

        for file_ending in [".err", ".out", ".mpmd"]:
            file_name = osp.join(self.path, self.name + file_ending)
            if osp.exists(file_name):
                os.remove(file_name)

    def _get_cluster_conf_filename(self, port):
        """Returns the .conf file name for the given port number

        :param port: port number
        :type port: int
        :return: the dbnode configuration file name
        :rtype: str
        """
        return "".join(("nodes-", self.name, "-", str(port), ".conf"))

    def _get_cluster_conf_filenames(self, port):
        """Returns the .conf file name for the given port number

        This function should bu used if and only if ``_multihost==True``

        :param port: port number
        :type port: int
        :return: the dbnode configuration file name
        :rtype: str
        """
        return [
            "".join(("nodes-", self.name + f"_{shard_id}", "-", str(port), ".conf"))
            for shard_id in self._shard_ids
        ]

    def _parse_db_host(self):
        """Parse the database host/IP from the output file

        this uses the RedisIP module that is built as a dependency
        The IP address is preferred, but if hostname is only present
        then a lookup to /etc/hosts is done through the socket library

        :raises SmartSimError: if host/ip could not be found
        :return: ip address | hostname
        :rtype: str
        """
        filepath = osp.join(self.path, self.name + ".out")
        trials = 5
        host = None
        ip = None

        # try a few times to give the database files time to
        # populate on busy systems.
        while not host and trials > 0:
            try:
                with open(filepath, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        content = line.split()
                        if "Hostname:" in content:
                            host = content[-1]
                        if "IP:" in content:
                            ip = content[-1]
                            break
            except FileNotFoundError:
                logger.debug("Waiting for RedisIP files to populate...")
                trials -= 1
                time.sleep(5)
            logger.debug("Waiting for RedisIP files to populate...")
            trials -= 1
            time.sleep(5)

        if not host and not ip:
            logger.error("RedisIP address lookup strategy failed.")
            raise SmartSimError("Failed to obtain database hostname")

        if self.hostname_converter:
            # Very unlikely case. In this case, the hostname_converter
            # should take IPs as input
            if ip and not host:
                host = ip
            host = self.hostname_converter[host]
        else:
            # prefer the ip address if present
            # TODO do socket lookup and ensure IP address matches
            # in the case where compute node returns 127.0.0.1 for its
            # own IP address
            if host and ip:
                host = ip

        return host

    def _parse_db_hosts(self):
        """Parse the database hosts/IPs from the output files

        this uses the RedisIP module that is built as a dependency
        The IP address is preferred, but if hostname is only present
        then a lookup to /etc/hosts is done through the socket library.
        This function must be called only if ``_multihost==True``.

        :raises SmartSimError: if host/ip could not be found
        :return: ip addresses | hostnames
        :rtype: list[str]
        """
        hosts = []

        for shard_id in self._shard_ids:
            trials = 5
            host = None
            ip = None
            filepath = osp.join(self.path, self.name + f"_{shard_id}.out")
            # try a few times to give the database files time to
            # populate on busy systems.
            while not host and trials > 0:
                try:
                    with open(filepath, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            content = line.split()
                            if "Hostname:" in content:
                                host = content[-1]
                            if "IP:" in content:
                                ip = content[-1]
                                break
                except FileNotFoundError:
                    logger.debug("Waiting for RedisIP files to populate...")
                    trials -= 1
                    time.sleep(5)
                logger.debug("Waiting for RedisIP files to populate...")
                trials -= 1
                time.sleep(5)

            if not host and not ip:
                logger.error("RedisIP address lookup strategy failed.")
                raise SmartSimError("Failed to obtain database hostname")

            if self.hostname_converter:
                # Very unlikely case. In this case, the hostname_converter
                # should take IPs as input
                if ip and not host:
                    host = ip
                host = self.hostname_converter(host)
            else:
                # prefer the ip address if present
                # TODO do socket lookup and ensure IP address matches
                # in the case where compute node returns 127.0.0.1 for its
                # own IP address
                if host and ip:
                    host = ip
            hosts.append(host)

        hosts = list(dict.fromkeys(hosts))

        return hosts

