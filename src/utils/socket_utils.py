"""Socket creation and connection utilities.

This module provides standardized helpers for creating and connecting different
types of sockets (Unix, TCP, UDP). It abstracts the boilerplate of socket
initialization and connection setup.
"""

from __future__ import annotations

import socket
from typing import Callable


def create_and_connect_unix_socket(
    socket_path: str,
    timeout_seconds: float = 5.0,
    socket_factory: Callable[..., socket.socket] | None = None,
) -> socket.socket:
    """Create a Unix domain stream socket and connect it to a path.

    Args:
        socket_path: The filesystem path to the Unix socket.
        timeout_seconds: The timeout for the connection attempt.
        socket_factory: Optional factory for creating the socket object (used for mocking).

    Returns:
        An open and connected Unix stream socket.
    """
    if socket_factory is None:
        socket_factory = socket.socket

    # 1. Create the socket with Unix address family and stream type
    unix_socket = socket_factory(socket.AF_UNIX, socket.SOCK_STREAM)
    
    # 2. Set the connection timeout
    unix_socket.settimeout(timeout_seconds)
    
    # 3. Connect to the specified path
    unix_socket.connect(socket_path)
    
    return unix_socket


def create_and_connect_tcp_socket(
    host: str,
    port: int,
    timeout_seconds: float = 5.0,
    socket_factory: Callable[..., socket.socket] | None = None,
) -> socket.socket:
    """Create a TCP stream socket and connect it to a host and port.

    Args:
        host: The destination hostname or IP address.
        port: The destination TCP port.
        timeout_seconds: The timeout for the connection attempt.
        socket_factory: Optional factory for creating the socket object (used for mocking).

    Returns:
        An open and connected TCP stream socket.
    """
    if socket_factory is None:
        socket_factory = socket.socket

    # 1. Create the socket with IPv4 address family and stream type
    tcp_socket = socket_factory(socket.AF_INET, socket.SOCK_STREAM)
    
    # 2. Set the connection timeout
    tcp_socket.settimeout(timeout_seconds)
    
    # 3. Connect to the host and port
    tcp_socket.connect((host, port))
    
    return tcp_socket


def create_udp_socket(
    socket_factory: Callable[..., socket.socket] | None = None,
) -> socket.socket:
    """Create an IPv4 UDP datagram socket.

    Args:
        socket_factory: Optional factory for creating the socket object (used for mocking).

    Returns:
        An open IPv4 UDP datagram socket.
    """
    if socket_factory is None:
        socket_factory = socket.socket

    # 1. Create the socket with IPv4 address family and datagram type
    udp_socket = socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
    
    return udp_socket
